"""Single-frame privileged critic state; never appended to the actor input."""
from __future__ import annotations
import torch
from ref2act.isaac_compat import to_torch as _to_torch
from .standup_support import sphere_support_points

CRITIC_MODE = 'privileged_single_frame_v1'


def critic_terms(env, extractor, reset_mask=None):
    from ref2act.common.math import quat_apply, quat_apply_inverse
    data = env.robot.data
    n = env.num_envs
    q, _, linear, _ = extractor._anchor_state()
    pos = _to_torch(data.body_link_pos_w)
    anchor = pos[:, env.anchor_body_index]
    ids = [env.robot.body_names.index(f'{side}_ankle_roll_link') for side in ('left', 'right')]
    feet, fq = pos[:, ids], _to_torch(data.body_link_quat_w)[:, ids]
    origin_z = _to_torch(env.scene.env_origins)[:, 2]
    def body(v):
        shape = v.shape
        quat = q[:, None].expand(-1, shape[1], -1) if v.ndim == 3 else q
        return quat_apply_inverse(quat.reshape(-1,4), v.reshape(-1,3)).reshape(shape)
    corners = q.new_tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
    sole = quat_apply(fq[:, :, None].expand(-1,-1,4,-1).reshape(-1,4),
                      corners[None,None].expand(n,2,-1,-1).reshape(-1,3)).reshape(n,2,4,3)
    sole = sphere_support_points(sole + feet[:,:,None])
    normal = quat_apply(fq.reshape(-1,4), q.new_tensor([0.,0.,1.]).expand(n*2,-1)).reshape(n,2,3)
    mass = _to_torch(data.body_mass)
    com = (_to_torch(data.body_com_pos_w)*mass[...,None]).sum(1)/mass.sum(1,keepdim=True)
    com_vel = (_to_torch(data.body_com_lin_vel_w)*mass[...,None]).sum(1)/mass.sum(1,keepdim=True)
    root, shoulder, _, _, _, _, _ = env._recovery_state()
    zero = q.new_zeros(n)
    reset_mask = (torch.zeros(n, device=q.device, dtype=torch.bool) if reset_mask is None else reset_mask)
    def cached(name, width=1, clear=False):
        value = getattr(env, name, zero if width == 1 else q.new_zeros(n,width)).float().reshape(n,width)
        return torch.where(reset_mask[:,None], 0., value) if clear else value
    support = getattr(env, '_support_state', {})
    def contact(name):
        value = support.get(name, q.new_zeros(n,2)).float()
        return torch.where(reset_mask[:,None], 0., value)
    persistence = float(env.cfg.pair_standup_support_cfg['persistence_s'])
    processor = env.action_processor
    sampler=getattr(env,'_episode_sampler',None)
    horizon=env.max_episode_length if sampler is None else sampler.deadline.clamp_min(1)
    terms = {
        'proprioception': extractor.extract(),
        'joint_target': extractor.action_history_value(),
        'applied_action': env._sim_to_policy_order(processor.applied_action),
        'previous_applied_action': env._sim_to_policy_order(processor.previous_applied_action),
        'base_linear_velocity_b': linear,
        'root_shoulder_height_m': torch.stack((root,shoulder),-1),
        'feet_position_b': body(feet-anchor[:,None]).flatten(1),
        'feet_linear_velocity_b': body(_to_torch(data.body_link_lin_vel_w)[:,ids]).flatten(1),
        'sole_point_height_m': (sole[...,2]-origin_z[:,None,None]).flatten(1),
        'feet_normal_b': body(normal).flatten(1),
        'com_position_b': body(com-anchor),
        'com_linear_velocity_b': body(com_vel),
        'ground_foot_load_mg': contact('foot_load'),
        'usable_foot_load_mg': cached('_transfer_usable_foot_load',2,True),
        'foot_contact': contact('contact'),
        'contact_persistence': (cached('_support_contact_steps',2,True)*env.step_dt/persistence).clamp(0,1),
        'filtered_load': cached('_transfer_filtered_load',clear=True),
        'episode_peak_load': cached('_transfer_peak_load',clear=True),
        'assistance_maximum_mg': torch.full_like(zero, float(env.cfg.pair_standup_assistance_current_gravity_ratio))[:,None],
        'assistance_episode_draw': cached('_standup_assistance_draw'),
        'assistance_actual_mg': cached('_standup_assistance_actual_ratio',clear=True),
        'assistance_latched_off': cached('_standup_assistance_latched_off',clear=True),
        'assistance_orientation_gate': cached('_standup_assistance_orientation_multiplier',clear=True),
        'assistance_height_gate': cached('_standup_assistance_multiplier',clear=True),
        'episode_remaining': (1-env.episode_length_buf.float()/horizon).clamp(0,1)[:,None],
        'settling_remaining': ((env._standup_settling_steps()-env.episode_length_buf.float())/max(env._standup_settling_steps(),1)).clamp(0,1)[:,None],
        'success_hold': (cached('_standup_success_hold',clear=True)*env.step_dt/float(env.cfg.pair_standup_success_hold_s)).clamp(0,1),
        'success_latched': cached('_standup_success_latched',clear=True),
    }
    return terms


def pack_critic_terms(terms):
    layout, offset = {}, 0
    for name, value in terms.items():
        if value.ndim != 2:
            raise RuntimeError(f'Invalid stand-up critic observation term: {name}')
        layout[name] = [offset, offset+value.shape[1]]
        offset += value.shape[1]
    result = torch.cat(tuple(terms.values()),-1).float()
    if not torch.isfinite(result).all():
        invalid = [name for name, value in terms.items() if not torch.isfinite(value).all()]
        raise RuntimeError(f'Nonfinite stand-up critic observation terms: {invalid}')
    return result, layout


def actor_noise_settings(config=None):
    """Same per-component uniform noise bounds as Ref2Act locomotion."""
    import math
    config = {} if config is None else config
    settings = {'enabled': bool(config.get('enabled', False))}
    for key, default in (('orientation_6d',.05),('angular_velocity',.2),('joint_position',.01),('joint_velocity',.5)):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'Actor observation noise {key} must be finite and nonnegative.')
        settings[key] = value
    return settings


def noisy_actor_frame(frame, settings):
    if frame.shape[-1] != 55:
        raise ValueError('Actor proprioception must have 55 components.')
    if not settings['enabled']:
        return frame.clone()
    scale = frame.new_tensor([settings['orientation_6d']]*6 + [settings['angular_velocity']]*3
                             + [settings['joint_position']]*23 + [settings['joint_velocity']]*23)
    # Sample once on acquisition; old history entries retain their measured noise.
    return frame + (2*torch.rand_like(frame)-1)*scale

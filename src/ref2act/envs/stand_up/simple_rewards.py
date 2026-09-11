"""V13: three objectives, three penalties, and optional gated posture credit.

Contact/placement geometry lives in standup_stages. No legacy recovery reward,
completion bonus, extra hold bonus, or hidden regularization multiplier is used.
"""
import math
import torch
from ref2act.isaac_compat import to_torch
from .standup_observation import critic_terms, pack_critic_terms


def validate_simple_rewards(cfg):
    expected = {'mode','action_rate_weight','joint_limit_weight',
                'soft_joint_position_limit','failure_penalty'}
    if 'hip_collision_weight' in cfg:
        if not math.isfinite(float(cfg['hip_collision_weight'])) or cfg['hip_collision_weight']<0:
            raise ValueError('Invalid hip collision weight')
        cfg={k:v for k,v in cfg.items() if k!='hip_collision_weight'}
    if 'posture' in cfg:
        from .posture import validate_posture
        validate_posture(cfg['posture'])
        cfg = {k:v for k,v in cfg.items() if k != 'posture'}
    if set(cfg) != expected or cfg['mode'] != 'simple_v13':
        raise ValueError('Invalid simple_v13 reward configuration')
    for key in expected-{'mode'}:
        if not math.isfinite(float(cfg[key])) or cfg[key] < 0:
            raise ValueError(f'Invalid reward setting {key}')
    if not 0 < cfg['soft_joint_position_limit'] <= 1:
        raise ValueError('soft_joint_position_limit must be in (0,1]')


def compose_rewards(stage_terms, *, action, previous_action, joint_position,
                    joint_limits, finite, failed, settling, cfg, dt, posture_reward=None, hip_collision_cost=None):
    """All continuous terms use seconds; failure is a one-time event penalty."""
    active = finite & ~failed & ~settling
    terms = {key: torch.where(active, stage_terms[key], 0.)
             for key in ('preparation','rise','stand')}
    action_rate = (action-previous_action).square().mean(-1)
    lower, upper = joint_limits.unbind(-1)
    normalized = (2*joint_position-lower-upper)/(upper-lower).clamp_min(1.e-6)
    excess = (normalized.abs()-cfg['soft_joint_position_limit']).clamp_min(0)
    terms['action_rate'] = torch.where(active,-dt*cfg['action_rate_weight']*action_rate,0.)
    terms['joint_limit'] = torch.where(active,-dt*cfg['joint_limit_weight']*excess.square().mean(-1),0.)
    terms['failure'] = -(failed | ~finite).to(action.dtype)*cfg['failure_penalty']
    if 'posture' in cfg:
        if posture_reward is None:
            raise ValueError('Enabled posture reward requires measured posture credit')
        terms['posture'] = torch.where(active,posture_reward,0.)
    if 'hip_collision_weight' in cfg:
        if hip_collision_cost is None:
            raise ValueError('Hip collision penalty requires measured contact cost')
        terms['hip_collision']=torch.where(active,-dt*cfg['hip_collision_weight']*hip_collision_cost.clamp(0,1),0.)
    return terms


def get_rewards(env):
    root, shoulder, upright, _, _, body_score, upright_score = env._recovery_state()
    linear, angular, _ = env._anchor_state_b()
    data = env.robot.data
    staged = env._staged_reward_result(root,shoulder,upright,linear,angular,to_torch(data.joint_vel))
    finite = env._standup_finite_this_step
    failed = env._standup_failure_this_step
    settling = env._standup_is_settling()
    active = finite & ~failed & ~settling
    env._stage_assistance_gate = torch.where(active,staged.assistance_gate,0.).detach()
    posture_reward=None
    posture_diagnostics={}
    if 'posture' in env.cfg.pair_standup_reward_cfg:
        from .posture import environment_posture
        posture_reward,posture_diagnostics=environment_posture(env,staged,root,shoulder,upright)
    terms = compose_rewards(staged.rewards,
        action=env.action_processor.applied_action,
        previous_action=env.action_processor.previous_applied_action,
        joint_position=to_torch(data.joint_pos), joint_limits=to_torch(data.joint_pos_limits),
        finite=finite,failed=failed,settling=settling,
        cfg=env.cfg.pair_standup_reward_cfg,dt=env.step_dt,posture_reward=posture_reward,
        hip_collision_cost=env._support_hip_cost)
    reward = sum(terms.values())
    log = env.extras.setdefault('log',{})
    for key,value in posture_diagnostics.items():
        log[f'Posture/{key}'] = torch.where(active,value,0.).mean().detach()
    for key,value in terms.items():
        log[f'Rewards/{key}'] = value.mean().detach()
    for key in ('preparation','rise','stand'):
        log[f'Stages/reward_{key}'] = log[f'Rewards/{key}']
    for key,value in staged.diagnostics.items():
        log[f'Stages/{key}'] = torch.where(active,value,0.).mean().detach()
    log['Stages/assistance_gate'] = env._stage_assistance_gate.mean().detach()
    log['StandUp/reward_total'] = reward.mean().detach()
    log['StandUp/reward_recovery_failure'] = log['Rewards/failure']  # old dashboards
    for key,value in {'root_height':root,'shoulder_height':shoulder,
                      'upright_projection':upright,'upright_score':upright_score,
                      'body_height_score':body_score}.items():
        valid = finite & torch.isfinite(value)
        log[f'StandUp/{key}'] = (torch.where(valid,value,0.).sum()/valid.sum().clamp_min(1)).detach()
    log['StandUp/settling_fraction'] = settling.float().mean().detach()
    log['StandUp/assistance_actual_gravity_ratio'] = env._standup_assistance_actual_ratio.mean().detach()
    log['StandUp/assistance_gravity_ratio'] = reward.new_tensor(env.cfg.pair_standup_assistance_current_gravity_ratio)
    log['StandUp/target_root_height'] = reward.new_tensor(env._pair_recovery_target_root_height)
    log['StandUp/target_shoulder_height'] = reward.new_tensor(env._pair_recovery_target_shoulder_height)
    if env.cfg.pair_standup_episode_sampling.get('enabled',False):
        mask = env.reset_time_outs & ~env.reset_terminated
        if mask.any():
            values=critic_terms(env,env._timeout_extractor)
            final,_=pack_critic_terms({k:v[mask] for k,v in values.items()})
            env.extras['final_critic_observation']=final.detach().clone()
            env.extras['final_critic_mask']=mask.clone()
    return reward

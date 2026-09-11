"""Small additive posture credit: supported waist, then unloaded upper arms."""
import math
import torch
from ref2act.isaac_compat import to_torch
from .recovery_reward import smooth_height_gate

ARM_JOINTS = tuple(f'{side}_{joint}_joint' for side in ('left','right')
                  for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow'))
JOINTS = ('waist_yaw_joint',)+ARM_JOINTS


def validate_posture(cfg):
    expected={'weight','waist_sigma','shoulder_pitch_sigma','shoulder_roll_sigma',
              'shoulder_yaw_sigma','elbow_sigma'}
    if set(cfg)!=expected:
        raise ValueError('Invalid posture reward settings')
    if any(not math.isfinite(float(v)) or v<0 for v in cfg.values()):
        raise ValueError('Posture settings must be finite and nonnegative')
    if any(cfg[k]<=0 for k in expected-{'weight'}):
        raise ValueError('Posture joint tolerances must be positive')


def ramp(value,start,full):
    return smooth_height_gate((value-start)/(full-start),start=0.,full=1.)


def posture_credit(position, target, sigma, *, support_ready, height, upright,
                   foot_load, hand_load, stage_cfg, support_cfg):
    """Positions are [N,9], ordered waist then left/right shoulder and elbow.

    Waist credit never depends on arm quality. Activating the arm gate only
    adds credit, so a bad arm pose cannot make remaining lower more rewarding.
    Foot loads use actual mg, not assistance-normalized remaining weight.
    """
    error=(position-target)/sigma
    waist=torch.exp(-error[:,0].square())
    arms=torch.exp(-error[:,1:].square().mean(-1))
    waist_gate=support_ready.clamp(0,1)
    arms_gate=(waist_gate
        *ramp(height,stage_cfg['standing_start'],stage_cfg['standing_full'])
        *ramp(upright,stage_cfg['upright_start'],stage_cfg['upright_full'])
        *ramp(foot_load,support_cfg['success_min_load'],stage_cfg['load_full'])
        *(1-ramp(hand_load,0.,stage_cfg['hands']['contact_load'])))
    credit=.5*waist_gate*waist+.5*arms_gate*arms
    return credit,dict(waist_gate=waist_gate,arms_gate=arms_gate,
                       waist_quality=waist,arms_quality=arms)


def environment_posture(env, staged, root, shoulder, upright):
    cfg=env.cfg.pair_standup_reward_cfg['posture']
    data=env.robot.data
    cached=getattr(env,'_posture_selection',None)
    if cached is None:
        names=list(env.robot.joint_names)
        if any(names.count(name)!=1 for name in JOINTS):
            raise ValueError('Posture reward requires exactly the G1 waist and shoulder/elbow joints')
        ids=torch.tensor([names.index(name) for name in JOINTS],device=env.device,dtype=torch.long)
        tolerances=[cfg['waist_sigma']]+[cfg[f'{joint}_sigma'] for _ in ('left','right')
            for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow')]
        sigma=torch.tensor(tolerances,device=env.device)
        target=to_torch(data.default_joint_pos)[:,ids]
        limits=to_torch(data.joint_pos_limits)[:,ids]
        if not torch.isfinite(target).all() or not ((target>=limits[...,0]) & (target<=limits[...,1])).all():
            raise ValueError('Default posture targets must be finite and within joint limits')
        env._posture_selection=(ids,sigma)
    else:
        ids,sigma=cached
    support=env.cfg.pair_standup_support_cfg
    height=torch.minimum(root/env._pair_recovery_target_root_height,
                         shoulder/env._pair_recovery_target_shoulder_height)
    credit,diagnostics=posture_credit(to_torch(data.joint_pos)[:,ids],
        to_torch(data.default_joint_pos)[:,ids],sigma,
        support_ready=staged.diagnostics['loaded_rise'],height=height,upright=upright,
        foot_load=env._transfer_usable_foot_load.sum(-1),
        hand_load=env._hand_state['load'].sum(-1),stage_cfg=support['stages'],support_cfg=support)
    return env.step_dt*cfg['weight']*credit,diagnostics

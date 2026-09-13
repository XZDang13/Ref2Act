"""Locomotion-style weighted squared default-pose excess in policy joint order."""
import math
import torch
from ref2act.isaac_compat import to_torch


def validate_default_pose_penalty(c):
    if set(c)!={'weight','joint_weights','tolerances'}:
        raise ValueError('Invalid default pose penalty keys')
    if not math.isfinite(float(c['weight'])) or c['weight']>0:
        raise ValueError('Default pose penalty weight must be nonpositive')
    for k in ('joint_weights','tolerances'):
        if len(c[k])!=23 or any(not math.isfinite(float(v)) or v<0 for v in c[k]):
            raise ValueError('Default pose penalty requires 23 nonnegative policy-order values')


def default_pose_cost(position, default, weights, tolerances):
    if position.shape[-1]!=weights.numel() or weights.shape!=tolerances.shape:
        raise ValueError('Pose penalty arrays must match policy joint order')
    excess=((position-default).abs()-tolerances).clamp_min(0)
    return (excess.square()*weights).sum(-1)


def environment_default_pose_penalty(env):
    c=env.cfg.pair_standup_reward_cfg['default_pose_penalty']
    q=env._sim_to_policy_order(to_torch(env.robot.data.joint_pos))
    default=env._sim_to_policy_order(to_torch(env.robot.data.default_joint_pos))
    if not hasattr(env,'_default_pose_arrays'):
        env._default_pose_arrays=tuple(torch.as_tensor(c[k],device=q.device,dtype=q.dtype) for k in ('joint_weights','tolerances'))
    return default_pose_cost(q,default,*env._default_pose_arrays)

"""Mix the existing ground reset with the robot's nominal standing state."""
import math
import torch
from ref2act.isaac_compat import to_torch
GROUPS=('ground','hold')

def load_mixed(initial):
    p=initial.get('default_stand_probability',.05)
    if isinstance(p,bool) or not math.isfinite(float(p)) or not 0<=p<=1:
        raise ValueError('default_stand_probability must be in [0,1]')
    result={'default_stand_probability':float(p)}
    if 'stand_fall' in initial:
        c=initial['stand_fall']
        if set(c)!={'height_ratio','immediate_height_ratio','upright_projection','persistence_s'} or any(isinstance(v,bool) or not math.isfinite(float(v)) for v in c.values()):
            raise ValueError('Invalid stand-reset fall configuration')
        if not (0<c['immediate_height_ratio']<c['height_ratio']<1 and 0<c['upright_projection']<1 and 0<c['persistence_s']<=1):
            raise ValueError('Invalid stand-reset fall thresholds')
        result['stand_fall']=dict(c)
    return result

def sample_states(env,ids,root,q,qd,origins):
    if not hasattr(env,'_mixed_group'):
        env._mixed_group=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
        env._mixed_target=torch.zeros_like(env.action_processor.target_joint_position)
    group=(torch.rand(ids.numel(),device=env.device)<env.cfg.pair_standup_mixed_reset['default_stand_probability']).long()
    env._mixed_group[ids]=group
    if hasattr(env,'_stand_reset_fall_steps'):
        env._stand_reset_fall_steps[ids]=0
    pos=(group==1).nonzero().flatten()
    if pos.numel():
        selected=ids[pos]
        root[pos]=to_torch(env.robot.data.default_root_state)[selected]
        root[pos,:3]+=origins[pos]
        root[pos,7:]=0.
        q[pos]=to_torch(env.robot.data.default_joint_pos)[selected]
        qd[pos]=0.
    env._mixed_target[ids]=q
    log=env.extras.setdefault('log',{})
    for g,name in enumerate(GROUPS):log[f'Reset/{name}/sampled']=(group==g).float().sum().detach()
    return group


def stand_reset_fall(group, settling, height, upright, previous, target_height, cfg, dt):
    """True terminal for sustained falls of stand-initialized episodes only."""
    from .standup_support import crouch_fall_state
    from .standup_reward import standup_hold_steps
    active=(group==1)&~settling
    fallen,steps=crouch_fall_state(height,upright,torch.where(active,previous,0),
        minimum_root_height=target_height*cfg['height_ratio'],
        immediate_root_height=target_height*cfg['immediate_height_ratio'],
        minimum_upright_projection=cfg['upright_projection'],
        required_steps=standup_hold_steps(cfg['persistence_s'],dt))
    return fallen&active,torch.where(active,steps,0)

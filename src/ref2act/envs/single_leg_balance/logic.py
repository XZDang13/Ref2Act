"""Pure Torch balance, persistence and disturbance primitives (no Isaac imports)."""
import math
import torch

ACTION_HISTORY_CONTRACT = 'post_delay_normalized_applied_action_policy_order_v1'


def capture_point(com, velocity, ground_height):
    height = (com[:,2]-ground_height).clamp_min(.10)
    return com[:,:2] + velocity[:,:2] / torch.sqrt(9.81/height)[:,None]


def persistence(previous, active, dt):
    return torch.where(active, previous+dt, torch.zeros_like(previous))


def reward_terms(s, q, target, qvel, action, previous_action, cfg):
    support = (s['load'][:,0]/cfg['support_load']).clamp(0,1)
    contact_gate = support * (~s['swing_contact']).float() * (~s['bad_contact']).float()
    lo,hi = cfg['swing_range']
    clearance = s['clearance'][:,1]
    error = (lo-clearance).clamp_min(0)+(clearance-hi).clamp_min(0)
    penalty=lambda x: x.clamp(0,1) if cfg.get('penalty_clip',True) else x
    return dict(
        balance=torch.exp(-(s['balance_error']/cfg['balance_sigma']).square())*contact_gate,
        upright=s['upright'].clamp(0,1).square(),
        swing=torch.exp(-(error/cfg['swing_sigma']).square())*contact_gate,
        height=torch.exp(-((s['height']-cfg['target_height'])/cfg['height_sigma']).square()),
        pose=torch.exp(-((q-target)/cfg['pose_sigma']).square().mean(-1)),
        velocity=penalty((qvel/cfg['velocity_scale']).square().mean(-1)),
        action_rate=penalty(((action-previous_action)/cfg['action_rate_scale']).square().mean(-1)),
        bad_contact=(s['swing_contact']|s['bad_contact']|~s['support_contact']).float(),
    )


def safety_reward_terms(q, soft_limits, joint_acc, torque, raw_target, clipped_target):
    """Uncapped physical penalties; targets and joints must share simulator order."""
    return dict(
        joint_limit=((soft_limits[...,0]-q).clamp_min(0)+(q-soft_limits[...,1]).clamp_min(0)).sum(-1),
        joint_acc=joint_acc.square().sum(-1),
        torque=torque.square().sum(-1),
        target_clip=(raw_target-clipped_target).square().sum(-1),
    )


def target_smoothness_terms(target, previous, previous2, second_valid):
    """Radians, summed over joints; suppress fictitious second differences at reset."""
    return dict(target_rate=(target-previous).square().sum(-1),
                target_acc=(target-2*previous+previous2).square().sum(-1)*second_valid)


def pulse_force(mass, delta_v, direction, elapsed, start, duration, dt):
    """Control-interval overlap integrates to m*delta_v, even off the dt grid."""
    overlap = (torch.minimum(elapsed+dt,start+duration)-torch.maximum(elapsed,start)).clamp(0,dt)
    magnitude = mass*delta_v/duration*(overlap/dt)
    return torch.stack((magnitude*direction.cos(),magnitude*direction.sin(),torch.zeros_like(mass)),-1)


class EpisodeRandom:
    """Counter-based draws indexed by seed/env/episode/stream, independent of reset order.

    Integer arithmetic is identical on CPU and CUDA. Reset and push streams never
    consume each other's draws; changing a policy's episode length cannot shift
    another environment's scenario sequence.
    """
    def __init__(self,num_envs,device,seed):
        self.episode=torch.full((num_envs,),-1,device=device,dtype=torch.long)
        self.seed=int(seed)%2147483647

    def advance(self,ids):
        self.episode[ids]+=1

    def uniform(self,ids,width,stream):
        prime=2147483647
        x=((ids[:,None]+1)*104729+(self.episode[ids,None]+1)*13007+self.seed
           +int(stream)*32452843+torch.arange(width,device=ids.device)[None]*49999)%prime
        for _ in range(3):
            x=torch.bitwise_xor(x,torch.bitwise_right_shift(x,16))
            x=(x*73244475)%prime
        return x.to(torch.float64).div(prime).float()


def validate_config(task, episode_s, dt):
    def positive(value,name):
        if not math.isfinite(float(value)) or value<=0: raise ValueError(f'{name} must be positive and finite')
    positive(episode_s,'episode length');positive(dt,'dt')
    if task['initial']['support_leg']!='left': raise ValueError('V1 requires fixed left support')
    push=task['push'];positive(push['duration_s'],'push duration')
    for key in ('time_range','delta_v_range'):
        a,b=push[key]
        if not all(math.isfinite(float(v)) for v in (a,b)) or not 0<=a<=b: raise ValueError(f'Invalid push {key}')
    if push['time_range'][0]<.8 or push['time_range'][1]+push['duration_s']>=episode_s:
        raise ValueError('Push must follow the push-free window and finish before timeout')
    for key in ('evaluation_delta_v','evaluation_direction'):
        v=push[key]
        if v is not None and (not math.isfinite(float(v)) or (key.endswith('delta_v') and v<0)):
            raise ValueError(f'Invalid {key}')
    for group in ('reset_noise','contact','termination','recovery'):
        for key,value in task[group].items():
            if group=='reset_noise' and value==0: continue
            positive(value,f'{group}.{key}')
    for key in ('balance_sigma','swing_sigma','height_sigma','pose_sigma','velocity_scale','action_rate_scale'):
        positive(task['reward'][key],key)
    lo,hi=task['reward']['swing_range']
    if not 0<lo<hi: raise ValueError('Invalid swing clearance interval')
    if not 0<task['termination']['height_ratio']<1 or not -1<task['termination']['upright']<1:
        raise ValueError('Invalid fall thresholds')
    for k,v in task['actor_noise'].items():
        if k!='enabled' and (not math.isfinite(float(v)) or v<0):
            raise ValueError(f'Invalid actor noise {k}')
    expected={'balance','upright','swing','height','pose','velocity','action_rate','bad_contact'}
    if task.get('version',1) in (2,3):
        expected|={'joint_limit','joint_acc','torque','target_clip'}
        if task['reward'].get('penalty_clip',True): raise ValueError('V2/V3 penalties must not be clipped')
        if task['version']==3: expected|={'target_rate','target_acc'}
    elif task.get('version',1)!=1: raise ValueError('Unsupported single-leg task version')
    if set(task['reward']['weights'])!=expected:
        raise ValueError('Reward weights must cover exactly the documented components')
    for k,v in task['reward']['weights'].items():
        if not math.isfinite(float(v)): raise ValueError(f'Nonfinite weight {k}')
        if k in {'joint_limit','joint_acc','torque','target_clip','target_rate','target_acc'} and v>=0: raise ValueError(f'{k} requires a negative weight')

"""Small supported waist and hand-unloading credits on the V12 baseline."""
import math
import torch
from ref2act.isaac_compat import to_torch
from .posture import ramp

def validate_supported_style(c):
    expected={'waist_weight','waist_sigma','hand_weight','hand_load_scale'}
    if set(c)!=expected or any(not math.isfinite(float(v)) or v<0 for v in c.values()):
        raise ValueError('Invalid supported waist/hand settings')
    if c['waist_sigma']<=0 or c['hand_load_scale']<=0:
        raise ValueError('Supported style scales must be positive')

def supported_style(waist,default,feet,hands,ready,c,dt,*,unload_height_gate=None):
    # Actual bilateral load, never assistance-normalized ground weight.
    gate=ready.clamp(0,1)*ramp(feet.amin(-1),.08,.2)
    waist_quality=1/(1+((waist-default)/c['waist_sigma']).square())
    extension=torch.ones_like(ready) if unload_height_gate is None else unload_height_gate.clamp(0,1)
    hand_gate=gate*ramp(feet.sum(-1),.5,.9)*extension
    # Total load: swapping the supporting hand or adding a second cannot help.
    hand_quality=1/(1+hands.clamp_min(0).sum(-1)/c['hand_load_scale'])
    terms=dict(waist_alignment=dt*c['waist_weight']*gate*waist_quality,
               hand_unload=dt*c['hand_weight']*hand_gate*hand_quality)
    return terms,dict(waist_gate=gate,hand_gate=hand_gate,unload_height_gate=extension,
        waist_quality=waist_quality,hand_unload_quality=hand_quality)

def environment_supported_style(env,staged):
    c=env.cfg.pair_standup_reward_cfg['supported_style']
    if not hasattr(env,'_supported_waist_index'):
        names=list(env.robot.joint_names)
        if names.count('waist_yaw_joint')!=1:raise ValueError('Expected one waist_yaw_joint')
        env._supported_waist_index=names.index('waist_yaw_joint')
    i=env._supported_waist_index;data=env.robot.data
    return supported_style(to_torch(data.joint_pos)[:,i],to_torch(data.default_joint_pos)[:,i],
        env._transfer_usable_foot_load,env._hand_state['load'],staged.diagnostics['loaded_rise'],c,env.step_dt,
        unload_height_gate=staged.diagnostics.get('unload_height_gate'))

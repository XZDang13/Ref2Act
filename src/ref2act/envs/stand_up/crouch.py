"""Autonomous crouch objective: bounded height, bilateral support, hands clear."""
import math
import torch


def validate_crouch(c):
    expected={'root_min','root_max','shoulder_min','shoulder_max','minimum_upright',
        'minimum_total_load','minimum_foot_load','maximum_hand_load','minimum_hand_height',
        'maximum_com_distance','weight'}
    if 'maximum_pelvis_load' in c:expected.add('maximum_pelvis_load')
    if set(c)!=expected or any(not math.isfinite(float(v)) or (v<0 if k=='minimum_hand_height' else v<=0) for k,v in c.items()):
        raise ValueError('Invalid crouch goal')
    if not (c['root_min']<c['root_max']<c['shoulder_max'] and c['shoulder_min']<c['shoulder_max']
            and c['root_min']<c['shoulder_min'] and c['minimum_upright']<1
            and 2*c['minimum_foot_load']<=c['minimum_total_load']<=1):
        raise ValueError('Invalid crouch goal intervals')


def height_quality(height, low, high):
    # Continuous below the target band, decreasing above it: standing is not the goal.
    return (height/low).clamp(0,1)/(1+((height-high).clamp_min(0)/.1).square())


def crouch_credit(x,c,ready,hands,com,motion):
    height=torch.minimum(height_quality(x.root_height,c['root_min'],c['root_max']),
                         height_quality(x.shoulder_height,c['shoulder_min'],c['shoulder_max']))
    upright=((x.upright+1)/(1+c['minimum_upright'])).clamp(0,1)
    load=(x.usable_load.clamp_min(0).sum(-1)/c['minimum_total_load']).clamp(0,1)
    unload=1/(1+(hands['load'].clamp_min(0).sum(-1)/c['maximum_hand_load']).square())
    if c['minimum_hand_height']==0:
        excess=(hands['load'].clamp_min(0).sum(-1)-c['maximum_hand_load']).clamp_min(0)
        unload=1/(1+(excess/c['maximum_hand_load']).square())
        clear=torch.ones_like(ready)
    else:
        clear=(hands['height'].amin(-1)/c['minimum_hand_height']).clamp(0,1)
    hold=load*unload*clear*com*motion
    posture=ready*height*upright
    return posture*(.6+.4*hold),dict(crouch_posture=posture,crouch_hold_quality=hold,
        crouch_height_quality=height,crouch_hand_unload=unload,crouch_hand_clear=clear)


def crouch_stable(root,shoulder,upright,ready,loads,hands,com_distance,assistance,c,*,pelvis_ground_load=None):
    if 'maximum_pelvis_load' in c and pelvis_ground_load is None:
        raise ValueError('Crouch success requires pelvis ground measurement')
    pelvis_clear=torch.ones_like(root,dtype=torch.bool) if 'maximum_pelvis_load' not in c else pelvis_ground_load<=c['maximum_pelvis_load']
    return (pelvis_clear & (root>=c['root_min']) & (root<=c['root_max'])
        & (shoulder>=c['shoulder_min']) & (shoulder<=c['shoulder_max'])
        & (upright>=c['minimum_upright']) & (ready>=.99)
        & (loads.amin(-1)>=c['minimum_foot_load'])
        & (loads.sum(-1)>=c['minimum_total_load'])
        & (hands['load'].clamp_min(0).sum(-1)<=c['maximum_hand_load'])
        & ((c['minimum_hand_height']==0) | (hands['height'].amin(-1)>=c['minimum_hand_height']))
        & (com_distance<=c['maximum_com_distance']) & (assistance.abs()<1.e-6))

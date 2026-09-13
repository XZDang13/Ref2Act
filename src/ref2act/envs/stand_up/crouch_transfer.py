"""Dense transition from a supported sit to autonomous crouch."""
import torch


def transfer_credit(x, hands, ready, c, goal):
    if x.pelvis_ground_load is None:
        raise ValueError('Crouch transfer requires measured pelvis ground load')
    root=((x.root_height-c['root_start'])/(goal['root_min']-c['root_start'])).clamp(0,1)
    shoulder=((x.shoulder_height-c['shoulder_start'])/(goal['shoulder_min']-c['shoulder_start'])).clamp(0,1)
    height=torch.minimum(root,shoulder)
    upper=1/(1+((x.root_height-goal['root_max']).clamp_min(0)/.1).square()+((x.shoulder_height-goal['shoulder_max']).clamp_min(0)/.1).square())
    # Both feet must progressively take the weight; one-foot support is not sufficient.
    bilateral=(2*x.usable_load.clamp_min(0).amin(-1)/goal['minimum_total_load']).clamp(0,1)
    pelvis_unload=1/(1+(x.pelvis_ground_load.clamp_min(0)/c['pelvis_load_scale']).square())
    hand_load=hands['load'].clamp_min(0).sum(-1)
    if goal['minimum_hand_height']==0:
        hand_load=(hand_load-goal['maximum_hand_load']).clamp_min(0)
    hand_unload=1/(1+(hand_load/c['hand_load_scale']).square())
    com=1/(1+(x.com_distance.clamp_min(0)/.08).square())
    credit=ready*upper*(.35*height+.30*bilateral+.15*com+.20*bilateral*pelvis_unload*hand_unload)
    return credit,dict(transfer_height=height,transfer_bilateral_load=bilateral,
        pelvis_ground_load=x.pelvis_ground_load,pelvis_unload=pelvis_unload,
        transfer_hand_unload=hand_unload,transfer_com=com,transfer_upper_band=upper)

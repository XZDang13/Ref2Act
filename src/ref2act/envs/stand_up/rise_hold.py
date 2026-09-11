"""Support-frame torso guidance and independently scored standing/holding."""
import torch
from itertools import combinations
from .recovery_reward import smooth_height_gate


def gate(x, a, b):
    return smooth_height_gate((x-a)/(b-a), start=0., full=1.)


def support_margin(point, vertices, valid):
    """Signed inward distance to convex-hull supporting lines, metres.

    Positive inside; non-area/empty contact regions return -infinity.
    Eight sole corners allow a small batched hull-edge enumeration on device.
    """
    pairs = torch.tensor(list(combinations(range(vertices.shape[1]), 2)), device=point.device)
    a,b = vertices[:,pairs[:,0]],vertices[:,pairs[:,1]]
    edge=b-a
    norm=edge.norm(dim=-1)
    normal=torch.stack((-edge[...,1],edge[...,0]),-1)/norm.clamp_min(1.e-8)[...,None]
    side=((vertices[:,None]-a[:,:,None])*normal[:,:,None]).sum(-1)
    lo=side.masked_fill(~valid[:,None],torch.inf).amin(-1)
    hi=side.masked_fill(~valid[:,None],-torch.inf).amax(-1)
    positive=(lo>=-1.e-6)&(hi>1.e-6)
    negative=(hi<=1.e-6)&(lo< -1.e-6)
    boundary=valid[:,pairs].all(-1)&(norm>1.e-6)&(positive|negative)
    distance=((point[:,None]-a)*normal).sum(-1)*torch.where(positive,1.,-1.)
    margin=distance.masked_fill(~boundary,torch.inf).amin(-1)
    return torch.where(boundary.any(-1),margin,-torch.inf)


def support_frame_lean(torso_up, sole_points):
    """Torso up-axis lean along average heel-to-toe direction, not torso yaw."""
    forward=(sole_points[:,:,2:,:2].mean(-2)-sole_points[:,:,:2,:2].mean(-2))
    forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    forward=forward.sum(1)
    forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    return torch.atan2((torso_up[:,:2]*forward).sum(-1),torso_up[:,2])


def rise_hold_credit(x,cfg,ready,load,takeover,hand_load):
    if cfg.get('rise_height_v3',False):
        return height_rise_hold_credit(x,cfg,hand_load)
    if x.com_margin is None or x.torso_upright is None:
        raise ValueError('rise_hold_v2 requires measured support margin and torso axis')
    height=torch.minimum(x.root_height/x.root_target,x.shoulder_height/x.shoulder_target).clamp(0,1)
    balance=1/(1+((.025-x.com_margin).clamp_min(0)/cfg['com_sigma']).square())
    # Smooth actual progress toward autonomous extension, never a constant bonus.
    extension=gate(height,cfg['rise_transfer']['fade_full'],cfg['standing_full'])
    target=cfg['rise_transfer']['lean_full_deg']*(1-extension*balance*takeover)
    angle=torch.rad2deg(x.torso_forward_lean)
    torso=1/(1+((angle-target)/20.).square())
    rise=ready*(.5*height+.25*torso+.15*balance+.1*takeover)
    complete=gate(height,cfg['standing_start'],cfg['standing_full'])
    upright=gate(torch.minimum(x.upright,x.torso_upright),cfg['upright_start'],cfg['upright_full'])
    # Actual bilateral load, not assistance-normalized load, for autonomous hold.
    feet=gate(x.usable_load.amin(-1),cfg['touch_start'],.2)
    actual_load=(x.usable_load.sum(-1)/cfg['load_full']).clamp(0,1)
    unloaded=1-gate(hand_load,0.,cfg['hands']['contact_load'])
    posture=complete*upright*torso
    # A bounded additive hold contribution cannot remove posture already earned.
    motion=1/(1+(x.linear_speed/cfg['linear_speed_scale']).square()
        +(x.angular_speed/cfg['angular_speed_scale']).square()
        +(x.joint_speed/cfg['joint_speed_scale']).square())
    autonomous=feet*actual_load*balance*unloaded
    stand=ready*posture*(.6+.4*autonomous*motion)
    return rise,stand,dict(torso_target_deg=target,torso_alignment=torso,
        com_margin_m=x.com_margin.nan_to_num(neginf=-1.),com_margin_quality=balance,
        bilateral_hold_quality=feet,stand_posture_credit=ready*posture,
        hold_credit=ready*posture*autonomous*motion,
        lean_guidance_credit=ready*torso,lean_quality=torso,lean_fade=extension,
        torso_forward_lean_deg=angle,hand_hold_unload=unloaded)


def height_rise_hold_credit(x,cfg,hand_load):
    """Height-only rising after a physical bilateral support validity check."""
    if x.com_margin is None or x.torso_upright is None or x.torso_forward_lean is None:
        raise ValueError('rise_height_v3 requires measured support and torso state')
    valid=(x.persistent_contact.bool().all(-1)
        & (x.usable_load>=cfg['touch_full']).all(-1)
        & (x.sole_plant>=cfg['plant_full']).all(-1)
        & (x.foot_distance<cfg['distance_start']).all(-1))
    if 'stance' in cfg:
        if x.sole_heading_xy is None:
            raise ValueError('Bilateral support validity requires sole geometry')
        from .standup_stance import stance_geometry
        valid &= stance_geometry(x.sole_heading_xy,cfg['stance'])['ready']>=.99
    support=valid.to(x.root_height.dtype)
    pelvis=(x.root_height/x.root_target).clamp(0,1)
    shoulder=(x.shoulder_height/x.shoulder_target).clamp(0,1)
    minimum=torch.minimum(pelvis,shoulder)
    height=.8*minimum+.1*(pelvis+shoulder)
    rise=support*height
    # Completion uses the minimum, never the partly compensating rise score.
    complete=gate(minimum,cfg['standing_start'],cfg['standing_full'])
    upright=gate(torch.minimum(x.upright,x.torso_upright),cfg['upright_start'],cfg['upright_full'])
    angle=torch.rad2deg(x.torso_forward_lean)
    upright *= 1-gate(angle.abs(),10.,30.)
    feet=gate(x.usable_load.amin(-1),cfg['touch_start'],.2)
    actual_load=(x.usable_load.sum(-1)/cfg['load_full']).clamp(0,1)
    balance=gate(x.com_margin,0.,.025)
    unloaded=1-gate(hand_load,0.,cfg['hands']['contact_load'])
    motion=1/(1+(x.linear_speed/cfg['linear_speed_scale']).square()
        +(x.angular_speed/cfg['angular_speed_scale']).square()
        +(x.joint_speed/cfg['joint_speed_scale']).square())
    pose=support*complete*upright
    hold=pose*feet*actual_load*balance*unloaded*motion
    zero=torch.zeros_like(height)
    return rise,hold,dict(rise_support_valid=support,pelvis_height_fraction=pelvis,
        shoulder_height_fraction=shoulder,standing_height_fraction=minimum,
        rise_height_completion=height,stand_posture_credit=pose,hold_credit=hold,
        com_margin_m=x.com_margin.nan_to_num(neginf=-1.),com_margin_quality=balance,
        bilateral_hold_quality=feet,hand_hold_unload=unloaded,
        torso_forward_lean_deg=angle,lean_guidance_credit=zero,lean_quality=zero,
        lean_fade=zero)

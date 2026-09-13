"""Continuous stand completion and full-axis posture guidance (opt-in V12 update)."""
import math
import torch


def support_frame_up(up, points):
    forward=points[:,:,2:,:2].mean(-2)-points[:,:,:2,:2].mean(-2)
    forward=forward/forward.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    mean=forward.sum(1)
    # Opposing sole headings: use first sole; fully degenerate geometry: world X.
    mean=torch.where((mean.norm(dim=-1)<1.e-6)[:,None],forward[:,0],mean)
    fallback=torch.zeros_like(mean); fallback[:,0]=1
    mean=torch.where((mean.norm(dim=-1)<1.e-6)[:,None],fallback,mean)
    forward=mean/mean.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    lateral=torch.stack((-forward[:,1],forward[:,0]),-1)
    return torch.stack(((up[:,:2]*forward).sum(-1),(up[:,:2]*lateral).sum(-1),up[:,2]),-1)


def posture_scores(up, height, cfg):
    # Linear target transition retains a nonzero improvement signal throughout extension.
    extension=((height-cfg['lean_transition_start'])/(1-cfg['lean_transition_start'])).clamp(0,1)
    target=math.radians(cfg['low_lean_deg'])*(1-extension)
    axis=up/up.norm(dim=-1,keepdim=True).clamp_min(1.e-6)
    cosine=axis[:,0]*torch.sin(target)+axis[:,2]*torch.cos(target)
    sigma=math.radians(cfg['tilt_sigma_deg'])
    # Squared chord error approximates angular error squared without acos singularities.
    guidance=torch.exp(-2*(1-cosine.clamp(-1,1))/sigma**2)
    upright=torch.exp(-2*(1-axis[:,2].clamp(-1,1))/sigma**2)
    return guidance,upright,target


def hold_height_gate(height, cfg):
    if not cfg.get('shared_target_hold', False):
        return torch.ones_like(height)
    # Same extension interval as the posture target: no quiet-hold incentive
    # in the deep squat, full autonomous hold credit at target standing height.
    return ((height-cfg['lean_transition_start'])/(1-cfg['lean_transition_start'])).clamp(0,1)


def stand_completion(height, posture_quality, ready, hold, cfg):
    completion=ready*height.square()*posture_quality
    a=cfg['completion_fraction']
    return completion*(a+(1-a)*hold_height_gate(height,cfg)*hold),completion

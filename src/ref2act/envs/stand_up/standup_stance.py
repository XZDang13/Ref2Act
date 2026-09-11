"""Bilateral sole geometry in a yaw-only body frame; no contact-force inference."""
import torch


def stance_geometry(sole_xy, cfg):
    # Input corner order: heel-left, heel-right, toe-left, toe-right.
    polygon = sole_xy[..., [0, 1, 3, 2], :]
    edges = torch.roll(polygon,-1,-2)-polygon
    axes = torch.stack((-edges[...,1],edges[...,0]),-1).flatten(1,2)
    norm = axes.norm(dim=-1,keepdim=True)
    valid = norm[...,0]>1.e-6
    axes = axes/norm.clamp_min(1.e-6)
    projection = torch.einsum('nfvc,nac->nfva',polygon,axes)
    low,high = projection.amin(2),projection.amax(2)
    gaps = torch.maximum(low[:,0]-high[:,1],low[:,1]-high[:,0])
    # Separating-axis clearance: >0 disjoint, <0 overlapping. This is not
    # Euclidean corner distance or physical penetration depth.
    gap = gaps.masked_fill(~valid,-1.).amax(-1)
    centers = sole_xy.mean(-2)
    side = torch.stack((centers[:,0,1],-centers[:,1,1]),-1)
    width = centers[:,0,1]-centers[:,1,1]
    clearance = 1/(1+((cfg['clearance_full']-gap).clamp_min(0)/cfg['clearance_sigma']).square())
    own_side = 1/(1+((cfg['side_full']-side).clamp_min(0)/cfg['side_sigma']).square())
    broad = 1/(1+((width-cfg['width_full']).clamp_min(0)/cfg['width_sigma']).square())
    quality = (.5*clearance+.5*own_side.mean(-1))*broad
    def gate(x,a,b):
        v=((x-a)/(b-a)).clamp(0,1)
        return v*v*(3-2*v)
    ready = gate(gap,0.,cfg['clearance_full'])*gate(side,0.,cfg['side_full']).amin(-1)
    ready *= 1-gate(width,cfg['width_full'],cfg['width_zero'])
    return dict(quality=quality,ready=ready,gap=gap,width=width,side_min=side.amin(-1))


def preparation_stance_factors(sole_xy, geometry, cfg):
    """Continuous, nonzero feedback when uncrossing; gate all preparation credit.

    Each foot must stay on its own side. Separating projected soles alone cannot
    legitimize swapped feet. Existing width bounds also prevent splaying exploits.
    """
    centers=sole_xy.mean(-2)
    side=torch.stack((centers[:,0,1],-centers[:,1,1]),-1)
    side_error=(cfg['side_full']-side).clamp_min(0)/cfg['side_sigma']
    side_factor=(1+side_error).pow(-2)
    gap_error=(cfg['clearance_full']-geometry['gap']).clamp_min(0)/cfg['clearance_sigma']
    width_error=(geometry['width']-cfg['width_full']).clamp_min(0)/cfg['width_sigma']
    separation=(1+gap_error+width_error).pow(-2)
    return side_factor,separation

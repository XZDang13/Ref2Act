"""Soft rotational constraints without restricting sagittal leg flexion."""
import torch


def rotation_costs(hip_yaw, hip_default, waist_velocity, sole_xy, body_ready, c, dt):
    hip_error = (hip_yaw-hip_default).abs()
    hip_cost = (hip_error-c['hip_yaw_tolerance']).clamp_min(0).square().sum(-1)
    # Corner order is heel-left, heel-right, toe-left, toe-right.
    direction = sole_xy[:,:,2:].mean(-2)-sole_xy[:,:,:2].mean(-2)
    length = direction.norm(dim=-1)
    angle = torch.atan2(direction[...,1],direction[...,0]).abs()
    # Near-vertical feet have no reliable projected heading. This constraint
    # activates smoothly only once the body is righted and the feet project.
    gate = body_ready.clamp(0,1)[:,None]*(length/.12).clamp(0,1)
    heading_cost = (gate*(angle-c['foot_heading_tolerance']).clamp_min(0).square()).sum(-1)
    speed_cost = (waist_velocity.abs()-c['waist_speed_tolerance']).clamp_min(0).square()
    return dict(hip_yaw=-dt*c['hip_yaw_weight']*hip_cost,
                foot_heading=-dt*c['foot_heading_weight']*heading_cost,
                waist_speed=-dt*c['waist_speed_weight']*speed_cost), dict(
        hip_yaw_error_rad=hip_error.mean(-1), hip_yaw_cost=hip_cost,
        foot_heading_error_rad=angle.mean(-1), foot_heading_cost=heading_cost,
        waist_speed_rad_s=waist_velocity.abs(), waist_speed_cost=speed_cost)

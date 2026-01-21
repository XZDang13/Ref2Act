import numpy as np
import torch
from typing import Sequence

IndexLike = torch.Tensor | Sequence[int]

def interpolate(
    a: torch.Tensor,
    b: torch.Tensor|None = None,
    blend: torch.Tensor|None = None,
    start: np.ndarray|None = None,
    end: np.ndarray|None = None,
) -> torch.Tensor:
    if start is not None and end is not None:
        return interpolate(a=a[start], b=a[end], blend=blend)
    if a.ndim >= 2:
        blend = blend.unsqueeze(-1)
    if a.ndim >= 3:
        blend = blend.unsqueeze(-1)
    return (1.0 - blend) * a + blend * b

def slerp(
    q0: torch.Tensor,
    q1: torch.Tensor|None = None,
    blend: torch.Tensor|None = None,
    start: np.ndarray|None = None,
    end: np.ndarray|None = None,
) -> torch.Tensor:
    if start is not None and end is not None:
        return slerp(q0=q0[start], q1=q0[end], blend=blend)
    if q0.ndim >= 2:
        blend = blend.unsqueeze(-1)
    if q0.ndim >= 3:
        blend = blend.unsqueeze(-1)

    qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
    cos_half_theta = (
        q0[..., qw] * q1[..., qw]
        + q0[..., qx] * q1[..., qx]
        + q0[..., qy] * q1[..., qy]
        + q0[..., qz] * q1[..., qz]
    )

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
    new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
    new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
    new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

    new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q

def compute_frame_blend(times: torch.Tensor, duration: float,
                        num_frames: int, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = torch.clamp(times / duration, 0.0, 1.0)
    index_0 = torch.round(phase * (num_frames - 1)).to(torch.long)
    index_1 = torch.clamp(index_0 + 1, max=num_frames - 1)
    index_0_float = index_0.to(times.dtype)
    blend = (times - index_0_float * dt) / dt
    blend = torch.round(blend * 1e5) / 1e5
    return index_0, index_1, blend

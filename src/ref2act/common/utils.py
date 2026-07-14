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

    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_b = torch.sin(blend * half_theta) / sin_half_theta

    new_q = ratio_a * q0 + ratio_b * q1
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
    return new_q / torch.linalg.norm(new_q, dim=-1, keepdim=True).clamp_min(1.0e-8)

def compute_frame_blend(times: torch.Tensor,
                        duration: float,
                        num_frames: int):
    phase = torch.clamp(times / duration, 0.0, 1.0)
    frame = phase * (num_frames - 1)

    index_0 = torch.floor(frame).long()
    index_1 = torch.clamp(index_0 + 1, max=num_frames - 1)
    blend = frame - index_0

    return index_0, index_1, blend


def compute_frame_blend_from_fps(times: torch.Tensor, fps: float, num_frames: int):
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    frame = torch.clamp(times.to(dtype=torch.float32) * float(fps), 0.0, float(num_frames - 1))
    index_0 = torch.floor(frame).long()
    index_1 = torch.clamp(index_0 + 1, max=num_frames - 1)
    blend = frame - index_0.to(dtype=frame.dtype)
    return index_0, index_1, blend

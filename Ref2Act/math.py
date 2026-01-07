import torch
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_mul, quat_conjugate

def relative_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = anchor_pos.shape[0]
    K = key_pos.shape[1]

    # [B, 1, 3] → [B, K, 3]
    anchor_pos = anchor_pos.expand(-1, K, -1)
    # [B, 1, 4] → [B, K, 4]
    anchor_quat = anchor_quat.expand(-1, K, -1)

    key_pos = key_pos
    key_quat = key_quat

    pos, quat = subtract_frame_transforms(
        anchor_pos,
        anchor_quat,
        key_pos,
        key_quat,
    )

    return pos, quat

def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

def quat_diff(q1: torch.Tensor, q2: torch.Tensor):
    return quat_mul(q1, quat_conjugate(q2))

def exp_error(error: torch.Tensor, std:float):
    return torch.exp(-error / std**2)
import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_mul, quat_conjugate, yaw_quat, quat_inv
from .motion_lib import ReferenceMotions


def get_relative_reference_motion_pose(
    robot_anchor_pos: torch.Tensor,
    robot_anchor_quat: torch.Tensor,
    reference_anchor_pos: torch.Tensor,
    reference_anchor_quat: torch.Tensor,
    reference_key_body_pos: torch.Tensor,
    reference_key_body_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    robot_anchor_pos = robot_anchor_pos.expand_as(reference_key_body_pos)
    robot_anchor_quat = robot_anchor_quat.expand_as(reference_key_body_quat)

    reference_anchor_pos = reference_anchor_pos.expand_as(reference_key_body_pos)
    reference_anchor_quat = robot_anchor_quat.expand_as(reference_key_body_quat)

    delta_pos_w = robot_anchor_pos.clone()
    delta_pos_w[..., 2] = reference_anchor_pos[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(reference_anchor_quat)))

    relative_reference_key_body_quat = quat_mul(delta_ori_w, reference_key_body_quat)
    relative_reference_key_body_pos = delta_pos_w + quat_apply(delta_ori_w, reference_key_body_pos - reference_anchor_pos)

    return relative_reference_key_body_pos, relative_reference_key_body_quat

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
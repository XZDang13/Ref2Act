from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return quat_conjugate(q) / torch.sum(q * q, dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_inv(q), v)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(
        2.0 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
        1.0 - 2.0 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3]),
    )
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def subtract_frame_transforms(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos = quat_apply(quat_inv(anchor_quat), key_pos - anchor_pos)
    rel_quat = quat_mul(quat_inv(anchor_quat), key_quat)
    return rel_pos, rel_quat


def get_relative_reference_motion_pose(
    robot_anchor_pos: torch.Tensor,
    robot_anchor_quat: torch.Tensor,
    reference_anchor_pos: torch.Tensor,
    reference_anchor_quat: torch.Tensor,
    reference_key_body_pos: torch.Tensor,
    reference_key_body_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reference_key_body_pos.dim() == 3:
        robot_anchor_pos = robot_anchor_pos[:, None, :].expand_as(reference_key_body_pos)
        robot_anchor_quat = robot_anchor_quat[:, None, :].expand_as(reference_key_body_quat)
        reference_anchor_pos = reference_anchor_pos[:, None, :].expand_as(reference_key_body_pos)
        reference_anchor_quat = reference_anchor_quat[:, None, :].expand_as(reference_key_body_quat)

    delta_pos_w = robot_anchor_pos.clone()
    delta_pos_w[..., 2] = reference_anchor_pos[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(reference_anchor_quat)))

    relative_reference_key_body_quat = quat_mul(delta_ori_w, reference_key_body_quat)
    relative_reference_key_body_pos = delta_pos_w + quat_apply(
        delta_ori_w,
        reference_key_body_pos - reference_anchor_pos,
    )
    return relative_reference_key_body_pos, relative_reference_key_body_quat


def relative_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key_pos.dim() == 3:
        anchor_pos = anchor_pos[:, None, :].expand_as(key_pos)
        anchor_quat = anchor_quat[:, None, :].expand_as(key_quat)
    return subtract_frame_transforms(anchor_pos, anchor_quat, key_pos, key_quat)


def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1.0
    ref_normal[..., -1] = 1.0
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=tangent.ndim - 1)


def quaternion_to_rotation_6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    rot_00 = 1.0 - 2.0 * (y * y + z * z)
    rot_01 = 2.0 * (x * y - z * w)
    rot_10 = 2.0 * (x * y + z * w)
    rot_11 = 1.0 - 2.0 * (x * x + z * z)
    rot_20 = 2.0 * (x * z - y * w)
    rot_21 = 2.0 * (y * z + x * w)
    return torch.stack((rot_00, rot_01, rot_10, rot_11, rot_20, rot_21), dim=-1)


def quat_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return quat_mul(q1, quat_conjugate(q2))


def exp_error(error: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-error / std**2)

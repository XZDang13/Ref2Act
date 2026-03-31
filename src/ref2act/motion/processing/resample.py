from __future__ import annotations

import numpy as np
import torch

from ref2act.common.utils import compute_frame_blend_from_fps, interpolate, slerp


def _quat_conjugate(quaternions: torch.Tensor) -> torch.Tensor:
    conjugate = quaternions.clone()
    conjugate[..., 1:] *= -1.0
    return conjugate


def _quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def _axis_angle_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    normalized = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)
    normalized = torch.where(normalized[..., :1] < 0.0, -normalized, normalized)

    vector = normalized[..., 1:]
    vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, normalized[..., :1].clamp(min=-1.0, max=1.0))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    axis_angle = axis * angle
    return torch.where(vector_norm > 1.0e-8, axis_angle, 2.0 * vector)


def _linear_derivative(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    return torch.gradient(values, spacing=dt, dim=0)[0]


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    num_frames = rotations.shape[0]
    angular_velocity_shape = (*rotations.shape[:-1], 3)
    if num_frames == 0:
        return torch.empty(angular_velocity_shape, device=rotations.device, dtype=rotations.dtype)
    if num_frames == 1:
        return torch.zeros(angular_velocity_shape, device=rotations.device, dtype=rotations.dtype)
    if num_frames == 2:
        q_rel = _quat_mul(rotations[1:], _quat_conjugate(rotations[:-1]))
        omega = _axis_angle_from_quat(q_rel) / dt
        return torch.cat([omega, omega], dim=0)

    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = _quat_mul(q_next, _quat_conjugate(q_prev))
    omega = _axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _build_frame_times(num_frames: int, fps: int, device: torch.device) -> torch.Tensor:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    return torch.arange(num_frames, dtype=torch.float32, device=device) / float(fps)


def _normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _resample_frames(
    values: torch.Tensor,
    *,
    source_fps: int,
    target_fps: int,
    target_num_frames: int,
    quaternion: bool = False,
) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.clone()

    target_times = _build_frame_times(target_num_frames, target_fps, values.device)
    index_0, index_1, blend = compute_frame_blend_from_fps(target_times, source_fps, values.shape[0])

    if quaternion:
        return _normalize_quaternions(slerp(values, q1=values, blend=blend, start=index_0, end=index_1))
    return interpolate(values, b=values, blend=blend, start=index_0, end=index_1)


def _resample_motion_log(
    log: dict[str, object],
    foot_heights: np.ndarray,
    *,
    source_fps: int,
    target_fps: int,
) -> tuple[dict[str, object], np.ndarray]:
    source_num_frames = int(np.asarray(log["joint_pos"]).shape[0])
    if source_num_frames < 1:
        raise ValueError("Converted motion log must contain at least one frame.")

    target_num_frames = max(int(round(float(source_num_frames) * float(target_fps) / float(source_fps))), 1)
    target_dt = 1.0 / float(target_fps)

    joint_pos = torch.as_tensor(log["joint_pos"], dtype=torch.float32)
    body_pos_w = torch.as_tensor(log["body_pos_w"], dtype=torch.float32)
    body_quat_w = torch.as_tensor(log["body_quat_w"], dtype=torch.float32)
    foot_height_tensor = torch.as_tensor(foot_heights, dtype=torch.float32)

    resampled_joint_pos = _resample_frames(
        joint_pos,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )
    resampled_body_pos_w = _resample_frames(
        body_pos_w,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )
    resampled_body_quat_w = _resample_frames(
        body_quat_w,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
        quaternion=True,
    )
    resampled_foot_heights = _resample_frames(
        foot_height_tensor,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )

    resampled_log = {
        "fps": float(target_fps),
        "joint_names": log["joint_names"],
        "body_names": log["body_names"],
        "joint_pos": resampled_joint_pos.cpu().numpy(),
        "joint_vel": _linear_derivative(resampled_joint_pos, target_dt).cpu().numpy(),
        "body_pos_w": resampled_body_pos_w.cpu().numpy(),
        "body_quat_w": resampled_body_quat_w.cpu().numpy(),
        "body_lin_vel_w": _linear_derivative(resampled_body_pos_w, target_dt).cpu().numpy(),
        "body_ang_vel_w": _so3_derivative(resampled_body_quat_w, target_dt).cpu().numpy(),
    }
    return resampled_log, resampled_foot_heights.cpu().numpy()

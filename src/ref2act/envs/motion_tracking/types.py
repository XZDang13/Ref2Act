from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from ref2act.common.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, yaw_quat


class ActionMod(Enum):
    Median = 0
    Offset = 1
    Residual = 2
    CurrentResidual = 3


@dataclass
class ReferenceMotions:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_positions: torch.Tensor
    body_quaternions: torch.Tensor
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor
    anchor_body_index: int
    tracked_body_indices: tuple[int, ...] | None = None
    robot_body_positions: torch.Tensor | None = None
    robot_body_quaternions: torch.Tensor | None = None
    body_pos_relative: torch.Tensor = field(init=False)
    body_quat_relative: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if not (0 <= self.anchor_body_index < self.body_positions.shape[1]):
            raise IndexError(f"anchor_body_index out of range: {self.anchor_body_index}")

        batch_size, num_bodies, _ = self.body_positions.shape
        if self.tracked_body_indices is None:
            self.tracked_body_indices = tuple(range(num_bodies))
        elif any(index < 0 or index >= num_bodies for index in self.tracked_body_indices):
            raise IndexError("tracked_body_indices contains an out-of-range body index.")
        ref_anchor_pos = self.body_positions[:, self.anchor_body_index]
        ref_anchor_quat = self.body_quaternions[:, self.anchor_body_index]

        if self.robot_body_positions is None or self.robot_body_quaternions is None:
            robot_anchor_pos = ref_anchor_pos
            robot_anchor_quat = ref_anchor_quat
        else:
            if self.robot_body_positions.shape[:2] != (batch_size, num_bodies):
                raise ValueError(
                    f"robot_body_positions must be [B, N, 3], got {tuple(self.robot_body_positions.shape)}"
                )
            if self.robot_body_quaternions.shape[:2] != (batch_size, num_bodies):
                raise ValueError(
                    f"robot_body_quaternions must be [B, N, 4], got {tuple(self.robot_body_quaternions.shape)}"
                )
            robot_anchor_pos = self.robot_body_positions[:, self.anchor_body_index]
            robot_anchor_quat = self.robot_body_quaternions[:, self.anchor_body_index]

        ref_anchor_pos = ref_anchor_pos[:, None, :].expand_as(self.body_positions)
        ref_anchor_quat = ref_anchor_quat[:, None, :].expand_as(self.body_quaternions)
        robot_anchor_pos = robot_anchor_pos[:, None, :].expand_as(self.body_positions)
        robot_anchor_quat = robot_anchor_quat[:, None, :].expand_as(self.body_quaternions)

        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = ref_anchor_pos[..., 2]
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))

        self.body_quat_relative = quat_mul(delta_ori, self.body_quaternions)
        relative_positions = self.body_positions - ref_anchor_pos
        self.body_pos_relative = delta_pos + quat_apply(delta_ori, relative_positions)


@dataclass
class MotionState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos: torch.Tensor
    anchor_quat: torch.Tensor
    anchor_lin_vel: torch.Tensor
    anchor_ang_vel: torch.Tensor
    key_pos: torch.Tensor
    key_quat: torch.Tensor
    key_lin_vel: torch.Tensor
    key_ang_vel: torch.Tensor


POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

JOINT_POSITION_RANGE = (-0.1, 0.1)


def pose_noise(size: int, noise_ranges: dict[str, tuple[float, float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    position_noise = []
    for key in ("x", "y", "z"):
        low_range, high_range = noise_ranges[key]
        position_noise.append(torch.empty(size, 1, device=device).uniform_(low_range, high_range))

    euler_noise = []
    for key in ("roll", "pitch", "yaw"):
        low_range, high_range = noise_ranges[key]
        euler_noise.append(torch.empty(size, device=device).uniform_(low_range, high_range))

    return torch.cat(position_noise, dim=-1), quat_from_euler_xyz(*euler_noise)


def velocity_noise(
    size: int,
    noise_ranges: dict[str, tuple[float, float]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear_vel_noise = []
    for key in ("x", "y", "z"):
        low_range, high_range = noise_ranges[key]
        linear_vel_noise.append(torch.empty(size, 1, device=device).uniform_(low_range, high_range))

    ang_vel_noise = []
    for key in ("roll", "pitch", "yaw"):
        low_range, high_range = noise_ranges[key]
        ang_vel_noise.append(torch.empty(size, 1, device=device).uniform_(low_range, high_range))

    return torch.cat(linear_vel_noise, dim=-1), torch.cat(ang_vel_noise, dim=-1)

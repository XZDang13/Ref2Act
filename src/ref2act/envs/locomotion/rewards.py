from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LocomotionRewardCfg:
    track_linear_velocity: float = 1.5
    track_yaw_rate: float = 0.75
    alive: float = 0.2
    vertical_velocity: float = -2.0
    roll_pitch_angular_velocity: float = -0.05
    orientation: float = -1.0
    joint_torque: float = -2.0e-6
    joint_acceleration: float = -2.5e-7
    action_rate: float = -1.0e-2
    joint_deviation: float = -5.0e-2
    linear_velocity_std: float = 0.5
    yaw_rate_std: float = 0.5


@dataclass(frozen=True)
class LocomotionRewardInputs:
    commands: torch.Tensor
    base_linear_velocity_b: torch.Tensor
    base_angular_velocity_b: torch.Tensor
    projected_gravity_b: torch.Tensor
    joint_pos: torch.Tensor
    default_joint_pos: torch.Tensor
    joint_acc: torch.Tensor
    applied_torque: torch.Tensor
    applied_action: torch.Tensor
    previous_applied_action: torch.Tensor


def compute_locomotion_reward_terms(
    inputs: LocomotionRewardInputs,
    cfg: LocomotionRewardCfg,
) -> dict[str, torch.Tensor]:
    linear_error = torch.sum(
        (inputs.commands[:, :2] - inputs.base_linear_velocity_b[:, :2]).square(),
        dim=-1,
    )
    yaw_error = (inputs.commands[:, 2] - inputs.base_angular_velocity_b[:, 2]).square()

    return {
        "track_linear_velocity": cfg.track_linear_velocity
        * torch.exp(-linear_error / (cfg.linear_velocity_std**2)),
        "track_yaw_rate": cfg.track_yaw_rate * torch.exp(-yaw_error / (cfg.yaw_rate_std**2)),
        "alive": torch.full_like(linear_error, cfg.alive),
        "vertical_velocity": cfg.vertical_velocity * inputs.base_linear_velocity_b[:, 2].square(),
        "roll_pitch_angular_velocity": cfg.roll_pitch_angular_velocity
        * torch.sum(inputs.base_angular_velocity_b[:, :2].square(), dim=-1),
        "orientation": cfg.orientation * torch.sum(inputs.projected_gravity_b[:, :2].square(), dim=-1),
        "joint_torque": cfg.joint_torque * torch.sum(inputs.applied_torque.square(), dim=-1),
        "joint_acceleration": cfg.joint_acceleration * torch.sum(inputs.joint_acc.square(), dim=-1),
        "action_rate": cfg.action_rate
        * torch.sum((inputs.applied_action - inputs.previous_applied_action).square(), dim=-1),
        "joint_deviation": cfg.joint_deviation
        * torch.sum((inputs.joint_pos - inputs.default_joint_pos).square(), dim=-1),
    }


__all__ = ["LocomotionRewardCfg", "LocomotionRewardInputs", "compute_locomotion_reward_terms"]

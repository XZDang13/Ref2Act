from __future__ import annotations

from dataclasses import dataclass

import torch


LOCOMOTION_REWARD_WEIGHT_FIELDS = (
    "termination_penalty",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "alive",
    "base_height_l2",
    "feet_air_time",
    "gait",
    "feet_clearance",
    "feet_slide",
    "undesired_contacts",
    "dof_pos_limits",
    "joint_deviation_hip",
    "joint_deviation_arms",
    "joint_deviation_torso",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "joint_vel_l2",
    "flat_orientation_l2",
    "action_rate_l2",
    "dof_acc_l2",
    "dof_torques_l2",
    "energy",
)


@dataclass
class LocomotionRewardCfg:
    """Unitree locomotion rewards adapted to G1-23DoF and mixed terrain."""

    # Keep legacy terms configurable for task-free PAIR subclasses, but do not
    # use them in the downstream locomotion objective.
    termination_penalty: float = 0.0
    track_lin_vel_xy_exp: float = 1.0
    track_ang_vel_z_exp: float = 0.5
    alive: float = 0.15
    base_height_l2: float = -10.0
    feet_air_time: float = 0.0
    gait: float = 0.5
    feet_clearance: float = 1.0
    feet_slide: float = -0.2
    undesired_contacts: float = -1.0
    dof_pos_limits: float = -5.0
    joint_deviation_hip: float = -1.0
    joint_deviation_arms: float = -0.1
    joint_deviation_torso: float = -1.0
    lin_vel_z_l2: float = -2.0
    ang_vel_xy_l2: float = -0.05
    joint_vel_l2: float = -0.001
    flat_orientation_l2: float = -5.0
    action_rate_l2: float = -0.05
    dof_acc_l2: float = -2.5e-7
    dof_torques_l2: float = 0.0
    energy: float = -2.0e-5
    feet_air_time_threshold: float = 0.4
    # Used only by Ref2Act's legacy completed-flight exploration signal.
    feet_air_time_maximum: float = 0.50
    linear_velocity_std: float = 0.5
    yaw_rate_std: float = 0.5
    base_height_target: float = 0.76
    gait_period: float = 0.8
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    gait_stance_threshold: float = 0.55
    feet_clearance_target: float = 0.1
    feet_clearance_std: float = 0.05
    feet_clearance_tanh_mult: float = 2.0
    undesired_contact_force_threshold: float = 1.0


@dataclass(frozen=True)
class LocomotionRewardInputs:
    commands: torch.Tensor
    base_linear_velocity_b: torch.Tensor
    base_angular_velocity_b: torch.Tensor
    base_linear_velocity_yaw_frame: torch.Tensor
    projected_gravity_b: torch.Tensor
    base_height: torch.Tensor
    joint_velocity: torch.Tensor
    joint_acc: torch.Tensor
    applied_torque: torch.Tensor
    applied_action: torch.Tensor
    previous_applied_action: torch.Tensor
    terminated: torch.Tensor
    feet_air_time: torch.Tensor
    gait: torch.Tensor
    feet_clearance: torch.Tensor
    feet_slide: torch.Tensor
    undesired_contacts: torch.Tensor
    dof_pos_limits: torch.Tensor
    joint_deviation_hip: torch.Tensor
    joint_deviation_arms: torch.Tensor
    joint_deviation_torso: torch.Tensor


def compute_feet_air_time_reward(
    last_air_time: torch.Tensor,
    first_contact: torch.Tensor,
    *,
    threshold: float,
    maximum: float,
) -> torch.Tensor:
    """Reward completed foot flights at their first subsequent contact."""

    if last_air_time.shape != first_contact.shape or last_air_time.ndim != 2:
        raise ValueError("Foot air-time tensors must have matching [env, foot] shapes.")
    if threshold < 0.0 or maximum <= threshold:
        raise ValueError("feet-air-time requires 0 <= threshold < maximum.")
    completed_flight = torch.clamp(
        last_air_time - float(threshold),
        min=0.0,
        max=float(maximum) - float(threshold),
    )
    return torch.sum(completed_flight * first_contact.to(last_air_time.dtype), dim=-1)


def compute_feet_air_time_positive_biped_reward(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    commands: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Exact IsaacLab ``feet_air_time_positive_biped`` formulation."""

    if current_air_time.shape != current_contact_time.shape or current_air_time.ndim != 2:
        raise ValueError("Foot air/contact-time tensors must have matching [env, foot] shapes.")
    if threshold <= 0.0:
        raise ValueError("feet-air-time threshold must be positive.")
    in_contact = current_contact_time > 0.0
    in_mode_time = torch.where(in_contact, current_contact_time, current_air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=float(threshold))
    reward *= torch.linalg.vector_norm(commands[:, :2], dim=-1) > 0.1
    return reward


def compute_feet_gait_reward(
    current_contact_time: torch.Tensor,
    commands: torch.Tensor,
    episode_step: torch.Tensor,
    *,
    step_dt: float,
    period: float,
    offsets: tuple[float, ...],
    stance_threshold: float,
) -> torch.Tensor:
    """Reward contacts matching a deterministic alternating-foot gait clock."""

    if current_contact_time.ndim != 2:
        raise ValueError("current_contact_time must have shape [env, foot].")
    if len(offsets) != current_contact_time.shape[1]:
        raise ValueError("gait offsets must contain one phase per foot.")
    if period <= 0.0 or step_dt <= 0.0:
        raise ValueError("gait period and step_dt must be positive.")
    if not 0.0 < stance_threshold < 1.0:
        raise ValueError("stance_threshold must be in (0, 1).")
    is_contact = current_contact_time > 0.0
    global_phase = torch.remainder(
        episode_step.to(dtype=current_contact_time.dtype) * float(step_dt),
        float(period),
    ) / float(period)
    phase_offsets = torch.tensor(
        offsets, device=current_contact_time.device, dtype=current_contact_time.dtype
    )
    leg_phase = torch.remainder(global_phase.unsqueeze(-1) + phase_offsets, 1.0)
    target_contact = leg_phase < float(stance_threshold)
    reward = torch.sum(target_contact == is_contact, dim=-1).to(current_contact_time.dtype)
    reward *= torch.linalg.vector_norm(commands, dim=-1) > 0.1
    return reward


def compute_foot_clearance_reward(
    foot_clearance: torch.Tensor,
    foot_velocity_xy: torch.Tensor,
    *,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward moving feet for tracking terrain-relative swing clearance."""

    if foot_clearance.ndim != 2 or foot_velocity_xy.shape != (*foot_clearance.shape, 2):
        raise ValueError("Foot clearance/velocity must have [env, foot] and [env, foot, 2] shapes.")
    if std <= 0.0 or tanh_mult < 0.0:
        raise ValueError("feet-clearance std must be positive and tanh_mult non-negative.")
    height_error = (foot_clearance - float(target_height)).square()
    velocity_gate = torch.tanh(
        float(tanh_mult) * torch.linalg.vector_norm(foot_velocity_xy, dim=-1)
    )
    return torch.exp(-torch.sum(height_error * velocity_gate, dim=-1) / float(std))


def compute_locomotion_reward_terms(
    inputs: LocomotionRewardInputs,
    cfg: LocomotionRewardCfg,
) -> dict[str, torch.Tensor]:
    """Compute Unitree-style G1 locomotion terms without ManagerBasedRLEnv."""

    linear_error = torch.sum(
        (inputs.commands[:, :2] - inputs.base_linear_velocity_yaw_frame[:, :2]).square(),
        dim=-1,
    )
    yaw_error = (inputs.commands[:, 2] - inputs.base_angular_velocity_b[:, 2]).square()
    return {
        "termination_penalty": cfg.termination_penalty * inputs.terminated.float(),
        "track_lin_vel_xy_exp": cfg.track_lin_vel_xy_exp
        * torch.exp(-linear_error / (cfg.linear_velocity_std**2)),
        "track_ang_vel_z_exp": cfg.track_ang_vel_z_exp
        * torch.exp(-yaw_error / (cfg.yaw_rate_std**2)),
        "alive": torch.full_like(linear_error, cfg.alive),
        "base_height_l2": cfg.base_height_l2
        * (inputs.base_height - cfg.base_height_target).square(),
        "feet_air_time": cfg.feet_air_time * inputs.feet_air_time,
        "gait": cfg.gait * inputs.gait,
        "feet_clearance": cfg.feet_clearance * inputs.feet_clearance,
        "feet_slide": cfg.feet_slide * inputs.feet_slide,
        "undesired_contacts": cfg.undesired_contacts * inputs.undesired_contacts,
        "dof_pos_limits": cfg.dof_pos_limits * inputs.dof_pos_limits,
        "joint_deviation_hip": cfg.joint_deviation_hip * inputs.joint_deviation_hip,
        "joint_deviation_arms": cfg.joint_deviation_arms * inputs.joint_deviation_arms,
        "joint_deviation_torso": cfg.joint_deviation_torso * inputs.joint_deviation_torso,
        "lin_vel_z_l2": cfg.lin_vel_z_l2 * inputs.base_linear_velocity_b[:, 2].square(),
        "ang_vel_xy_l2": cfg.ang_vel_xy_l2
        * torch.sum(inputs.base_angular_velocity_b[:, :2].square(), dim=-1),
        "joint_vel_l2": cfg.joint_vel_l2 * torch.sum(inputs.joint_velocity.square(), dim=-1),
        "flat_orientation_l2": cfg.flat_orientation_l2
        * torch.sum(inputs.projected_gravity_b[:, :2].square(), dim=-1),
        "action_rate_l2": cfg.action_rate_l2
        * torch.sum((inputs.applied_action - inputs.previous_applied_action).square(), dim=-1),
        "dof_acc_l2": cfg.dof_acc_l2 * torch.sum(inputs.joint_acc.square(), dim=-1),
        "dof_torques_l2": cfg.dof_torques_l2
        * torch.sum(inputs.applied_torque.square(), dim=-1),
        "energy": cfg.energy
        * torch.sum(torch.abs(inputs.joint_velocity) * torch.abs(inputs.applied_torque), dim=-1),
    }


__all__ = [
    "LOCOMOTION_REWARD_WEIGHT_FIELDS",
    "LocomotionRewardCfg",
    "LocomotionRewardInputs",
    "compute_feet_air_time_reward",
    "compute_feet_air_time_positive_biped_reward",
    "compute_feet_gait_reward",
    "compute_foot_clearance_reward",
    "compute_locomotion_reward_terms",
]

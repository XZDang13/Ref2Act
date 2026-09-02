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
    "feet_phase",
    "pose",
    "close_feet_xy",
    "feet_orientation",
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
    """Compact G1 locomotion objective with an observable gait phase.

    The active terms follow HoloSoma's locomotion core.  Penalties are fixed
    (no policy-dependent curriculum) and translated to Ref2Act's action scale
    so every compared actor sees an identical stationary objective.
    """

    # Keep legacy terms configurable for task-free PAIR subclasses, but do not
    # use them in the downstream locomotion objective.
    termination_penalty: float = 0.0
    track_lin_vel_xy_exp: float = 2.0
    track_ang_vel_z_exp: float = 1.5
    alive: float = 1.0
    base_height_l2: float = -10.0
    feet_air_time: float = 0.0
    feet_phase: float = 5.0
    pose: float = -0.05
    close_feet_xy: float = -1.0
    feet_orientation: float = -5.0
    # Legacy terms stay configurable for task-free PAIR subclasses, but are
    # inactive in the downstream locomotion default.
    gait: float = 0.0
    feet_clearance: float = 0.0
    feet_slide: float = 0.0
    undesired_contacts: float = 0.0
    dof_pos_limits: float = 0.0
    joint_deviation_hip: float = 0.0
    joint_deviation_arms: float = 0.0
    joint_deviation_torso: float = 0.0
    lin_vel_z_l2: float = 0.0
    ang_vel_xy_l2: float = -0.1
    joint_vel_l2: float = 0.0
    flat_orientation_l2: float = -1.0
    action_rate_l2: float = -0.05
    dof_acc_l2: float = 0.0
    dof_torques_l2: float = 0.0
    energy: float = 0.0
    feet_air_time_threshold: float = 0.4
    # Used only by Ref2Act's legacy completed-flight exploration signal.
    feet_air_time_maximum: float = 0.50
    # Commands are much narrower laterally and in yaw than longitudinally.
    # Axis-specific scales prevent an actor from collecting most of the
    # tracking reward while ignoring those smaller commands.
    linear_velocity_std: float = 0.5
    lateral_velocity_std: float = 0.2
    # HoloSoma divides the squared yaw error by 0.25.  This field is the
    # equivalent standard-deviation form, so 0.5**2 reproduces that denominator.
    yaw_rate_std: float = 0.5
    base_height_target: float = 0.76
    gait_period: float = 1.0
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    gait_randomize_phase: bool = True
    gait_stand_phase: float = torch.pi
    gait_stance_threshold: float = 0.55
    feet_phase_swing_height: float = 0.09
    feet_phase_tracking_sigma: float = 0.008
    # HoloSoma measures a dedicated sole contact point 37 mm below G1's ankle
    # roll link.  Ref2Act represents the same point virtually so its world
    # height changes correctly as the foot pitches and rolls.
    feet_contact_point_offset: tuple[float, float, float] = (0.0, 0.0, -0.037)
    feet_stance_height: float = 0.0
    close_feet_threshold: float = 0.15
    pose_weights: tuple[float, ...] = (
        0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
        0.01, 1.0, 5.0, 0.01, 5.0, 5.0,
        50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
        50.0, 50.0, 50.0, 50.0, 50.0,
    )
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
    feet_phase: torch.Tensor
    pose: torch.Tensor
    close_feet_xy: torch.Tensor
    feet_orientation: torch.Tensor
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


def compute_locomotion_gait_phase(
    episode_step: torch.Tensor,
    phase_offset: torch.Tensor,
    commands: torch.Tensor,
    *,
    step_dt: float,
    period: float,
    offsets: tuple[float, ...] = (0.0, 0.5),
    stand_phase: float = torch.pi,
) -> torch.Tensor:
    """Return per-foot phase in ``[-pi, pi)`` with standing phases grounded."""

    if episode_step.ndim != 1 or phase_offset.shape != episode_step.shape:
        raise ValueError("episode_step and phase_offset must have matching [env] shapes.")
    if commands.ndim != 2 or commands.shape[0] != episode_step.shape[0]:
        raise ValueError("commands must have shape [env, command].")
    if step_dt <= 0.0 or period <= 0.0:
        raise ValueError("gait step_dt and period must be positive.")
    offset = torch.as_tensor(offsets, device=episode_step.device, dtype=phase_offset.dtype)
    if offset.ndim != 1 or offset.numel() == 0:
        raise ValueError("gait offsets must contain at least one foot phase.")
    global_phase = (
        episode_step.to(phase_offset.dtype) * (2.0 * torch.pi * float(step_dt) / float(period))
        + phase_offset
    )
    phase = global_phase.unsqueeze(-1) + 2.0 * torch.pi * offset.unsqueeze(0)
    phase = torch.remainder(phase + torch.pi, 2.0 * torch.pi) - torch.pi
    standing = torch.linalg.vector_norm(commands[:, :3], dim=-1) < 0.01
    return torch.where(standing.unsqueeze(-1), float(stand_phase), phase)


def compute_locomotion_phase_features(phase: torch.Tensor) -> torch.Tensor:
    """Encode two-foot phase as ``sinL, cosL, sinR, cosR``."""

    if phase.ndim != 2 or phase.shape[1] != 2:
        raise ValueError("locomotion phase must have shape [env, 2].")
    return torch.stack(
        (torch.sin(phase[:, 0]), torch.cos(phase[:, 0]),
         torch.sin(phase[:, 1]), torch.cos(phase[:, 1])),
        dim=-1,
    )


def expected_foot_height_from_phase(
    phase: torch.Tensor,
    *,
    swing_height: float,
) -> torch.Tensor:
    """HoloSoma/MuJoCo-Playground cubic-Bezier swing-foot profile."""

    if swing_height < 0.0:
        raise ValueError("swing_height must be non-negative.")
    x = (phase + torch.pi) / (2.0 * torch.pi)

    def interpolate(start: torch.Tensor, end: torch.Tensor, progress: torch.Tensor):
        blend = progress**3 + 3.0 * progress.square() * (1.0 - progress)
        return start + (end - start) * blend

    ground = torch.zeros_like(x)
    apex = torch.full_like(x, float(swing_height))
    rising = interpolate(ground, apex, 2.0 * x)
    falling = interpolate(apex, ground, 2.0 * x - 1.0)
    return torch.where(x <= 0.5, rising, falling)


def compute_feet_phase_reward(
    foot_height: torch.Tensor,
    phase: torch.Tensor,
    *,
    stance_height: float,
    swing_height: float,
    tracking_sigma: float,
) -> torch.Tensor:
    """Reward terrain-relative foot height following the observable phase."""

    if foot_height.shape != phase.shape or foot_height.ndim != 2:
        raise ValueError("foot_height and phase must have matching [env, foot] shapes.")
    if tracking_sigma <= 0.0:
        raise ValueError("feet phase tracking_sigma must be positive.")
    target = float(stance_height) + expected_foot_height_from_phase(
        phase, swing_height=swing_height
    )
    error = torch.sum((foot_height - target).square(), dim=-1)
    return torch.exp(-error / float(tracking_sigma))


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

    if cfg.linear_velocity_std <= 0.0 or cfg.lateral_velocity_std <= 0.0:
        raise ValueError("Longitudinal and lateral velocity std must be positive.")
    if cfg.yaw_rate_std <= 0.0:
        raise ValueError("Yaw-rate std must be positive.")
    linear_delta = inputs.commands[:, :2] - inputs.base_linear_velocity_yaw_frame[:, :2]
    linear_error = (
        linear_delta[:, 0].square() / (cfg.linear_velocity_std**2)
        + linear_delta[:, 1].square() / (cfg.lateral_velocity_std**2)
    )
    yaw_error = (inputs.commands[:, 2] - inputs.base_angular_velocity_b[:, 2]).square()
    return {
        "termination_penalty": cfg.termination_penalty * inputs.terminated.float(),
        "track_lin_vel_xy_exp": cfg.track_lin_vel_xy_exp
        * torch.exp(-linear_error),
        "track_ang_vel_z_exp": cfg.track_ang_vel_z_exp
        * torch.exp(-yaw_error / (cfg.yaw_rate_std**2)),
        "alive": torch.full_like(linear_error, cfg.alive),
        "base_height_l2": cfg.base_height_l2
        * (inputs.base_height - cfg.base_height_target).square(),
        "feet_air_time": cfg.feet_air_time * inputs.feet_air_time,
        "feet_phase": cfg.feet_phase * inputs.feet_phase,
        "pose": cfg.pose * inputs.pose,
        "close_feet_xy": cfg.close_feet_xy * inputs.close_feet_xy,
        "feet_orientation": cfg.feet_orientation * inputs.feet_orientation,
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
    "compute_feet_phase_reward",
    "compute_feet_gait_reward",
    "compute_foot_clearance_reward",
    "compute_locomotion_gait_phase",
    "compute_locomotion_phase_features",
    "compute_locomotion_reward_terms",
    "expected_foot_height_from_phase",
]

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


LOCOMOTION_TASK_REWARD_WEIGHTS = (
    "track_command_exp",
    "stand_still_exp",
    "inactive_command_axes",
    "swing_contact_penalty",
    "stance_missing_contact_penalty",
    "unexpected_double_support_penalty",
    "swing_foot_under_clearance_penalty",
    "moving_flight",
    "base_height_l2",
    "feet_air_time",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "flat_orientation_l2",
    "feet_slide",
    "action_rate_l2",
    "dof_acc_l2",
    "dof_torques_l2",
    "dof_pos_limits",
    "termination_penalty",
)


@dataclass
class FlatLocomotionRewardCfg:
    """Command tracking plus an observable alternating gait for flat G1 locomotion.

    Moving commands receive a positive exponential score averaged over only
    their active axes.  Explicit stand commands use a separate positive score.
    A shared phase drives both the policy observation and phase-conditioned
    foot-height/contact targets, preventing synchronized jumping from being an
    equally good solution to velocity tracking.
    """

    track_command_exp: float = 2.0
    stand_still_exp: float = 2.0
    inactive_command_axes: float = -0.3
    swing_contact_penalty: float = -0.25
    stance_missing_contact_penalty: float = -0.25
    unexpected_double_support_penalty: float = -0.25
    swing_foot_under_clearance_penalty: float = -0.25
    moving_flight: float = -0.5
    base_height_l2: float = -5.0
    feet_air_time: float = 0.5
    lin_vel_z_l2: float = -0.2
    ang_vel_xy_l2: float = -0.02
    flat_orientation_l2: float = -1.0
    feet_slide: float = -0.1
    action_rate_l2: float = -0.002
    dof_acc_l2: float = -1.0e-7
    dof_torques_l2: float = -2.0e-6
    dof_pos_limits: float = -1.0
    termination_penalty: float = -200.0
    linear_velocity_scales: tuple[float, float] = (0.5, 0.3)
    yaw_rate_scale: float = 0.5
    command_activity_threshold: float = 0.05
    feet_air_time_threshold: float = 0.4
    base_height_target: float = 0.76
    gait_period: float = 1.0
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    gait_randomize_phase: bool = True
    gait_stand_phase: float = torch.pi
    gait_stance_ratio: float = 0.55
    gait_swing_height: float = 0.09
    feet_contact_point_offset: tuple[float, float, float] = (0.0, 0.0, -0.037)

    def __post_init__(self) -> None:
        if len(self.linear_velocity_scales) != 2 or min(self.linear_velocity_scales) <= 0.0:
            raise ValueError("linear_velocity_scales must contain two positive values.")
        if self.yaw_rate_scale <= 0.0:
            raise ValueError("yaw_rate_scale must be positive.")
        if self.command_activity_threshold < 0.0:
            raise ValueError("command_activity_threshold must be non-negative.")
        if self.feet_air_time_threshold <= 0.0:
            raise ValueError("feet_air_time_threshold must be positive.")
        if self.gait_period <= 0.0:
            raise ValueError("gait_period must be positive.")
        if len(self.gait_offsets) != 2:
            raise ValueError("gait_offsets must contain one offset per foot.")
        if not 0.5 <= self.gait_stance_ratio < 1.0:
            raise ValueError("gait_stance_ratio must be in [0.5, 1).")
        if self.gait_swing_height <= 0.0:
            raise ValueError("gait_swing_height must be positive.")
        if len(self.feet_contact_point_offset) != 3:
            raise ValueError("feet_contact_point_offset must be a 3-vector.")

    def contract(self) -> dict[str, object]:
        values = asdict(self)
        return {
            "source": "Ref2Act flat locomotion v3",
            "weights": {
                name: values[name] for name in LOCOMOTION_TASK_REWARD_WEIGHTS
            },
            "linear_velocity_scales": list(self.linear_velocity_scales),
            "yaw_rate_scale": self.yaw_rate_scale,
            "command_activity_threshold": self.command_activity_threshold,
            "feet_air_time_threshold": self.feet_air_time_threshold,
            "base_height_target": self.base_height_target,
            "gait_period": self.gait_period,
            "gait_offsets": list(self.gait_offsets),
            "gait_randomize_phase": self.gait_randomize_phase,
            "gait_stand_phase": self.gait_stand_phase,
            "gait_stance_ratio": self.gait_stance_ratio,
            "gait_swing_height": self.gait_swing_height,
            "feet_contact_point_offset": list(self.feet_contact_point_offset),
            "tracking": "positive active-axis exponential with explicit stand score",
            "gait_clock": "observable alternating phase with matched foot targets",
            "gait_objective": "zero-at-compliance contact and under-clearance penalties",
            "alive_reward": False,
            "pose_reward": False,
            "penalty_curriculum": False,
        }


@dataclass(frozen=True)
class FlatLocomotionRewardInputs:
    commands: torch.Tensor
    base_linear_velocity_b: torch.Tensor
    base_angular_velocity_b: torch.Tensor
    base_linear_velocity_yaw_frame: torch.Tensor
    projected_gravity_b: torch.Tensor
    base_height: torch.Tensor
    gait_phase: torch.Tensor
    feet_height: torch.Tensor
    feet_contact: torch.Tensor
    joint_acc: torch.Tensor
    applied_torque: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    terminated: torch.Tensor
    feet_air_time: torch.Tensor
    feet_slide: torch.Tensor
    dof_pos_limits: torch.Tensor


def phase_gait_targets(
    phase: torch.Tensor,
    *,
    stance_ratio: float,
    swing_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mutually consistent per-foot height and contact targets.

    Phase zero is the middle of swing.  Stance occupies the wrapped region
    around ``-pi/pi``.  With two feet separated by half a cycle and a stance
    ratio of at least 0.5, the schedule contains alternating single support
    plus optional double support, but never asks both feet to fly.
    """

    if phase.ndim != 2 or phase.shape[1] != 2:
        raise ValueError("phase must have shape [env, 2].")
    if not 0.5 <= float(stance_ratio) < 1.0:
        raise ValueError("stance_ratio must be in [0.5, 1).")
    if swing_height < 0.0:
        raise ValueError("swing_height must be non-negative.")

    swing_half_width = torch.pi * (1.0 - float(stance_ratio))
    wrapped = torch.remainder(phase + torch.pi, 2.0 * torch.pi) - torch.pi
    in_swing = wrapped.abs() < swing_half_width
    progress = ((wrapped + swing_half_width) / (2.0 * swing_half_width)).clamp(0.0, 1.0)
    half_progress = torch.where(progress <= 0.5, 2.0 * progress, 2.0 * (1.0 - progress))
    smooth_height = half_progress.square() * (3.0 - 2.0 * half_progress)
    target_height = float(swing_height) * smooth_height * in_swing
    return target_height, ~in_swing


def phase_gait_signals(
    inputs: FlatLocomotionRewardInputs,
    cfg: FlatLocomotionRewardCfg,
) -> dict[str, torch.Tensor]:
    """Return zero-at-compliance gait violations and diagnostic signals."""

    target_foot_height, target_contact = phase_gait_targets(
        inputs.gait_phase,
        stance_ratio=cfg.gait_stance_ratio,
        swing_height=cfg.gait_swing_height,
    )
    if inputs.feet_height.shape != target_foot_height.shape:
        raise ValueError("feet_height must match the two-foot gait phase shape.")
    if inputs.feet_contact.shape != target_contact.shape:
        raise ValueError("feet_contact must match the two-foot gait phase shape.")

    actual_contact = inputs.feet_contact.bool()
    target_swing = ~target_contact
    dtype = inputs.commands.dtype
    swing_count = target_swing.sum(dim=-1).clamp(min=1).to(dtype)
    stance_count = target_contact.sum(dim=-1).clamp(min=1).to(dtype)
    moving = (inputs.commands.abs() > float(cfg.command_activity_threshold)).any(dim=-1)
    target_contact_count = target_contact.sum(dim=-1)
    actual_contact_count = actual_contact.sum(dim=-1)

    swing_contact = (target_swing & actual_contact).to(dtype).sum(dim=-1) / swing_count
    swing_contact *= target_swing.any(dim=-1)
    stance_missing = (target_contact & ~actual_contact).to(dtype).sum(dim=-1) / stance_count
    unexpected_double_support = (
        (target_contact_count == 1) & (actual_contact_count == 2) & moving
    ).to(dtype)
    clearance_deficit = torch.clamp(target_foot_height - inputs.feet_height, min=0.0)
    normalized_deficit = clearance_deficit / float(cfg.gait_swing_height)
    under_clearance = (
        normalized_deficit.square() * target_swing.to(dtype)
    ).sum(dim=-1) / swing_count
    under_clearance *= target_swing.any(dim=-1)
    moving_flight = ((actual_contact_count == 0) & moving).to(dtype)

    return {
        "swing_contact_penalty": swing_contact,
        "stance_missing_contact_penalty": stance_missing,
        "unexpected_double_support_penalty": unexpected_double_support,
        "swing_foot_under_clearance_penalty": under_clearance,
        "moving_flight": moving_flight,
        "phase_contact_match": (actual_contact == target_contact).to(dtype).mean(dim=-1),
        "target_single_stance": ((target_contact_count == 1) & moving).to(dtype),
        "target_double_support": ((target_contact_count == 2) & moving).to(dtype),
    }


def positive_command_tracking(
    commands: torch.Tensor,
    measured: torch.Tensor,
    scale: torch.Tensor,
    *,
    activity_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return moving tracking, stand score, and inactive-axis error.

    Moving-command tracking is averaged over active axes, so sagittal,
    lateral, turning, and mixed commands all have a maximum raw score of one.
    Stand commands receive their own positive score.  Inactive-axis error is
    reported only for moving commands and prevents cross-axis drift without
    granting free positive reward for already-zero axes.
    """

    if commands.shape != measured.shape or commands.ndim != 2:
        raise ValueError("commands and measured must have matching [env, axis] shapes.")
    if scale.shape != (commands.shape[1],):
        raise ValueError("scale must contain one value per command axis.")
    active = commands.abs() > float(activity_threshold)
    moving = active.any(dim=-1)
    axis_score = torch.exp(-((commands - measured) / scale).square())
    active_count = active.sum(dim=-1).clamp(min=1).to(measured.dtype)
    moving_tracking = (axis_score * active).sum(dim=-1) / active_count
    moving_tracking *= moving

    stand_score = torch.exp(-(measured / scale).square().sum(dim=-1))
    stand_score *= ~moving

    inactive = (~active) & moving.unsqueeze(-1)
    inactive_count = inactive.sum(dim=-1).clamp(min=1).to(measured.dtype)
    # A bounded cost prevents the initially noisy Gaussian policy from making
    # this secondary cross-axis regularizer dominate the task reward.
    inactive_axis_cost = 1.0 - torch.exp(-(measured / scale).square())
    inactive_error = (inactive_axis_cost * inactive).sum(dim=-1) / inactive_count
    return moving_tracking, stand_score, inactive_error


def compute_flat_locomotion_reward_terms(
    inputs: FlatLocomotionRewardInputs,
    cfg: FlatLocomotionRewardCfg,
) -> dict[str, torch.Tensor]:
    """Return weighted per-second terms; the environment applies ``dt``."""

    velocity_scale = torch.as_tensor(
        (*cfg.linear_velocity_scales, cfg.yaw_rate_scale),
        device=inputs.commands.device,
        dtype=inputs.commands.dtype,
    )
    measured = torch.stack(
        (
            inputs.base_linear_velocity_yaw_frame[:, 0],
            inputs.base_linear_velocity_yaw_frame[:, 1],
            inputs.base_angular_velocity_b[:, 2],
        ),
        dim=-1,
    )
    moving_tracking, stand_score, inactive_error = positive_command_tracking(
        inputs.commands,
        measured,
        velocity_scale,
        activity_threshold=cfg.command_activity_threshold,
    )
    gait = phase_gait_signals(inputs, cfg)
    raw = {
        "track_command_exp": moving_tracking,
        "stand_still_exp": stand_score,
        "inactive_command_axes": inactive_error,
        "swing_contact_penalty": gait["swing_contact_penalty"],
        "stance_missing_contact_penalty": gait["stance_missing_contact_penalty"],
        "unexpected_double_support_penalty": gait[
            "unexpected_double_support_penalty"
        ],
        "swing_foot_under_clearance_penalty": gait[
            "swing_foot_under_clearance_penalty"
        ],
        "moving_flight": gait["moving_flight"],
        "base_height_l2": (inputs.base_height - float(cfg.base_height_target)).square(),
        "feet_air_time": inputs.feet_air_time,
        "lin_vel_z_l2": inputs.base_linear_velocity_b[:, 2].square(),
        "ang_vel_xy_l2": inputs.base_angular_velocity_b[:, :2].square().sum(-1),
        "flat_orientation_l2": inputs.projected_gravity_b[:, :2].square().sum(-1),
        "feet_slide": inputs.feet_slide,
        "action_rate_l2": (inputs.action - inputs.previous_action).square().sum(-1),
        "dof_acc_l2": inputs.joint_acc.square().sum(-1),
        "dof_torques_l2": inputs.applied_torque.square().sum(-1),
        "dof_pos_limits": inputs.dof_pos_limits,
        "termination_penalty": inputs.terminated.float(),
    }
    return {
        name: raw[name] * float(getattr(cfg, name))
        for name in LOCOMOTION_TASK_REWARD_WEIGHTS
    }


__all__ = [
    "FlatLocomotionRewardCfg",
    "FlatLocomotionRewardInputs",
    "LOCOMOTION_TASK_REWARD_WEIGHTS",
    "compute_flat_locomotion_reward_terms",
    "phase_gait_targets",
    "phase_gait_signals",
    "positive_command_tracking",
]

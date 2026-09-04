from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ref2act.common.math import quat_apply_inverse, yaw_quat


LOCOMOTION_TASK_REWARD_WEIGHTS = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "swing_contact_penalty",
    "stance_missing_contact_penalty",
    "swing_foot_height_l2",
    "pose",
    "leg_clearance",
    "base_height_l2",
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
    """Full command tracking, soft whole-body pose and one shared phase objective.

    Zero-command axes are tracking targets too. Tracking has a total maximum
    of two for every command, including stand. All other terms are costs.
    """

    track_lin_vel_xy_exp: float = 1.0
    track_ang_vel_z_exp: float = 1.0
    swing_contact_penalty: float = -0.5
    stance_missing_contact_penalty: float = -0.25
    swing_foot_height_l2: float = -0.5
    pose: float = -0.5
    leg_clearance: float = -0.5
    # Left-minus-right separation in the pelvis yaw frame, feet then knees.
    # USD collision widths: 0.070 / 0.07177 m, plus about 0.020 m margin.
    leg_min_lateral_separation: tuple[float, float] = (0.09, 0.092)
    # G1 policy order: hip pitch/roll/yaw, knee, ankle pitch/roll per leg;
    # waist; shoulder pitch/roll/yaw, elbow, wrist roll per arm.
    # Every joint is covered. Sagittal leg motion has wider, weaker priors.
    pose_weights: tuple[float, ...] = (
        0.1, 1.0, 1.0, 0.1, 0.1, 1.0,
    ) * 2 + (1.0,) + (
        0.25, 1.0, 1.0, 0.25, 1.0,
    ) * 2
    # Radians of unpenalized deviation from the actual default joint pose.
    pose_tolerances: tuple[float, ...] = (
        0.35, 0.10, 0.15, 0.50, 0.25, 0.10,
    ) * 2 + (0.10,) + (
        0.15, 0.10, 0.10, 0.20, 0.10,
    ) * 2
    base_height_l2: float = -20.0
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
    # USD FK: hip/knee/ankle pitch = -0.15/0.35/-0.20 rad; level soles.
    base_height_target: float = 0.7841
    gait_period: float = 1.0
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    gait_randomize_phase: bool = True
    gait_stand_phase: float = torch.pi
    gait_stance_ratio: float = 0.55
    gait_swing_height: float = 0.09
    feet_contact_point_offset: tuple[float, float, float] = (0.0, 0.0, -0.037)

    def __post_init__(self) -> None:
        if len(self.leg_min_lateral_separation) != 2 or any(
            not math.isfinite(v) or v <= 0.0 for v in self.leg_min_lateral_separation
        ):
            raise ValueError("leg_min_lateral_separation must contain two finite positive values.")
        if len(self.linear_velocity_scales) != 2 or min(self.linear_velocity_scales) <= 0.0:
            raise ValueError("linear_velocity_scales must contain two positive values.")
        if self.yaw_rate_scale <= 0.0:
            raise ValueError("yaw_rate_scale must be positive.")
        if self.command_activity_threshold < 0.0:
            raise ValueError("command_activity_threshold must be non-negative.")
        for name in ("pose_weights", "pose_tolerances"):
            values = getattr(self, name)
            if len(values) != 23 or any(not math.isfinite(v) or v < 0.0 for v in values):
                raise ValueError(f"{name} must contain 23 finite non-negative values in policy order.")
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
            "source": "Ref2Act flat locomotion v8",
            "weights": {
                name: values[name] for name in LOCOMOTION_TASK_REWARD_WEIGHTS
            },
            "linear_velocity_scales": list(self.linear_velocity_scales),
            "yaw_rate_scale": self.yaw_rate_scale,
            "command_activity_threshold": self.command_activity_threshold,
            "pose_weights": list(self.pose_weights),
            "pose_tolerances": list(self.pose_tolerances),
            "pose_objective": "sum(weight * relu(abs(q-q_default)-tolerance_rad)^2)",
            "leg_min_lateral_separation": list(self.leg_min_lateral_separation),
            "leg_clearance_objective": "sum(relu(1-signed_y_left_minus_right/min_separation_m)^2), feet then knees, pelvis yaw frame",
            "base_height_target": self.base_height_target,
            "gait_period": self.gait_period,
            "gait_offsets": list(self.gait_offsets),
            "gait_randomize_phase": self.gait_randomize_phase,
            "gait_stand_phase": self.gait_stand_phase,
            "gait_stance_ratio": self.gait_stance_ratio,
            "gait_swing_height": self.gait_swing_height,
            "feet_contact_point_offset": list(self.feet_contact_point_offset),
            "tracking": "full xy error plus independent yaw error, including zero-command axes",
            "gait_clock": "observable alternating phase with matched foot targets",
            "gait_objective": "contact mismatch and two-sided swing-height penalties",
            "alive_reward": False,
            "pose_reward": True,
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
    leg_lateral_separation: torch.Tensor
    joint_acc: torch.Tensor
    applied_torque: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    terminated: torch.Tensor
    pose: torch.Tensor
    feet_slide: torch.Tensor
    dof_pos_limits: torch.Tensor


def leg_lateral_separation(
    feet_position_w: torch.Tensor,
    knee_position_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Signed feet/knee widths; each position tensor is ordered left, right.

    Yaw-only coordinates keep world heading and root roll/pitch from changing
    the lateral reference. Fore-aft spacing is intentionally unconstrained.
    """
    heading = yaw_quat(pelvis_quat_w)
    return torch.stack([
        quat_apply_inverse(heading, positions[:, 0] - positions[:, 1])[:, 1]
        for positions in (feet_position_w, knee_position_w)
    ], dim=-1)


def leg_clearance_penalty(
    separation: torch.Tensor, minimum: tuple[float, float]
) -> torch.Tensor:
    """Zero above the minimum; crossing remains penalized (never use abs)."""
    if separation.ndim != 2 or separation.shape[1] != 2:
        raise ValueError("leg_lateral_separation must have shape [env, 2].")
    threshold = separation.new_tensor(minimum)
    return (1.0 - separation / threshold).clamp(min=0.0).square().sum(-1)


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
    dtype = inputs.commands.dtype
    moving = (inputs.commands.abs() > float(cfg.command_activity_threshold)).any(dim=-1)
    # Stand requires grounded feet even if a caller supplies a stale phase.
    target_contact = target_contact | ~moving.unsqueeze(-1)
    target_swing = ~target_contact
    target_foot_height = target_foot_height * moving.unsqueeze(-1)
    swing_count = target_swing.sum(dim=-1).clamp(min=1).to(dtype)
    stance_count = target_contact.sum(dim=-1).clamp(min=1).to(dtype)
    target_contact_count = target_contact.sum(dim=-1)
    actual_contact_count = actual_contact.sum(dim=-1)

    swing_contact = (target_swing & actual_contact).to(dtype).sum(dim=-1) / swing_count
    swing_contact *= target_swing.any(dim=-1)
    stance_missing = (target_contact & ~actual_contact).to(dtype).sum(dim=-1) / stance_count
    unexpected_double_support = (
        (target_contact_count == 1) & (actual_contact_count == 2) & moving
    ).to(dtype)
    normalized_error = (target_foot_height - inputs.feet_height) / float(cfg.gait_swing_height)
    height_error = (
        normalized_error.square() * target_swing.to(dtype)
    ).sum(dim=-1) / swing_count
    height_error *= target_swing.any(dim=-1)
    moving_flight = ((actual_contact_count == 0) & moving).to(dtype)

    return {
        "swing_contact_penalty": swing_contact,
        "stance_missing_contact_penalty": stance_missing,
        "unexpected_double_support_penalty": unexpected_double_support,
        "swing_foot_height_l2": height_error,
        "moving_flight": moving_flight,
        "phase_contact_match": (actual_contact == target_contact).to(dtype).mean(dim=-1),
        "target_single_stance": ((target_contact_count == 1) & moving).to(dtype),
        "target_double_support": ((target_contact_count == 2) & moving).to(dtype),
    }


def command_tracking(
    commands: torch.Tensor,
    measured: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track both planar axes jointly and yaw independently, also at stand."""

    if commands.shape != measured.shape or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands and measured must have matching [env, 3] shapes.")
    if scale.shape != (3,):
        raise ValueError("scale must contain one value per command axis.")
    squared_error = ((commands - measured) / scale).square()
    return torch.exp(-squared_error[:, :2].sum(-1)), torch.exp(-squared_error[:, 2])


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
    linear_tracking, yaw_tracking = command_tracking(
        inputs.commands,
        measured,
        velocity_scale,
    )
    gait = phase_gait_signals(inputs, cfg)
    raw = {
        "track_lin_vel_xy_exp": linear_tracking,
        "track_ang_vel_z_exp": yaw_tracking,
        "swing_contact_penalty": gait["swing_contact_penalty"],
        "stance_missing_contact_penalty": gait["stance_missing_contact_penalty"],
        "swing_foot_height_l2": gait["swing_foot_height_l2"],
        "pose": inputs.pose,
        "leg_clearance": leg_clearance_penalty(
            inputs.leg_lateral_separation, cfg.leg_min_lateral_separation
        ),
        "base_height_l2": (inputs.base_height - float(cfg.base_height_target)).square(),
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
    "command_tracking",
    "leg_lateral_separation",
    "leg_clearance_penalty",
]

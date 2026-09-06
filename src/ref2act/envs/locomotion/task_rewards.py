from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch


LOCOMOTION_TASK_REWARD_WEIGHTS = (
    "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "termination_penalty",
    "lin_vel_z_l2", "ang_vel_xy_l2", "dof_torques_l2", "dof_acc_l2",
    "action_rate_l2", "feet_air_time", "both_feet_air", "feet_slide", "flat_orientation_l2",
    "dof_pos_limits", "joint_deviation_hip", "joint_deviation_arms",
    "joint_deviation_torso",
)


@dataclass
class FlatLocomotionRewardCfg:
    """jloganolson G1-23DoF locomotion rewards on Ref2Act's existing controller.

    Upstream commit: fbfa38706b817e2d4b19e444db95ae7fb2537b46.
    Tracking, air-time and regularization coefficients are overridden for the PAIR experiment.
    Coefficients are per second, including termination. The environment applies
    control dt once. Phase settings preserve the existing observation contract;
    phase is not a reward input. There is no reward penalty curriculum.
    """

    track_lin_vel_xy_exp: float = 1.0
    track_ang_vel_z_exp: float = 2.0
    termination_penalty: float = -200.0
    lin_vel_z_l2: float = 0.0
    ang_vel_xy_l2: float = -0.05
    dof_torques_l2: float = -1.5e-7
    dof_acc_l2: float = -1.25e-7
    action_rate_l2: float = -0.01
    feet_air_time: float = 0.25
    both_feet_air: float = -0.5
    feet_slide: float = -0.1
    flat_orientation_l2: float = -1.0
    dof_pos_limits: float = -1.0
    joint_deviation_hip: float = -0.1
    joint_deviation_arms: float = -0.2
    joint_deviation_torso: float = -0.1
    linear_velocity_scales: tuple[float, float] = (0.5, 0.5)
    yaw_rate_scale: float = 0.5
    feet_air_time_threshold: float = 0.4
    # Retained solely for the unchanged command/phase observation stream.
    gait_period: float = 1.0
    gait_offsets: tuple[float, float] = (0.0, 0.5)
    gait_randomize_phase: bool = True
    gait_stand_phase: float = math.pi

    def __post_init__(self) -> None:
        if len(self.linear_velocity_scales) != 2 or any(
            not math.isfinite(v) or v <= 0 for v in self.linear_velocity_scales
        ):
            raise ValueError("linear_velocity_scales must contain two finite positive values.")
        for name in ("yaw_rate_scale", "feet_air_time_threshold", "gait_period"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if len(self.gait_offsets) != 2:
            raise ValueError("Expected two gait offsets.")

    def contract(self) -> dict[str, object]:
        values = asdict(self)
        return {
            "source": "Ref2Act jloganolson G1-23DoF locomotion reward",
            "version": 2,
            "upstream_repository": "https://github.com/jloganolson/g1_23dof_locomotion_isaac",
            "upstream_commit": "fbfa38706b817e2d4b19e444db95ae7fb2537b46",
            "weights": {name: values.pop(name) for name in LOCOMOTION_TASK_REWARD_WEIGHTS},
            **values,
            "tracking_frame": "pelvis yaw frame xy; world frame angular z",
            "joint_selection": {
                "torques_and_acceleration": [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                "position_limits": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                "hip_deviation": [".*_hip_yaw_joint", ".*_hip_roll_joint"],
                "arm_deviation": [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
                "torso_deviation": ["waist_yaw_joint"],
            },
            "pose_objective": "L1 deviation from live defaults on selected joints; no fingers on G1-23DoF",
            "air_time_objective": "single support min(current mode times), capped at feet_air_time_threshold; norm(command_xy) > 0.1",
            "both_feet_air_objective": "both feet current_air_time > 0; no command gate, no phase target",
            "action_rate_input": "raw policy action before delay, scaling and joint-limit clamp",
            "termination_input": "existing Ref2Act non-timeout termination mask",
            "reward_dt_scaling": "sum of weighted terms multiplied by control dt exactly once, including termination",
            "only_positive_rewards": False,
        }


@dataclass(frozen=True)
class FlatLocomotionRewardInputs:
    commands: torch.Tensor
    base_linear_velocity_b: torch.Tensor
    base_angular_velocity_b: torch.Tensor
    base_angular_velocity_w: torch.Tensor
    base_linear_velocity_yaw_frame: torch.Tensor
    projected_gravity_b: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    terminated: torch.Tensor
    # Joint tensors are already restricted to the configured hip/knee/ankle group.
    applied_torque: torch.Tensor
    joint_acc: torch.Tensor
    feet_air_time: torch.Tensor
    feet_current_air_time: torch.Tensor
    feet_slide: torch.Tensor
    dof_pos_limits: torch.Tensor
    joint_deviation_hip: torch.Tensor
    joint_deviation_arms: torch.Tensor
    joint_deviation_torso: torch.Tensor


def compute_both_feet_air(current_air_time: torch.Tensor) -> torch.Tensor:
    """Upstream both_feet_air: penalize simultaneous flight, also at stand/turn."""
    if current_air_time.ndim != 2 or current_air_time.shape[1] != 2:
        raise ValueError("Expected current_air_time [env, 2].")
    return ((current_air_time > 0.0).sum(dim=-1) == 2).to(current_air_time.dtype)


def standing_commands(commands: torch.Tensor) -> torch.Tensor:
    """Preserve the actor's standing phase gate, including pure yaw commands."""
    return (torch.linalg.vector_norm(commands[:, :2], dim=-1) < 0.01) & (commands[:, 2].abs() < 0.01)


def command_tracking(commands: torch.Tensor, measured: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Track both planar axes jointly and yaw independently, also at stand."""
    if commands.shape != measured.shape or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("commands and measured must have matching [env, 3] shapes.")
    if scale.shape != (3,):
        raise ValueError("scale must contain one value per command axis.")
    squared_error = ((commands - measured) / scale).square()
    return torch.exp(-squared_error[:, :2].sum(-1)), torch.exp(-squared_error[:, 2])


def compute_flat_locomotion_reward_terms(
    inputs: FlatLocomotionRewardInputs, cfg: FlatLocomotionRewardCfg,
) -> dict[str, torch.Tensor]:
    """Return weighted per-second terms; the environment applies dt once."""
    scale = inputs.commands.new_tensor((*cfg.linear_velocity_scales, cfg.yaw_rate_scale))
    measured = torch.cat((inputs.base_linear_velocity_yaw_frame[:, :2], inputs.base_angular_velocity_w[:, 2:3]), dim=-1)
    linear, yaw = command_tracking(inputs.commands, measured, scale)
    raw = {
        "track_lin_vel_xy_exp": linear,
        "track_ang_vel_z_exp": yaw,
        "termination_penalty": inputs.terminated.to(linear.dtype),
        "lin_vel_z_l2": inputs.base_linear_velocity_b[:, 2].square(),
        "ang_vel_xy_l2": inputs.base_angular_velocity_b[:, :2].square().sum(-1),
        "dof_torques_l2": inputs.applied_torque.square().sum(-1),
        "dof_acc_l2": inputs.joint_acc.square().sum(-1),
        "action_rate_l2": (inputs.action - inputs.previous_action).square().sum(-1),
        "feet_air_time": inputs.feet_air_time,
        "both_feet_air": compute_both_feet_air(inputs.feet_current_air_time),
        "feet_slide": inputs.feet_slide,
        "flat_orientation_l2": inputs.projected_gravity_b[:, :2].square().sum(-1),
        "dof_pos_limits": inputs.dof_pos_limits,
        "joint_deviation_hip": inputs.joint_deviation_hip,
        "joint_deviation_arms": inputs.joint_deviation_arms,
        "joint_deviation_torso": inputs.joint_deviation_torso,
    }
    return {name: value * float(getattr(cfg, name)) for name, value in raw.items()}


__all__ = [
    "FlatLocomotionRewardCfg", "FlatLocomotionRewardInputs", "LOCOMOTION_TASK_REWARD_WEIGHTS",
    "compute_flat_locomotion_reward_terms", "compute_both_feet_air", "standing_commands", "command_tracking",
]

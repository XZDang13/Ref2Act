from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StandUpRewardCfg:
    """Compact V5 objective for assisted discovery and unassisted standing."""

    rise_progress: float = 2.0
    stand_quality: float = 8.0
    completion: float = 5.0
    unsafe_termination: float = -2.0
    torque: float = -0.002
    joint_velocity: float = -0.005
    action_rate: float = -0.005
    joint_limit: float = -0.10

    rise_exponent: float = 2.0
    rise_target_upright: float = 0.90
    stand_root_std: float = 0.08
    stand_shoulder_std: float = 0.10
    stand_upright_std: float = 0.12
    default_pose_std: float = 0.25
    linear_velocity_std: float = 0.30
    angular_velocity_std: float = 0.60
    joint_velocity_std: float = 2.0
    support_total_load_ratio: float = 0.70
    support_each_foot_load_ratio: float = 0.10

    def contract(self) -> dict[str, object]:
        return {
            "type": "fixed_supine_stand_up_v5_unassisted_support",
            "rise_objective": "coupled_root_shoulder_upright_progress",
            "stand_objective": "unassisted_gaussian_pose_with_physical_foot_load",
            "feet_support_role": "stand_quality_gate_only",
            "non_foot_support_role": "stand_quality_gate_only",
            "completion": "strict_unassisted_supine_reset_event",
            "persistent_terms_scaled_by_dt": True,
            "success_terminates": False,
            "weights": {
                name: float(getattr(self, name))
                for name in (
                    "rise_progress",
                    "stand_quality",
                    "completion",
                    "unsafe_termination",
                    "torque",
                    "joint_velocity",
                    "action_rate",
                    "joint_limit",
                )
            },
            "scales": {
                name: float(getattr(self, name))
                for name in (
                    "rise_exponent",
                    "rise_target_upright",
                    "stand_root_std",
                    "stand_shoulder_std",
                    "stand_upright_std",
                    "default_pose_std",
                    "linear_velocity_std",
                    "angular_velocity_std",
                    "joint_velocity_std",
                    "support_total_load_ratio",
                    "support_each_foot_load_ratio",
                )
            },
        }


@dataclass(frozen=True)
class StandUpRewardInputs:
    root_height: torch.Tensor
    shoulder_height: torch.Tensor
    rise_reference_root_height: torch.Tensor
    rise_reference_shoulder_height: torch.Tensor
    target_root_height: torch.Tensor
    target_shoulder_height: torch.Tensor
    upright_projection: torch.Tensor
    rise_reference_upright_projection: torch.Tensor
    base_linear_velocity_b: torch.Tensor
    base_angular_velocity_b: torch.Tensor
    joint_position: torch.Tensor
    default_joint_position: torch.Tensor
    soft_joint_position_limits: torch.Tensor
    joint_velocity: torch.Tensor
    joint_velocity_limits: torch.Tensor
    applied_torque: torch.Tensor
    joint_effort_limits: torch.Tensor
    target_joint_position: torch.Tensor
    previous_target_joint_position: torch.Tensor
    left_foot_load_ratio: torch.Tensor
    right_foot_load_ratio: torch.Tensor
    non_foot_contact: torch.Tensor
    unassisted: torch.Tensor
    completion_event: torch.Tensor
    standing_reset: torch.Tensor
    unsafe_termination: torch.Tensor
    settling: torch.Tensor


def normalized_exp_progress(progress: torch.Tensor, *, exponent: float) -> torch.Tensor:
    """Map clipped progress to [0, 1] without saturating before the target."""

    if exponent <= 0.0:
        raise ValueError("progress exponent must be positive")
    clipped = progress.clamp(0.0, 1.0)
    scalar = torch.as_tensor(exponent, device=progress.device, dtype=progress.dtype)
    return torch.expm1(scalar * clipped) / torch.expm1(scalar)


def stand_up_progress_scores(
    inputs: StandUpRewardInputs,
    cfg: StandUpRewardCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return root, shoulder, upright, and fully coupled rise scores."""

    root_denominator = (
        inputs.target_root_height - inputs.rise_reference_root_height
    ).clamp_min(1.0e-3)
    root_progress = (
        inputs.root_height - inputs.rise_reference_root_height
    ) / root_denominator
    shoulder_denominator = (
        inputs.target_shoulder_height - inputs.rise_reference_shoulder_height
    ).clamp_min(1.0e-3)
    shoulder_progress = (
        inputs.shoulder_height - inputs.rise_reference_shoulder_height
    ) / shoulder_denominator
    upright_denominator = (
        float(cfg.rise_target_upright) - inputs.rise_reference_upright_projection
    ).clamp_min(1.0e-3)
    upright_progress = (
        inputs.upright_projection - inputs.rise_reference_upright_projection
    ) / upright_denominator
    root_score = normalized_exp_progress(root_progress, exponent=cfg.rise_exponent)
    shoulder_score = normalized_exp_progress(
        shoulder_progress, exponent=cfg.rise_exponent
    )
    upright_score = normalized_exp_progress(
        upright_progress, exponent=cfg.rise_exponent
    )
    height_score = torch.minimum(root_score, shoulder_score)
    # The harmonic mean stays smooth and bounded while becoming exactly zero
    # when either the weakest height component or uprightness has no progress.
    rise_score = (
        2.0
        * height_score
        * upright_score
        / (height_score + upright_score).clamp_min(1.0e-6)
    )
    return root_score, shoulder_score, upright_score, rise_score


def gaussian_score(error: torch.Tensor, *, std: float) -> torch.Tensor:
    if std <= 0.0:
        raise ValueError("Gaussian std must be positive")
    return torch.exp(-error.square() / float(std) ** 2)


def support_score(
    inputs: StandUpRewardInputs, cfg: StandUpRewardCfg
) -> torch.Tensor:
    """Measure real two-foot loading without rewarding contact on its own."""

    if cfg.support_total_load_ratio <= 0.0:
        raise ValueError("support_total_load_ratio must be positive")
    if cfg.support_each_foot_load_ratio <= 0.0:
        raise ValueError("support_each_foot_load_ratio must be positive")
    left = inputs.left_foot_load_ratio.clamp_min(0.0)
    right = inputs.right_foot_load_ratio.clamp_min(0.0)
    total = ((left + right) / float(cfg.support_total_load_ratio)).clamp(0.0, 1.0)
    both = (
        torch.minimum(left, right) / float(cfg.support_each_foot_load_ratio)
    ).clamp(0.0, 1.0)
    no_other_support = (~inputs.non_foot_contact.bool()).to(total.dtype)
    return total * both * no_other_support


def compute_stand_up_reward_terms(
    inputs: StandUpRewardInputs,
    cfg: StandUpRewardCfg,
    *,
    step_dt: float,
) -> dict[str, torch.Tensor]:
    """Return a compact objective that cannot pay for assisted suspension."""

    if step_dt <= 0.0:
        raise ValueError("step_dt must be positive")

    _, _, _, rise_score = stand_up_progress_scores(inputs, cfg)
    root_score = gaussian_score(
        inputs.root_height - inputs.target_root_height, std=cfg.stand_root_std
    )
    final_shoulder_score = gaussian_score(
        inputs.shoulder_height - inputs.target_shoulder_height,
        std=cfg.stand_shoulder_std,
    )
    final_upright_score = gaussian_score(
        1.0 - inputs.upright_projection.clamp(-1.0, 1.0),
        std=cfg.stand_upright_std,
    )
    posture_score = (
        root_score * final_shoulder_score * final_upright_score
    ).clamp_min(1.0e-12).pow(1.0 / 3.0)

    joint_range = (
        inputs.soft_joint_position_limits[..., 1]
        - inputs.soft_joint_position_limits[..., 0]
    ).clamp_min(1.0e-3)
    pose_error = (
        (inputs.joint_position - inputs.default_joint_position) / joint_range
    ).square().mean(dim=-1)
    pose_score = torch.exp(-pose_error / cfg.default_pose_std**2)
    stability_error = (
        inputs.base_linear_velocity_b.square().sum(dim=-1)
        / cfg.linear_velocity_std**2
        + inputs.base_angular_velocity_b.square().sum(dim=-1)
        / cfg.angular_velocity_std**2
        + inputs.joint_velocity.square().mean(dim=-1)
        / cfg.joint_velocity_std**2
    )
    stability_score = torch.exp(-stability_error)
    final_quality = posture_score * (
        0.70 + 0.20 * stability_score + 0.10 * pose_score
    )
    stand_gate = inputs.unassisted.to(final_quality.dtype) * support_score(inputs, cfg)

    effort = inputs.joint_effort_limits.clamp_min(1.0)
    velocity_limit = inputs.joint_velocity_limits.clamp_min(1.0)
    torque_cost = (inputs.applied_torque / effort).square().mean(dim=-1)
    joint_velocity_cost = (
        inputs.joint_velocity / velocity_limit
    ).square().mean(dim=-1)
    action_rate_cost = (
        (inputs.target_joint_position - inputs.previous_target_joint_position)
        / joint_range
    ).square().mean(dim=-1)
    below = (
        inputs.soft_joint_position_limits[..., 0] - inputs.joint_position
    ).clamp_min(0.0)
    above = (
        inputs.joint_position - inputs.soft_joint_position_limits[..., 1]
    ).clamp_min(0.0)
    joint_limit_cost = ((below + above) / joint_range).square().mean(dim=-1)

    active = (~inputs.settling).to(dtype=inputs.root_height.dtype)
    persistent_scale = active * float(step_dt)
    completion = (
        inputs.completion_event.bool()
        & inputs.unassisted.bool()
        & (~inputs.standing_reset.bool())
    ).to(active.dtype)
    return {
        "rise_progress": cfg.rise_progress * rise_score * persistent_scale,
        "stand_quality": cfg.stand_quality
        * stand_gate
        * final_quality
        * persistent_scale,
        "completion": cfg.completion * completion,
        "torque": cfg.torque * torque_cost * persistent_scale,
        "joint_velocity": cfg.joint_velocity * joint_velocity_cost * persistent_scale,
        "action_rate": cfg.action_rate * action_rate_cost * persistent_scale,
        "joint_limit": cfg.joint_limit * joint_limit_cost * persistent_scale,
        "unsafe_termination": cfg.unsafe_termination
        * inputs.unsafe_termination.to(active.dtype),
    }


__all__ = [
    "StandUpRewardCfg",
    "StandUpRewardInputs",
    "compute_stand_up_reward_terms",
    "gaussian_score",
    "normalized_exp_progress",
    "stand_up_progress_scores",
    "support_score",
]

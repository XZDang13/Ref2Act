from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from .recovery_reward import compute_recovery_reward_terms, smooth_height_gate


def standup_hold_steps(duration_s: float, step_dt: float) -> int:
    """Require at least the requested duration, including non-integral windows."""

    if (
        not math.isfinite(duration_s)
        or not math.isfinite(step_dt)
        or min(duration_s, step_dt) <= 0
    ):
        raise ValueError("Stand-up hold duration and step_dt must be finite and positive.")
    return max(1, math.ceil(duration_s / step_dt - 1.0e-9))


def standup_finite_state(*tensors: torch.Tensor) -> torch.Tensor:
    """Identify valid environments without allowing one bad row to affect others."""

    valid = torch.ones(tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device)
    for value in tensors:
        valid &= torch.isfinite(value).reshape(value.shape[0], -1).all(dim=-1)
    return valid


@dataclass(frozen=True)
class StandUpRewardTerms:
    recovery_height_deficit: torch.Tensor
    recovery_upright_deficit: torch.Tensor
    recovery_hold: torch.Tensor
    recovery_completion: torch.Tensor
    recovery_failure: torch.Tensor
    standing_stability: torch.Tensor
    standing_motion_penalty: torch.Tensor
    regularization: torch.Tensor

    @property
    def recovery(self) -> torch.Tensor:
        return (
            self.recovery_height_deficit
            + self.recovery_upright_deficit
            + self.recovery_hold
            + self.recovery_completion
            + self.recovery_failure
        )

    @property
    def total(self) -> torch.Tensor:
        return (
            self.recovery
            + self.standing_stability
            + self.standing_motion_penalty
            + self.regularization
        )


def smooth_gate(value: torch.Tensor, *, start: float, full: float) -> torch.Tensor:
    """Compatibility alias for the conservative body-height phase gate."""

    return smooth_height_gate(value, start=start, full=full)


def assistance_height_multiplier(
    body_height_score: torch.Tensor, *, start: float, full: float
) -> torch.Tensor:
    """Fade training-only lift out before the robot reaches standing height."""

    return 1.0 - smooth_height_gate(body_height_score, start=start, full=full)


def standup_stable_state(
    *,
    body_height_score: torch.Tensor,
    upright_projection: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_velocity: torch.Tensor,
    minimum_body_height_score: float,
    minimum_upright_projection: float,
    maximum_base_linear_velocity: float,
    maximum_base_angular_velocity: float,
    maximum_joint_velocity: float,
) -> torch.Tensor:
    """Identify a standing-and-still state rather than a transient jump."""

    if body_height_score.shape != upright_projection.shape:
        raise ValueError("Stand-up height and upright tensors must share a shape.")
    batch_size = int(body_height_score.shape[0])
    vector_tensors = (base_linear_velocity, base_angular_velocity, joint_velocity)
    if any(int(value.shape[0]) != batch_size for value in vector_tensors):
        raise ValueError("Stand-up stability tensors must share a batch dimension.")
    if base_linear_velocity.shape[-1] != 3 or base_angular_velocity.shape[-1] != 3:
        raise ValueError("Stand-up base velocity tensors must end in three dimensions.")
    limits = (
        maximum_base_linear_velocity,
        maximum_base_angular_velocity,
        maximum_joint_velocity,
    )
    if any(float(value) <= 0.0 for value in limits):
        raise ValueError("Stand-up stability velocity limits must be positive.")
    return (
        (body_height_score >= float(minimum_body_height_score))
        & (upright_projection >= float(minimum_upright_projection))
        & (
            torch.linalg.vector_norm(base_linear_velocity, dim=-1)
            <= float(maximum_base_linear_velocity)
        )
        & (
            torch.linalg.vector_norm(base_angular_velocity, dim=-1)
            <= float(maximum_base_angular_velocity)
        )
        & (joint_velocity.abs().amax(dim=-1) <= float(maximum_joint_velocity))
    )


def compute_standup_reward_terms(
    *,
    body_height_score: torch.Tensor,
    upright_score: torch.Tensor,
    recovering: torch.Tensor,
    holding: torch.Tensor,
    completed: torch.Tensor,
    failed: torch.Tensor,
    joint_position: torch.Tensor,
    joint_position_lower: torch.Tensor,
    joint_position_upper: torch.Tensor,
    joint_velocity: torch.Tensor,
    joint_velocity_limit: torch.Tensor,
    joint_torque: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_previous_action: torch.Tensor,
    step_dt: float,
    upright_gate_start: float,
    upright_gate_full: float,
    height_deficit_scale: float,
    upright_deficit_scale: float,
    hold_reward_scale: float,
    completion_bonus: float,
    failure_penalty: float,
    stability_reward_scale: float,
    stability_gate_start: float,
    stability_gate_full: float,
    base_linear_velocity_sigma: float,
    base_angular_velocity_sigma: float,
    stability_joint_velocity_sigma: float,
    motion_penalty_scale: float,
    regularization_weight: float,
    torque_scale: float,
    joint_power_scale: float,
    joint_velocity_scale: float,
    action_rate_scale: float,
    action_acceleration_scale: float,
    joint_position_limit_scale: float,
    joint_velocity_limit_scale: float,
    soft_joint_position_limit: float,
    soft_joint_velocity_limit: float,
    valid_state: torch.Tensor | None = None,
) -> StandUpRewardTerms:
    """Recovery objective that makes sustained standing better than jumping.

    Height deficit supplies dense get-up shaping.  There is deliberately no
    frame-to-frame height-progress reward: under discounted returns a quick
    rise followed by a later fall has positive value even when the undiscounted
    height differences cancel.  Once the body is high, a smooth stability
    reward and motion cost distinguish quiet standing from a ballistic apex.
    """

    if step_dt <= 0.0:
        raise ValueError("Stand-up step_dt must be positive.")
    batch_shape = body_height_score.shape
    scalar_tensors = (
        upright_score,
        recovering,
        holding,
        completed,
        failed,
    )
    if any(value.shape != batch_shape for value in scalar_tensors):
        raise ValueError("Stand-up scalar state tensors must share a shape.")
    vector_tensors = (
        joint_position_lower,
        joint_position_upper,
        joint_velocity,
        joint_velocity_limit,
        joint_torque,
    )
    if any(value.shape != joint_position.shape for value in vector_tensors):
        raise ValueError("Stand-up joint tensors must share a shape.")
    if action.shape != previous_action.shape or action.shape != previous_previous_action.shape:
        raise ValueError("Stand-up action-history tensors must share a shape.")
    if base_linear_velocity.shape != (batch_shape[0], 3):
        raise ValueError("Stand-up base linear velocity must have shape [batch, 3].")
    if base_angular_velocity.shape != (batch_shape[0], 3):
        raise ValueError("Stand-up base angular velocity must have shape [batch, 3].")
    if not 0.0 < soft_joint_position_limit <= 1.0:
        raise ValueError("soft_joint_position_limit must be in (0, 1].")
    if not 0.0 < soft_joint_velocity_limit <= 1.0:
        raise ValueError("soft_joint_velocity_limit must be in (0, 1].")
    nonnegative = (
        height_deficit_scale,
        upright_deficit_scale,
        hold_reward_scale,
        completion_bonus,
        failure_penalty,
        stability_reward_scale,
        motion_penalty_scale,
        regularization_weight,
        torque_scale,
        joint_power_scale,
        joint_velocity_scale,
        action_rate_scale,
        action_acceleration_scale,
        joint_position_limit_scale,
        joint_velocity_limit_scale,
    )
    if any(float(value) < 0.0 for value in nonnegative):
        raise ValueError("Stand-up reward and constraint scales must be non-negative.")
    if not 0.0 <= float(stability_gate_start) < float(stability_gate_full) <= 1.0:
        raise ValueError("Stand-up stability gate must satisfy 0 <= start < full <= 1.")
    if any(
        float(value) <= 0.0
        for value in (
            base_linear_velocity_sigma,
            base_angular_velocity_sigma,
            stability_joint_velocity_sigma,
        )
    ):
        raise ValueError("Stand-up stability velocity sigmas must be positive.")

    upright_gate = smooth_height_gate(
        body_height_score,
        start=float(upright_gate_start),
        full=float(upright_gate_full),
    )
    recovery_terms = compute_recovery_reward_terms(
        body_height_score=body_height_score,
        # Disable the shared recovery helper's progress term.  Its plain
        # difference is not potential-based for gamma < 1 and was the source
        # of the repeated launch/fall/launch exploit in this downstream task.
        previous_body_height_score=body_height_score,
        upright_score=upright_score,
        upright_gate=upright_gate,
        recovering=recovering,
        completed=completed,
        failed=failed,
        step_dt=step_dt,
        height_deficit_scale=height_deficit_scale,
        height_progress_scale=0.0,
        upright_deficit_scale=upright_deficit_scale,
        completion_bonus=completion_bonus,
        failure_penalty=failure_penalty,
    )
    recovery_hold = (
        holding.to(dtype=body_height_score.dtype) * float(hold_reward_scale) * float(step_dt)
    )

    stability_gate = smooth_height_gate(
        body_height_score,
        start=float(stability_gate_start),
        full=float(stability_gate_full),
    )
    linear_motion = torch.sum(base_linear_velocity.square(), dim=-1) / float(
        base_linear_velocity_sigma
    ) ** 2
    angular_motion = torch.sum(base_angular_velocity.square(), dim=-1) / float(
        base_angular_velocity_sigma
    ) ** 2
    joint_motion = torch.mean(joint_velocity.square(), dim=-1) / float(
        stability_joint_velocity_sigma
    ) ** 2
    normalized_motion = linear_motion + angular_motion + joint_motion
    # The phase gate saturates early; h**4 preserves an incentive to finish
    # extending instead of collecting the full stability reward in a crouch.
    stability_quality = (
        stability_gate
        * torch.clamp(body_height_score, 0.0, 1.0).pow(4)
        * torch.clamp(upright_score, 0.0, 1.0)
        * torch.exp(-normalized_motion)
    )
    standing_stability = (
        float(stability_reward_scale) * float(step_dt) * stability_quality
    )
    # Bound the cost so an isolated simulator spike cannot dominate PPO's
    # value targets, while ordinary launch velocities remain clearly costly.
    standing_motion_penalty = (
        -float(motion_penalty_scale)
        * float(step_dt)
        * stability_gate
        * torch.clamp(normalized_motion, max=10.0)
    )

    action_rate = action - previous_action
    action_acceleration = action - 2.0 * previous_action + previous_previous_action
    joint_range = torch.clamp(joint_position_upper - joint_position_lower, min=1.0e-6)
    joint_center = 0.5 * (joint_position_upper + joint_position_lower)
    normalized_position = 2.0 * (joint_position - joint_center) / joint_range
    position_limit_cost = torch.clamp(
        normalized_position.abs() - float(soft_joint_position_limit), min=0.0
    ).square().mean(dim=-1)
    normalized_velocity = joint_velocity.abs() / torch.clamp(
        joint_velocity_limit, min=1.0e-6
    )
    velocity_limit_cost = torch.clamp(
        normalized_velocity - float(soft_joint_velocity_limit), min=0.0
    ).square().mean(dim=-1)
    regularization_cost = (
        float(torque_scale) * joint_torque.square().mean(dim=-1)
        + float(joint_power_scale)
        * (joint_torque * joint_velocity).abs().mean(dim=-1)
        + float(joint_velocity_scale) * joint_velocity.square().mean(dim=-1)
        + float(action_rate_scale) * action_rate.square().mean(dim=-1)
        + float(action_acceleration_scale)
        * action_acceleration.square().mean(dim=-1)
        + float(joint_position_limit_scale) * position_limit_cost
        + float(joint_velocity_limit_scale) * velocity_limit_cost
    )
    regularization = (
        -float(regularization_weight) * float(step_dt) * regularization_cost
    )

    terms = StandUpRewardTerms(
        recovery_height_deficit=recovery_terms["height_deficit"],
        recovery_upright_deficit=recovery_terms["upright_deficit"],
        recovery_hold=recovery_hold,
        recovery_completion=recovery_terms["completion"],
        recovery_failure=recovery_terms["failure"],
        standing_stability=standing_stability,
        standing_motion_penalty=standing_motion_penalty,
        regularization=regularization,
    )
    # A failure penalty added to NaN is still NaN. Replace the entire invalid
    # row, including individual logged terms, with a finite failure reward.
    valid = standup_finite_state(
        body_height_score, upright_score, joint_position, joint_position_lower,
        joint_position_upper, joint_velocity, joint_velocity_limit, joint_torque,
        base_linear_velocity, base_angular_velocity, action, previous_action,
        previous_previous_action, terms.total,
    )
    if valid_state is not None:
        if valid_state.shape != batch_shape:
            raise ValueError("Stand-up valid_state must match the scalar state shape.")
        valid &= valid_state
    return StandUpRewardTerms(**{
        field.name: torch.where(
            valid,
            getattr(terms, field.name),
            torch.full_like(body_height_score, -float(failure_penalty))
            if field.name == "recovery_failure" else torch.zeros_like(body_height_score),
        )
        for field in fields(terms)
    })


__all__ = [
    "StandUpRewardTerms",
    "assistance_height_multiplier",
    "compute_standup_reward_terms",
    "smooth_gate",
    "standup_stable_state",
    "standup_finite_state",
    "standup_hold_steps",
]

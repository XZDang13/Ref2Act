from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoveryTrackerUpdate:
    active: torch.Tensor
    elapsed_steps: torch.Tensor
    hold_steps: torch.Tensor
    hold_active: torch.Tensor
    fall_events: torch.Tensor
    completion_events: torch.Tensor


def nominal_shoulder_height_from_relative_geometry(
    *,
    default_root_height: torch.Tensor,
    current_root_height: torch.Tensor,
    current_shoulder_height: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct nominal shoulder height without trusting spawn world Z."""

    if not (
        default_root_height.shape
        == current_root_height.shape
        == current_shoulder_height.shape
    ):
        raise ValueError("Standing-height geometry tensors must share a shape.")
    return default_root_height + current_shoulder_height - current_root_height


def update_recovery_tracker(
    *,
    active: torch.Tensor,
    elapsed_steps: torch.Tensor,
    hold_steps: torch.Tensor,
    fallen: torch.Tensor,
    recovered: torch.Tensor,
    required_hold_steps: int,
) -> RecoveryTrackerUpdate:
    """Advance recovery timing and consecutive-success state by one policy step."""

    if required_hold_steps <= 0:
        raise ValueError("required_hold_steps must be positive.")
    active_now = active | fallen
    fall_events = ~active & fallen
    hold_active = active_now & recovered
    next_hold_steps = torch.where(
        hold_active,
        hold_steps + 1,
        torch.zeros_like(hold_steps),
    )
    completion_events = hold_active & (next_hold_steps >= required_hold_steps)
    next_active = active_now & ~completion_events
    next_elapsed_steps = torch.where(
        next_active,
        elapsed_steps + 1,
        torch.zeros_like(elapsed_steps),
    )
    next_hold_steps = torch.where(
        completion_events,
        torch.zeros_like(next_hold_steps),
        next_hold_steps,
    )
    return RecoveryTrackerUpdate(
        active=next_active,
        elapsed_steps=next_elapsed_steps,
        hold_steps=next_hold_steps,
        hold_active=hold_active,
        fall_events=fall_events,
        completion_events=completion_events,
    )


def recovery_height_scores(
    root_height: torch.Tensor,
    shoulder_height: torch.Tensor,
    *,
    target_root_height: float,
    target_shoulder_height: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score standing height without assuming any particular contact pattern.

    The body score is the conservative minimum of root and shoulder progress.
    Raising only the pelvis or only the upper body therefore cannot mask the
    other incomplete part of the stand-up motion.
    """

    if root_height.shape != shoulder_height.shape:
        raise ValueError("Recovery root and shoulder heights must share a shape.")
    if target_root_height <= 0.0 or target_shoulder_height <= 0.0:
        raise ValueError("Recovery target heights must be positive.")
    root_score = torch.clamp(root_height / float(target_root_height), 0.0, 1.0)
    shoulder_score = torch.clamp(
        shoulder_height / float(target_shoulder_height), 0.0, 1.0
    )
    return root_score, shoulder_score, torch.minimum(root_score, shoulder_score)


def recovery_upright_score(
    upright_projection: torch.Tensor,
    *,
    minimum_upright_projection: float,
    target_upright_projection: float,
) -> torch.Tensor:
    """Normalize pelvis up-axis alignment; a horizontal pelvis scores zero."""

    if target_upright_projection <= minimum_upright_projection:
        raise ValueError(
            "target_upright_projection must exceed minimum_upright_projection."
        )
    return torch.clamp(
        (upright_projection - float(minimum_upright_projection))
        / float(target_upright_projection - minimum_upright_projection),
        0.0,
        1.0,
    )


def smooth_height_gate(
    body_height_score: torch.Tensor,
    *,
    start: float,
    full: float,
) -> torch.Tensor:
    """Activate upright shaping only after the body has substantially risen."""

    if not 0.0 <= start < full <= 1.0:
        raise ValueError("Height gate thresholds must satisfy 0 <= start < full <= 1.")
    normalized = torch.clamp(
        (body_height_score - float(start)) / float(full - start), 0.0, 1.0
    )
    return normalized.square() * (3.0 - 2.0 * normalized)


def smooth_stability_gate(
    body_height_score: torch.Tensor,
    upright_score: torch.Tensor,
    *,
    start: float,
    full: float,
) -> torch.Tensor:
    """Return a continuous gate for restoring hybrid locomotion constraints."""

    if body_height_score.shape != upright_score.shape:
        raise ValueError("Recovery height and upright scores must share a shape.")
    state_score = torch.minimum(body_height_score, upright_score)
    return smooth_height_gate(state_score, start=start, full=full)


def gate_locomotion_reward_terms(
    terms: Mapping[str, torch.Tensor],
    gate: torch.Tensor,
    recovery_scales: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Smoothly interpolate selected locomotion terms from recovery to upright."""

    gated: dict[str, torch.Tensor] = {}
    for name, value in terms.items():
        recovery_scale = float(recovery_scales.get(name, 1.0))
        if not 0.0 <= recovery_scale <= 1.0:
            raise ValueError(f"Recovery scale for {name!r} must be in [0, 1].")
        multiplier = recovery_scale + (1.0 - recovery_scale) * gate
        gated[name] = value * multiplier
    return gated


def compute_recovery_reward_terms(
    *,
    body_height_score: torch.Tensor,
    previous_body_height_score: torch.Tensor,
    upright_score: torch.Tensor,
    upright_gate: torch.Tensor,
    recovering: torch.Tensor,
    completed: torch.Tensor,
    failed: torch.Tensor,
    step_dt: float,
    height_deficit_scale: float,
    height_progress_scale: float,
    upright_deficit_scale: float,
    completion_bonus: float,
    failure_penalty: float,
) -> dict[str, torch.Tensor]:
    """Build the contact-free recovery objective.

    Incomplete static poses pay a bounded height deficit. Signed height progress
    supplies immediate credit without making raise/drop cycles profitable. The
    upright deficit is height-gated so rolling, hand support, knee support and
    other useful early recovery contacts remain unconstrained.
    """

    tensors = (
        previous_body_height_score,
        upright_score,
        upright_gate,
        recovering,
        completed,
        failed,
    )
    if any(value.shape != body_height_score.shape for value in tensors):
        raise ValueError("Recovery reward tensors must share a shape.")
    if step_dt <= 0.0:
        raise ValueError("Recovery step_dt must be positive.")
    scales = (
        height_deficit_scale,
        height_progress_scale,
        upright_deficit_scale,
        completion_bonus,
        failure_penalty,
    )
    if any(float(value) < 0.0 for value in scales):
        raise ValueError("Recovery reward scales must be non-negative.")

    active = recovering.to(dtype=body_height_score.dtype)
    completion = completed.to(dtype=body_height_score.dtype)
    failure = failed.to(dtype=body_height_score.dtype)
    body = torch.clamp(body_height_score, 0.0, 1.0)
    previous = torch.clamp(previous_body_height_score, 0.0, 1.0)
    upright = torch.clamp(upright_score, 0.0, 1.0)
    gate = torch.clamp(upright_gate, 0.0, 1.0)
    return {
        "height_deficit": active
        * -float(height_deficit_scale)
        * (1.0 - body)
        * float(step_dt),
        "height_progress": active
        * float(height_progress_scale)
        * (body - previous),
        "upright_deficit": active
        * -float(upright_deficit_scale)
        * gate
        * (1.0 - upright)
        * float(step_dt),
        "completion": completion * float(completion_bonus),
        "failure": failure * -float(failure_penalty),
    }


def linear_assistance_ratio(
    iteration: int,
    total_iterations: int,
    *,
    maximum_gravity_ratio: float,
    anneal_fraction: float,
) -> float:
    """Linearly remove training-only lift and reserve a final zero-force phase."""

    if iteration < 0 or total_iterations <= 0:
        raise ValueError("Assistance schedule iterations must be non-negative/positive.")
    if maximum_gravity_ratio < 0.0:
        raise ValueError("maximum_gravity_ratio must be non-negative.")
    if not 0.0 < anneal_fraction <= 1.0:
        raise ValueError("anneal_fraction must be in (0, 1].")
    anneal_iterations = max(1, round(total_iterations * float(anneal_fraction)))
    progress = min(float(iteration) / float(anneal_iterations), 1.0)
    return float(maximum_gravity_ratio) * (1.0 - progress)


__all__ = [
    "RecoveryTrackerUpdate",
    "compute_recovery_reward_terms",
    "gate_locomotion_reward_terms",
    "linear_assistance_ratio",
    "nominal_shoulder_height_from_relative_geometry",
    "recovery_height_scores",
    "recovery_upright_score",
    "smooth_height_gate",
    "smooth_stability_gate",
    "update_recovery_tracker",
]

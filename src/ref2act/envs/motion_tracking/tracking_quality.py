from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch

from .termination import FailureRule, TerminationContext


class TrackingQuality(IntEnum):
    OK = 0
    SOFT_VIOLATION = 1
    RECOVERY_NEEDED = 2
    HARD_TRACKING_FAILURE = 3


@dataclass(frozen=True)
class TrackingQualityGateCfg:
    enabled: bool = False
    soft_threshold: float = 1.0
    recovery_enter_threshold: float = 1.8
    recovery_exit_threshold: float = 1.2
    hard_tracking_threshold: float | None = None
    min_recovery_steps: int = 10
    min_soft_violation_steps: int = 1
    record_soft_violations: bool = False
    record_recovery_needed: bool = True
    record_hard_tracking_failures: bool = True
    aggregation: str = "max"
    anchor_pos_weight: float = 1.0
    anchor_ori_weight: float = 1.0
    end_effector_pos_weight: float = 1.0
    log_per_rule_errors: bool = True
    log_quality_counts: bool = True

    def __post_init__(self) -> None:
        if self.soft_threshold < 0.0:
            raise ValueError("soft_threshold must be non-negative.")
        if self.recovery_enter_threshold < self.soft_threshold:
            raise ValueError("recovery_enter_threshold must be >= soft_threshold.")
        if self.recovery_exit_threshold < 0.0:
            raise ValueError("recovery_exit_threshold must be non-negative.")
        if self.hard_tracking_threshold is not None and self.hard_tracking_threshold < 0.0:
            raise ValueError("hard_tracking_threshold must be non-negative when configured.")
        if self.min_recovery_steps < 0:
            raise ValueError("min_recovery_steps must be non-negative.")
        if self.min_soft_violation_steps < 1:
            raise ValueError("min_soft_violation_steps must be >= 1.")
        if self.aggregation not in {"max", "weighted_sum"}:
            raise ValueError(f"Unsupported tracking quality aggregation: {self.aggregation}")
        if self.anchor_pos_weight < 0.0 or self.anchor_ori_weight < 0.0 or self.end_effector_pos_weight < 0.0:
            raise ValueError("tracking quality aggregation weights must be non-negative.")
        if (
            self.aggregation == "weighted_sum"
            and self.anchor_pos_weight + self.anchor_ori_weight + self.end_effector_pos_weight <= 0.0
        ):
            raise ValueError("weighted_sum aggregation requires at least one positive weight.")


@dataclass(frozen=True)
class RobustTrackingCfg:
    enabled: bool = False
    quality_gate: TrackingQualityGateCfg = field(default_factory=TrackingQualityGateCfg)


@dataclass(frozen=True)
class TrackingQualityResult:
    state: torch.Tensor
    score: torch.Tensor
    previous_score: torch.Tensor
    soft_violation_mask: torch.Tensor
    recovery_needed_mask: torch.Tensor
    hard_tracking_failure_mask: torch.Tensor
    record_failure_mask: torch.Tensor
    per_rule_errors: dict[str, torch.Tensor]
    per_rule_normalized_errors: dict[str, torch.Tensor]


class TrackingQualityGate:
    def __init__(
        self,
        cfg: TrackingQualityGateCfg,
        termination_model: Any,
        num_envs: int,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.failure_rules: tuple[FailureRule, ...] = tuple(termination_model.failure_rules)
        self.state = torch.full(
            (num_envs,),
            int(TrackingQuality.OK),
            dtype=torch.long,
            device=device,
        )
        self.recovery_needed_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.soft_violation_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.score = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.has_score = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.state[env_ids] = int(TrackingQuality.OK)
        self.recovery_needed_steps[env_ids] = 0
        self.soft_violation_steps[env_ids] = 0
        self.score[env_ids] = 0.0
        self.has_score[env_ids] = False

    def evaluate(self, context: TerminationContext) -> TrackingQualityResult:
        per_rule_errors: dict[str, torch.Tensor] = {}
        per_rule_normalized_errors: dict[str, torch.Tensor] = {}
        normalized_errors: list[torch.Tensor] = []
        weights: list[float] = []

        for rule in self.failure_rules:
            error = self._reduce_rule_error(rule, rule.error(context))
            threshold = float(rule.threshold)
            if threshold <= 0.0:
                raise ValueError(f"Tracking quality rule threshold must be positive: {rule.id}")

            normalized = error / error.new_tensor(threshold)
            per_rule_errors[rule.id] = error
            per_rule_normalized_errors[rule.id] = normalized
            normalized_errors.append(normalized)
            weights.append(self._rule_weight(rule))

        if normalized_errors:
            score = self._aggregate_score(normalized_errors, weights)
        else:
            score = torch.zeros_like(context.episode_length_buf, dtype=torch.float32)
        previous_score = torch.where(self.has_score, self.score, score)

        raw_recovery = score >= self.cfg.recovery_enter_threshold
        raw_soft = score >= self.cfg.soft_threshold
        currently_recovery = self.state == int(TrackingQuality.RECOVERY_NEEDED)

        can_exit_recovery = (
            (score < self.cfg.recovery_exit_threshold)
            & (self.recovery_needed_steps >= self.cfg.min_recovery_steps)
        )
        recovery_needed = torch.where(currently_recovery, ~can_exit_recovery, raw_recovery)

        hard_tracking = torch.zeros_like(recovery_needed)
        if self.cfg.hard_tracking_threshold is not None:
            hard_tracking = score >= self.cfg.hard_tracking_threshold

        self.soft_violation_steps[raw_soft & ~recovery_needed & ~hard_tracking] += 1
        self.soft_violation_steps[~(raw_soft & ~recovery_needed & ~hard_tracking)] = 0
        soft_violation = (
            raw_soft
            & ~recovery_needed
            & ~hard_tracking
            & (self.soft_violation_steps >= self.cfg.min_soft_violation_steps)
        )

        state = torch.full_like(self.state, int(TrackingQuality.OK))
        state[soft_violation] = int(TrackingQuality.SOFT_VIOLATION)
        state[recovery_needed] = int(TrackingQuality.RECOVERY_NEEDED)
        state[hard_tracking] = int(TrackingQuality.HARD_TRACKING_FAILURE)

        self.state[:] = state
        self.recovery_needed_steps[recovery_needed] += 1
        self.recovery_needed_steps[~recovery_needed] = 0
        self.score[:] = score
        self.has_score[:] = True

        record_failure = (
            (soft_violation & self.cfg.record_soft_violations)
            | (recovery_needed & self.cfg.record_recovery_needed)
            | (hard_tracking & self.cfg.record_hard_tracking_failures)
        )

        return TrackingQualityResult(
            state=state.clone(),
            score=score,
            previous_score=previous_score.clone(),
            soft_violation_mask=soft_violation,
            recovery_needed_mask=recovery_needed,
            hard_tracking_failure_mask=hard_tracking,
            record_failure_mask=record_failure,
            per_rule_errors=per_rule_errors,
            per_rule_normalized_errors=per_rule_normalized_errors,
        )

    def _aggregate_score(self, normalized_errors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        stacked = torch.stack(normalized_errors, dim=0)
        if self.cfg.aggregation == "max":
            return stacked.max(dim=0).values
        if self.cfg.aggregation == "weighted_sum":
            weight_tensor = stacked.new_tensor(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
            weight_sum = torch.clamp(weight_tensor.sum(), min=torch.finfo(stacked.dtype).eps)
            return (stacked * weight_tensor).sum(dim=0) / weight_sum
        raise ValueError(f"Unsupported tracking quality aggregation: {self.cfg.aggregation}")

    def _reduce_rule_error(self, rule: FailureRule, error: torch.Tensor) -> torch.Tensor:
        if error.ndim <= 1:
            return error

        flattened = error.reshape(error.shape[0], -1)
        reduction = getattr(rule, "reduction", "any")
        if reduction == "any":
            return flattened.max(dim=1).values
        if reduction == "all":
            return flattened.min(dim=1).values
        raise ValueError(f"Unsupported end effector reduction: {reduction}")

    def _rule_weight(self, rule: FailureRule) -> float:
        if rule.id == "anchor_position_failure":
            return float(self.cfg.anchor_pos_weight)
        if rule.id == "anchor_orientation_failure":
            return float(self.cfg.anchor_ori_weight)
        if rule.id == "end_effector_position_failure":
            return float(self.cfg.end_effector_pos_weight)
        return 1.0

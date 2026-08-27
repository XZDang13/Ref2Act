from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch
from isaaclab.assets import Articulation
from ref2act.common.math import quat_apply_inverse
from ref2act.isaac_compat import to_torch
from ref2act.motion.sampling import MotionSampler

from .types import ReferenceMotions


@dataclass
class TerminationRuleCfg:
    id: str
    type: str
    enabled: bool = True


@dataclass
class EpisodeLengthTimeoutRuleCfg(TerminationRuleCfg):
    id: str = "episode_length_timeout"
    type: str = "episode_length_timeout"


@dataclass
class EndOfMotionTimeoutRuleCfg(TerminationRuleCfg):
    id: str = "end_of_motion_timeout"
    type: str = "end_of_motion_timeout"


@dataclass
class AnchorPositionFailureRuleCfg(TerminationRuleCfg):
    id: str = "anchor_position_failure"
    type: str = "anchor_position_failure"
    anchor_body_index: int = -1
    threshold: float = 0.25
    height_only: bool = True


@dataclass
class AnchorOrientationFailureRuleCfg(TerminationRuleCfg):
    id: str = "anchor_orientation_failure"
    type: str = "anchor_orientation_failure"
    anchor_body_index: int = -1
    threshold: float = 0.8


@dataclass
class EndEffectorPositionFailureRuleCfg(TerminationRuleCfg):
    id: str = "end_effector_position_failure"
    type: str = "end_effector_position_failure"
    end_effector_body_indices: tuple[int, ...] = ()
    threshold: float = 0.25
    height_only: bool = False
    reduction: str = "any"


@dataclass
class ProbabilisticRecoveryTerminationCfg:
    """Optional recovery window for tracking failures.

    Existing rule thresholds are soft thresholds. ``hard_thresholds`` retain
    immediate termination for clearly unrecoverable tracking errors.
    """

    enabled: bool = False
    grace_period_s: float = 0.5
    time_ramp_s: float = 1.0
    max_hazard_per_s: float = 2.0
    error_weight: float = 0.65
    time_weight: float = 0.35
    error_exponent: float = 2.0
    time_exponent: float = 2.0
    recovery_decay: float = 2.0
    hard_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "anchor_position_failure": 0.50,
            "anchor_orientation_failure": 1.40,
            "end_effector_position_failure": 0.60,
        }
    )


@dataclass
class TerminationSpec:
    timeout_rules: tuple[TerminationRuleCfg, ...]
    failure_rules: tuple[TerminationRuleCfg, ...]
    probabilistic_recovery: ProbabilisticRecoveryTerminationCfg = field(
        default_factory=ProbabilisticRecoveryTerminationCfg
    )


@dataclass
class TerminationContext:
    episode_length_buf: torch.Tensor
    max_episode_length: torch.Tensor
    robot: Articulation
    reference_motion: ReferenceMotions
    sampler: MotionSampler
    _cache: dict[tuple, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def anchor_pos_error(self, anchor_body_index: int, *, height_only: bool) -> torch.Tensor:
        cache_key = ("anchor_pos_error", anchor_body_index, height_only)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        robot_pos = to_torch(self.robot.data.body_link_pos_w)[:, anchor_body_index]
        ref_pos = self.reference_motion.body_positions[:, anchor_body_index]
        diff = robot_pos - ref_pos
        if height_only:
            result = diff[..., 2].abs()
        else:
            result = torch.norm(diff, dim=-1)
        self._cache[cache_key] = result
        return result

    def anchor_ori_error(self, anchor_body_index: int) -> torch.Tensor:
        cache_key = ("anchor_ori_error", anchor_body_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        robot_anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[:, anchor_body_index]
        reference_anchor_quat_w = self.reference_motion.body_quaternions[:, anchor_body_index]

        gravity_vec_w = to_torch(self.robot.data.GRAVITY_VEC_W)
        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat_w, gravity_vec_w)
        reference_projected_gravity_b = quat_apply_inverse(reference_anchor_quat_w, gravity_vec_w)
        robot_projected_gravity_b = torch.nn.functional.normalize(robot_projected_gravity_b, dim=-1)
        reference_projected_gravity_b = torch.nn.functional.normalize(reference_projected_gravity_b, dim=-1)
        # A z-only comparison cannot distinguish equal tilt magnitudes in
        # different directions.  ``1 - dot`` is yaw invariant, observes the
        # full tilt direction, and preserves the old scale for an upright
        # reference (1 - cos(theta)).
        result = (
            1.0 - torch.sum(robot_projected_gravity_b * reference_projected_gravity_b, dim=-1)
        ).clamp_(0.0, 2.0)
        self._cache[cache_key] = result
        return result

    def end_effector_pos_error(
        self,
        end_effector_body_indices: tuple[int, ...],
        *,
        height_only: bool,
    ) -> torch.Tensor:
        cache_key = ("end_effector_pos_error", end_effector_body_indices, height_only)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        robot_pos = to_torch(self.robot.data.body_link_pos_w)[:, end_effector_body_indices]
        ref_pos = self.reference_motion.body_pos_relative[:, end_effector_body_indices]
        diff = robot_pos - ref_pos
        if height_only:
            result = diff[..., 2].abs()
        else:
            result = torch.norm(diff, dim=-1)
        self._cache[cache_key] = result
        return result


class TimeoutRule(Protocol):
    id: str

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        ...


class FailureRule(Protocol):
    id: str
    threshold: float

    def error(self, context: TerminationContext) -> torch.Tensor:
        ...

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        ...


class EpisodeLengthTimeoutRule:
    def __init__(self, cfg: EpisodeLengthTimeoutRuleCfg) -> None:
        self.id = cfg.id

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        return context.episode_length_buf >= (context.max_episode_length - 1)


class EndOfMotionTimeoutRule:
    def __init__(self, cfg: EndOfMotionTimeoutRuleCfg) -> None:
        self.id = cfg.id

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        return context.sampler.current_times >= context.sampler.get_current_durations()


class AnchorPositionFailureRule:
    def __init__(self, cfg: AnchorPositionFailureRuleCfg) -> None:
        self.id = cfg.id
        self.anchor_body_index = cfg.anchor_body_index
        self.height_only = cfg.height_only
        self.threshold = cfg.threshold

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.anchor_pos_error(self.anchor_body_index, height_only=self.height_only)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        return error > self.threshold


class AnchorOrientationFailureRule:
    def __init__(self, cfg: AnchorOrientationFailureRuleCfg) -> None:
        self.id = cfg.id
        self.anchor_body_index = cfg.anchor_body_index
        self.threshold = cfg.threshold

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.anchor_ori_error(self.anchor_body_index)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        return error > self.threshold


class EndEffectorPositionFailureRule:
    def __init__(self, cfg: EndEffectorPositionFailureRuleCfg) -> None:
        self.id = cfg.id
        self.end_effector_body_indices = cfg.end_effector_body_indices
        self.height_only = cfg.height_only
        self.reduction = cfg.reduction
        self.threshold = cfg.threshold

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.end_effector_pos_error(self.end_effector_body_indices, height_only=self.height_only)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        hits = error > self.threshold
        if self.reduction == "any":
            return hits.any(dim=1)
        if self.reduction == "all":
            return hits.all(dim=1)
        raise ValueError(f"Unsupported end effector reduction: {self.reduction}")


TIMEOUT_RULE_REGISTRY = {
    "episode_length_timeout": EpisodeLengthTimeoutRule,
    "end_of_motion_timeout": EndOfMotionTimeoutRule,
}

FAILURE_RULE_REGISTRY = {
    "anchor_position_failure": AnchorPositionFailureRule,
    "anchor_orientation_failure": AnchorOrientationFailureRule,
    "end_effector_position_failure": EndEffectorPositionFailureRule,
}


def default_termination_spec(
    *,
    anchor_height_only: bool = True,
    end_effector_height_only: bool = False,
) -> TerminationSpec:
    return TerminationSpec(
        timeout_rules=(
            EpisodeLengthTimeoutRuleCfg(),
            EndOfMotionTimeoutRuleCfg(),
        ),
        failure_rules=(
            AnchorPositionFailureRuleCfg(height_only=anchor_height_only),
            AnchorOrientationFailureRuleCfg(),
            EndEffectorPositionFailureRuleCfg(height_only=end_effector_height_only),
        ),
    )


class Termination:
    def __init__(self, spec: TerminationSpec, *, step_dt: float = 0.02) -> None:
        self.spec = spec
        self.step_dt = float(step_dt)
        self.timeout_rules = [
            TIMEOUT_RULE_REGISTRY[rule_cfg.type](rule_cfg)
            for rule_cfg in spec.timeout_rules
            if rule_cfg.enabled
        ]
        self.failure_rules = [
            FAILURE_RULE_REGISTRY[rule_cfg.type](rule_cfg)
            for rule_cfg in spec.failure_rules
            if rule_cfg.enabled
        ]
        self.terminated_env_ids = torch.empty(0, dtype=torch.long)
        self.offtrack_time: torch.Tensor | None = None
        self.previous_offtrack: torch.Tensor | None = None
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self.last_recovery_state: dict[str, torch.Tensor] = {}
        self._validate_probabilistic_recovery()

    def _validate_probabilistic_recovery(self) -> None:
        cfg = self.spec.probabilistic_recovery
        if not cfg.enabled:
            return
        values = {
            "step_dt": self.step_dt,
            "grace_period_s": cfg.grace_period_s,
            "time_ramp_s": cfg.time_ramp_s,
            "max_hazard_per_s": cfg.max_hazard_per_s,
            "error_exponent": cfg.error_exponent,
            "time_exponent": cfg.time_exponent,
            "recovery_decay": cfg.recovery_decay,
        }
        for name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Probabilistic recovery {name} must be finite and > 0.")
        if cfg.error_weight < 0.0 or cfg.time_weight < 0.0:
            raise ValueError("Probabilistic recovery weights must be non-negative.")
        if cfg.error_weight + cfg.time_weight <= 0.0:
            raise ValueError("At least one probabilistic recovery weight must be positive.")
        if not self.failure_rules:
            raise ValueError("Probabilistic recovery requires at least one enabled failure rule.")
        for rule in self.failure_rules:
            hard = cfg.hard_thresholds.get(rule.id)
            if hard is None:
                raise ValueError(f"Missing hard threshold for failure rule {rule.id!r}.")
            if not math.isfinite(hard) or hard <= float(rule.threshold):
                raise ValueError(
                    f"Hard threshold for {rule.id!r} must exceed its soft threshold "
                    f"({hard} <= {float(rule.threshold)})."
                )

    def get_failure_rule(self, rule_id: str) -> FailureRule:
        for rule in self.failure_rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(f"Unknown termination failure rule: {rule_id}")

    def build_context(
        self,
        episode_length_buf: torch.Tensor,
        max_episode_length: torch.Tensor,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        sampler: MotionSampler,
    ) -> TerminationContext:
        return TerminationContext(
            episode_length_buf=episode_length_buf,
            max_episode_length=max_episode_length,
            robot=robot,
            reference_motion=reference_motion,
            sampler=sampler,
        )

    def evaluate_timeouts(self, context: TerminationContext) -> torch.Tensor:
        time_out = torch.zeros_like(context.episode_length_buf, dtype=torch.bool)
        for rule in self.timeout_rules:
            time_out |= rule.evaluate(context)
        return time_out

    def evaluate_failures(self, context: TerminationContext) -> torch.Tensor:
        """Evaluate the original immediate-threshold termination rules."""
        terminate = torch.zeros_like(context.episode_length_buf, dtype=torch.bool)
        for rule in self.failure_rules:
            terminate |= rule.evaluate(context)
        return terminate

    @staticmethod
    def _reduce_error(rule: FailureRule, error: torch.Tensor) -> torch.Tensor:
        while error.ndim > 1:
            error = (
                error.amin(dim=-1)
                if getattr(rule, "reduction", "any") == "all"
                else error.amax(dim=-1)
            )
        return error

    def _ensure_recovery_state(self, template: torch.Tensor) -> None:
        if (
            self.offtrack_time is None
            or self.offtrack_time.shape != template.shape
            or self.offtrack_time.device != template.device
        ):
            self.offtrack_time = torch.zeros_like(template, dtype=torch.float32)
            self.previous_offtrack = torch.zeros_like(template, dtype=torch.bool)

    def evaluate_probabilistic_recovery_failures(
        self,
        context: TerminationContext,
    ) -> torch.Tensor:
        """Evaluate severity/time hazard while preserving hard thresholds."""
        cfg = self.spec.probabilistic_recovery
        self._ensure_recovery_state(context.episode_length_buf)
        assert self.offtrack_time is not None and self.previous_offtrack is not None

        severities = []
        hard_hits = torch.zeros_like(context.episode_length_buf, dtype=torch.bool)
        for rule in self.failure_rules:
            error = self._reduce_error(rule, rule.error(context)).to(dtype=torch.float32)
            soft = float(rule.threshold)
            hard = float(cfg.hard_thresholds[rule.id])
            if hard <= soft:
                raise RuntimeError(
                    f"Runtime soft threshold for {rule.id!r} reached its hard threshold "
                    f"({soft} >= {hard})."
                )
            severities.append(((error - soft) / (hard - soft)).clamp(0.0, 1.0))
            hard_hits |= error >= hard

        severity = torch.stack(severities).amax(dim=0)
        offtrack = severity > 0.0
        offtrack_entries = offtrack & ~self.previous_offtrack
        recovered = self.previous_offtrack & ~offtrack
        self.offtrack_time = torch.where(
            offtrack,
            self.offtrack_time + self.step_dt,
            (self.offtrack_time - cfg.recovery_decay * self.step_dt).clamp_min(0.0),
        )
        time_severity = (
            (self.offtrack_time - cfg.grace_period_s) / cfg.time_ramp_s
        ).clamp(0.0, 1.0)
        normalized_hazard = (
            cfg.error_weight * severity.pow(cfg.error_exponent)
            + cfg.time_weight * time_severity.pow(cfg.time_exponent)
        ).clamp(0.0, 1.0)
        hazard = cfg.max_hazard_per_s * normalized_hazard
        probability = -torch.expm1(-hazard * self.step_dt)
        grace_complete = self.offtrack_time >= cfg.grace_period_s
        effective_hazard = torch.where(
            offtrack & grace_complete,
            hazard,
            torch.zeros_like(hazard),
        )
        effective_probability = torch.where(
            offtrack & grace_complete,
            probability,
            torch.zeros_like(probability),
        )
        probabilistic_hits = torch.rand_like(effective_probability) < effective_probability
        soft_hits = probabilistic_hits & ~hard_hits
        terminate = hard_hits | soft_hits
        self.last_recovery_state = {
            "severity": severity.detach(),
            "offtrack": offtrack.detach(),
            "offtrack_entries": offtrack_entries.detach(),
            "recovered": recovered.detach(),
            "hazard": hazard.detach(),
            "effective_hazard": effective_hazard.detach(),
            "effective_probability": effective_probability.detach(),
            "hard_hits": hard_hits.detach(),
            "probabilistic_hits": soft_hits.detach(),
        }
        self.last_diagnostics = {
            "severity_mean": severity.mean(),
            "severity_max": severity.max(),
            "offtrack_fraction": offtrack.float().mean(),
            "grace_fraction": (offtrack & ~grace_complete).float().mean(),
            "offtrack_time_mean_s": self.offtrack_time.mean(),
            "offtrack_time_max_s": self.offtrack_time.max(),
            "hazard_mean_per_s": hazard.mean(),
            "termination_probability_mean": effective_probability.mean(),
            "hard_terminations": hard_hits.sum(),
            "probabilistic_terminations": soft_hits.sum(),
            "recoveries": recovered.sum(),
        }
        self.previous_offtrack = offtrack.clone()
        return terminate

    def get_dones(
        self,
        episode_length_buf: torch.Tensor,
        max_episode_length: torch.Tensor,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        sampler: MotionSampler,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.build_context(
            episode_length_buf=episode_length_buf,
            max_episode_length=max_episode_length,
            robot=robot,
            reference_motion=reference_motion,
            sampler=sampler,
        )
        time_out = self.evaluate_timeouts(context)
        if self.spec.probabilistic_recovery.enabled:
            terminate = self.evaluate_probabilistic_recovery_failures(context)
            assert self.offtrack_time is not None and self.previous_offtrack is not None
            done = terminate | time_out
            self.offtrack_time[done] = 0.0
            self.previous_offtrack[done] = False
        else:
            self.last_recovery_state = {}
            terminate = self.evaluate_failures(context)

        self.track_terminated_env_ids(terminate)
        return terminate, time_out

    def track_terminated_env_ids(self, failed: torch.Tensor) -> torch.Tensor:
        terminated_env_ids = torch.nonzero(failed, as_tuple=False).squeeze(-1)
        self.terminated_env_ids = terminated_env_ids
        return terminated_env_ids

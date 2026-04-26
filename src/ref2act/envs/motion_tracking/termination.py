from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from ref2act.motion.sampling import MotionSampler

from .types import ReferenceMotions


@dataclass(frozen=True)
class ThresholdPolicyCfg:
    probabilistic: bool = False
    ramp_multiplier: float = 2.0
    sigmoid_steepness: float = 8.0

    def __post_init__(self) -> None:
        if self.ramp_multiplier <= 1.0:
            raise ValueError("error_termination_ramp_multiplier must be greater than 1.0.")
        if self.sigmoid_steepness <= 0.0:
            raise ValueError("error_termination_sigmoid_steepness must be positive.")


@dataclass(frozen=True)
class TerminationRuleCfg:
    id: str
    type: str
    enabled: bool = True


@dataclass(frozen=True)
class EpisodeLengthTimeoutRuleCfg(TerminationRuleCfg):
    id: str = "episode_length_timeout"
    type: str = "episode_length_timeout"


@dataclass(frozen=True)
class EndOfMotionTimeoutRuleCfg(TerminationRuleCfg):
    id: str = "end_of_motion_timeout"
    type: str = "end_of_motion_timeout"


@dataclass(frozen=True)
class AnchorPositionFailureRuleCfg(TerminationRuleCfg):
    id: str = "anchor_position_failure"
    type: str = "anchor_position_failure"
    anchor_body_index: int = -1
    threshold: float = 0.25
    height_only: bool = True
    policy: ThresholdPolicyCfg = ThresholdPolicyCfg()


@dataclass(frozen=True)
class AnchorOrientationFailureRuleCfg(TerminationRuleCfg):
    id: str = "anchor_orientation_failure"
    type: str = "anchor_orientation_failure"
    anchor_body_index: int = -1
    threshold: float = 0.8
    policy: ThresholdPolicyCfg = ThresholdPolicyCfg()


@dataclass(frozen=True)
class EndEffectorPositionFailureRuleCfg(TerminationRuleCfg):
    id: str = "end_effector_position_failure"
    type: str = "end_effector_position_failure"
    end_effector_body_indices: tuple[int, ...] = ()
    threshold: float = 0.15
    height_only: bool = False
    reduction: str = "any"
    policy: ThresholdPolicyCfg = ThresholdPolicyCfg()


@dataclass(frozen=True)
class TerminationSpec:
    timeout_rules: tuple[TerminationRuleCfg, ...]
    failure_rules: tuple[TerminationRuleCfg, ...]


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

        robot_pos = self.robot.data.body_pos_w[:, anchor_body_index]
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

        robot_anchor_quat_w = self.robot.data.body_quat_w[:, anchor_body_index]
        reference_anchor_quat_w = self.reference_motion.body_quaternions[:, anchor_body_index]

        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        reference_projected_gravity_b = quat_apply_inverse(reference_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        result = torch.abs(robot_projected_gravity_b[:, 2] - reference_projected_gravity_b[:, 2])
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

        robot_pos = self.robot.data.body_pos_w[:, end_effector_body_indices]
        ref_pos = self.reference_motion.body_pos_relative[:, end_effector_body_indices]
        diff = robot_pos - ref_pos
        if height_only:
            result = diff[..., 2].abs()
        else:
            result = torch.norm(diff, dim=-1)
        self._cache[cache_key] = result
        return result


class ThresholdPolicy:
    def __init__(self, cfg: ThresholdPolicyCfg) -> None:
        self.cfg = cfg

    def _error_to_termination_probability(self, error: torch.Tensor, threshold: float) -> torch.Tensor:
        threshold_tensor = error.new_tensor(threshold)
        over_threshold = error > threshold_tensor
        if not torch.any(over_threshold):
            return torch.zeros_like(error)

        ramp_span = threshold_tensor * (self.cfg.ramp_multiplier - 1.0)
        normalized_error = torch.clamp((error - threshold_tensor) / ramp_span, min=0.0, max=1.0)

        steepness = error.new_tensor(self.cfg.sigmoid_steepness)
        sigmoid_values = torch.sigmoid(steepness * (normalized_error - 0.5))
        sigmoid_start = torch.sigmoid(-0.5 * steepness)
        sigmoid_end = torch.sigmoid(0.5 * steepness)
        normalized_sigmoid = (sigmoid_values - sigmoid_start) / (sigmoid_end - sigmoid_start)

        probabilities = torch.where(over_threshold, normalized_sigmoid, torch.zeros_like(error))
        return probabilities.clamp_(0.0, 1.0)

    def evaluate(self, error: torch.Tensor, threshold: float) -> torch.Tensor:
        if not self.cfg.probabilistic:
            return error > threshold
        probabilities = self._error_to_termination_probability(error, threshold)
        return torch.rand_like(probabilities) < probabilities


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
        self.policy = ThresholdPolicy(cfg.policy)

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.anchor_pos_error(self.anchor_body_index, height_only=self.height_only)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        return self.policy.evaluate(error, self.threshold)


class AnchorOrientationFailureRule:
    def __init__(self, cfg: AnchorOrientationFailureRuleCfg) -> None:
        self.id = cfg.id
        self.anchor_body_index = cfg.anchor_body_index
        self.threshold = cfg.threshold
        self.policy = ThresholdPolicy(cfg.policy)

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.anchor_ori_error(self.anchor_body_index)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        return self.policy.evaluate(error, self.threshold)


class EndEffectorPositionFailureRule:
    def __init__(self, cfg: EndEffectorPositionFailureRuleCfg) -> None:
        self.id = cfg.id
        self.end_effector_body_indices = cfg.end_effector_body_indices
        self.height_only = cfg.height_only
        self.reduction = cfg.reduction
        self.threshold = cfg.threshold
        self.policy = ThresholdPolicy(cfg.policy)

    def error(self, context: TerminationContext) -> torch.Tensor:
        return context.end_effector_pos_error(self.end_effector_body_indices, height_only=self.height_only)

    def evaluate(self, context: TerminationContext) -> torch.Tensor:
        error = self.error(context)
        hits = self.policy.evaluate(error, self.threshold)
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
    probabilistic_error_termination: bool = False,
    error_termination_ramp_multiplier: float = 2.0,
    error_termination_sigmoid_steepness: float = 8.0,
) -> TerminationSpec:
    policy = ThresholdPolicyCfg(
        probabilistic=probabilistic_error_termination,
        ramp_multiplier=error_termination_ramp_multiplier,
        sigmoid_steepness=error_termination_sigmoid_steepness,
    )
    return TerminationSpec(
        timeout_rules=(
            EpisodeLengthTimeoutRuleCfg(),
            EndOfMotionTimeoutRuleCfg(),
        ),
        failure_rules=(
            AnchorPositionFailureRuleCfg(height_only=anchor_height_only, policy=policy),
            AnchorOrientationFailureRuleCfg(policy=policy),
            EndEffectorPositionFailureRuleCfg(height_only=end_effector_height_only, policy=policy),
        ),
    )


class Termination:
    def __init__(self, spec: TerminationSpec) -> None:
        self.spec = spec
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
        terminate = torch.zeros_like(context.episode_length_buf, dtype=torch.bool)
        for rule in self.failure_rules:
            terminate |= rule.evaluate(context)
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
        terminate = self.evaluate_failures(context)

        self.track_terminated_env_ids(terminate)
        return terminate, time_out

    def track_terminated_env_ids(self, failed: torch.Tensor) -> torch.Tensor:
        terminated_env_ids = torch.nonzero(failed, as_tuple=False).squeeze(-1)
        self.terminated_env_ids = terminated_env_ids
        return terminated_env_ids

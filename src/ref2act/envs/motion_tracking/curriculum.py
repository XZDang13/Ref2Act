from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .termination import Termination


@configclass
class CurriculumPointCfg:
    step: int = MISSING
    value: float = MISSING


@configclass
class TerminationRuleScheduleCfg:
    rule_id: str = MISSING
    points: list[CurriculumPointCfg] = MISSING


@configclass
class TerminationCurriculumCfg:
    schedules: list[TerminationRuleScheduleCfg] = []


class TerminationThresholdCurriculum:
    def __init__(self, termination: Termination, cfg: TerminationCurriculumCfg | None = None) -> None:
        self._termination = termination
        self._cfg = cfg if cfg is not None else TerminationCurriculumCfg()
        self._rules = {rule.id: rule for rule in termination.failure_rules if hasattr(rule, "threshold")}
        self._base_values = {rule_id: float(rule.threshold) for rule_id, rule in self._rules.items()}
        self._schedules = self._validate_and_build_schedules(self._cfg)
        self._current_values = dict(self._base_values)

    @property
    def has_schedules(self) -> bool:
        return bool(self._schedules)

    def apply(self, step: int) -> dict[str, float]:
        if step < 0:
            raise ValueError("Curriculum step must be non-negative.")

        for rule_id, rule in self._rules.items():
            value = self._base_values[rule_id]
            points = self._schedules.get(rule_id)
            if points:
                value = self._interpolate_value(value, points, step)
            rule.threshold = value
            self._current_values[rule_id] = value

        return self.get_current_values()

    def get_current_values(self) -> dict[str, float]:
        return dict(self._current_values)

    def _validate_and_build_schedules(
        self,
        cfg: TerminationCurriculumCfg,
    ) -> dict[str, tuple[tuple[int, float], ...]]:
        schedules: dict[str, tuple[tuple[int, float], ...]] = {}
        for schedule_cfg in cfg.schedules:
            rule_id = str(schedule_cfg.rule_id)
            if rule_id not in self._rules:
                raise ValueError(f"Unknown termination rule id in curriculum: {rule_id}.")
            if rule_id in schedules:
                raise ValueError(f"Duplicate curriculum schedule configured for {rule_id}.")
            if not schedule_cfg.points:
                raise ValueError(f"Curriculum schedule for {rule_id} must contain at least one point.")

            normalized_points: list[tuple[int, float]] = []
            previous_step = -1
            for point in schedule_cfg.points:
                if point.step < 0:
                    raise ValueError(f"Curriculum schedule step for {rule_id} must be non-negative.")
                if point.step <= previous_step:
                    raise ValueError(f"Curriculum schedule steps for {rule_id} must be strictly increasing.")
                normalized_points.append((int(point.step), float(point.value)))
                previous_step = point.step

            schedules[rule_id] = tuple(normalized_points)
        return schedules

    @staticmethod
    def _interpolate_value(base_value: float, points: tuple[tuple[int, float], ...], step: int) -> float:
        first_step, first_value = points[0]
        if step <= first_step:
            if first_step == 0:
                return first_value
            ratio = step / first_step
            return base_value + (first_value - base_value) * ratio

        for (previous_step, previous_value), (next_step, next_value) in zip(points, points[1:]):
            if step <= next_step:
                ratio = (step - previous_step) / (next_step - previous_step)
                return previous_value + (next_value - previous_value) * ratio

        return points[-1][1]

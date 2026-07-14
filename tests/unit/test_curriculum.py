import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest


def _configclass(cls, **kwargs):
    annotations = getattr(cls, "__annotations__", {})
    for name in annotations:
        if not hasattr(cls, name):
            continue
        value = getattr(cls, name)
        if isinstance(value, list):
            setattr(cls, name, field(default_factory=lambda value=value: list(value)))
        elif isinstance(value, dict):
            setattr(cls, name, field(default_factory=lambda value=value: dict(value)))
    return dataclass(cls, **kwargs)


def _load_curriculum_module():
    isaaclab = types.ModuleType("isaaclab")
    utils_mod = types.ModuleType("isaaclab.utils")
    configclass_mod = types.ModuleType("isaaclab.utils.configclass")
    configclass_mod.configclass = _configclass
    isaaclab.utils = utils_mod

    previous_isaaclab = sys.modules.get("isaaclab")
    previous_utils = sys.modules.get("isaaclab.utils")
    previous_configclass = sys.modules.get("isaaclab.utils.configclass")

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.configclass"] = configclass_mod
    try:
        sys.modules.pop("ref2act.envs.motion_tracking.curriculum", None)
        return importlib.import_module("ref2act.envs.motion_tracking.curriculum")
    finally:
        if previous_isaaclab is None:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = previous_isaaclab
        if previous_utils is None:
            sys.modules.pop("isaaclab.utils", None)
        else:
            sys.modules["isaaclab.utils"] = previous_utils
        if previous_configclass is None:
            sys.modules.pop("isaaclab.utils.configclass", None)
        else:
            sys.modules["isaaclab.utils.configclass"] = previous_configclass


curriculum_mod = _load_curriculum_module()
CurriculumPointCfg = curriculum_mod.CurriculumPointCfg
TerminationCurriculumCfg = curriculum_mod.TerminationCurriculumCfg
TerminationRuleScheduleCfg = curriculum_mod.TerminationRuleScheduleCfg
TerminationThresholdCurriculum = curriculum_mod.TerminationThresholdCurriculum


class _DummyRule:
    def __init__(self, rule_id: str, threshold: float) -> None:
        self.id = rule_id
        self.threshold = threshold


class _DummyTermination:
    def __init__(self) -> None:
        self.failure_rules = [
            _DummyRule("anchor_position_failure", 0.25),
            _DummyRule("anchor_orientation_failure", 0.8),
            _DummyRule("end_effector_position_failure", 0.25),
        ]


def test_empty_curriculum_is_a_no_op() -> None:
    termination = _DummyTermination()
    curriculum = TerminationThresholdCurriculum(termination, TerminationCurriculumCfg())

    values = curriculum.apply(step=7)

    assert not curriculum.has_schedules
    assert values == {
        "anchor_position_failure": 0.25,
        "anchor_orientation_failure": 0.8,
        "end_effector_position_failure": 0.25,
    }


def test_rule_id_schedule_interpolates_and_only_updates_target_rule() -> None:
    termination = _DummyTermination()
    curriculum = TerminationThresholdCurriculum(
        termination,
        TerminationCurriculumCfg(
            schedules=[
                TerminationRuleScheduleCfg(
                    rule_id="anchor_position_failure",
                    points=[CurriculumPointCfg(step=10, value=0.05)],
                )
            ]
        ),
    )

    curriculum.apply(step=5)

    thresholds = {rule.id: rule.threshold for rule in termination.failure_rules}
    assert thresholds["anchor_position_failure"] == pytest.approx(0.15)
    assert thresholds["anchor_orientation_failure"] == pytest.approx(0.8)
    assert thresholds["end_effector_position_failure"] == pytest.approx(0.25)


def test_invalid_curriculum_configurations_raise_value_error() -> None:
    termination = _DummyTermination()

    with pytest.raises(ValueError, match="Unknown termination rule id"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationRuleScheduleCfg(
                        rule_id="missing_rule",
                        points=[CurriculumPointCfg(step=5, value=0.2)],
                    )
                ]
            ),
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationRuleScheduleCfg(
                        rule_id="anchor_position_failure",
                        points=[
                            CurriculumPointCfg(step=5, value=0.2),
                            CurriculumPointCfg(step=5, value=0.1),
                        ],
                    )
                ]
            ),
        )

    with pytest.raises(ValueError, match="Duplicate curriculum schedule"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationRuleScheduleCfg(
                        rule_id="anchor_position_failure",
                        points=[CurriculumPointCfg(step=5, value=0.2)],
                    ),
                    TerminationRuleScheduleCfg(
                        rule_id="anchor_position_failure",
                        points=[CurriculumPointCfg(step=10, value=0.1)],
                    ),
                ]
            ),
        )

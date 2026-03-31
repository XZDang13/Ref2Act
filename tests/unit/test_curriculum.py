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
    utils_mod.configclass = _configclass
    isaaclab.utils = utils_mod

    previous_isaaclab = sys.modules.get("isaaclab")
    previous_utils = sys.modules.get("isaaclab.utils")

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = utils_mod
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


curriculum_mod = _load_curriculum_module()
CurriculumPointCfg = curriculum_mod.CurriculumPointCfg
TerminationCurriculumCfg = curriculum_mod.TerminationCurriculumCfg
TerminationThresholdCurriculum = curriculum_mod.TerminationThresholdCurriculum
TerminationThresholdField = curriculum_mod.TerminationThresholdField
TerminationThresholdScheduleCfg = curriculum_mod.TerminationThresholdScheduleCfg


class _DummyTermination:
    def __init__(
        self,
        *,
        anchor_pos_error_threshold: float = 0.25,
        anchor_ori_error_threshold: float = 0.8,
        end_effector_pos_error_threshold: float = 0.25,
    ) -> None:
        self.anchor_pos_error_threshold = anchor_pos_error_threshold
        self.anchor_ori_error_threshold = anchor_ori_error_threshold
        self.end_effector_pos_error_threshold = end_effector_pos_error_threshold


def test_empty_curriculum_is_a_no_op() -> None:
    termination = _DummyTermination()
    curriculum = TerminationThresholdCurriculum(termination, TerminationCurriculumCfg())

    values = curriculum.apply(step=7)

    assert not curriculum.has_schedules
    assert values == {
        "anchor_pos_error_threshold": 0.25,
        "anchor_ori_error_threshold": 0.8,
        "end_effector_pos_error_threshold": 0.25,
    }
    assert termination.anchor_pos_error_threshold == pytest.approx(0.25)
    assert termination.anchor_ori_error_threshold == pytest.approx(0.8)
    assert termination.end_effector_pos_error_threshold == pytest.approx(0.25)


def test_single_segment_schedule_interpolates_from_static_threshold() -> None:
    termination = _DummyTermination(anchor_pos_error_threshold=0.25)
    curriculum = TerminationThresholdCurriculum(
        termination,
        TerminationCurriculumCfg(
            schedules=[
                TerminationThresholdScheduleCfg(
                    field=TerminationThresholdField.AnchorPosErrorThreshold,
                    points=[CurriculumPointCfg(step=10, value=0.05)],
                )
            ]
        ),
    )

    curriculum.apply(step=0)
    assert termination.anchor_pos_error_threshold == pytest.approx(0.25)

    curriculum.apply(step=5)
    assert termination.anchor_pos_error_threshold == pytest.approx(0.15)

    curriculum.apply(step=10)
    assert termination.anchor_pos_error_threshold == pytest.approx(0.05)

    curriculum.apply(step=20)
    assert termination.anchor_pos_error_threshold == pytest.approx(0.05)


def test_multi_segment_schedule_interpolates_between_points_and_holds_tail() -> None:
    termination = _DummyTermination(anchor_ori_error_threshold=0.8)
    curriculum = TerminationThresholdCurriculum(
        termination,
        TerminationCurriculumCfg(
            schedules=[
                TerminationThresholdScheduleCfg(
                    field=TerminationThresholdField.AnchorOriErrorThreshold,
                    points=[
                        CurriculumPointCfg(step=10, value=0.6),
                        CurriculumPointCfg(step=20, value=0.2),
                    ],
                )
            ]
        ),
    )

    curriculum.apply(step=15)
    assert termination.anchor_ori_error_threshold == pytest.approx(0.4)

    curriculum.apply(step=25)
    assert termination.anchor_ori_error_threshold == pytest.approx(0.2)


def test_first_point_after_step_zero_interpolates_from_static_threshold() -> None:
    termination = _DummyTermination(end_effector_pos_error_threshold=0.25)
    curriculum = TerminationThresholdCurriculum(
        termination,
        TerminationCurriculumCfg(
            schedules=[
                TerminationThresholdScheduleCfg(
                    field=TerminationThresholdField.EndEffectorPosErrorThreshold,
                    points=[CurriculumPointCfg(step=4, value=0.05)],
                )
            ]
        ),
    )

    curriculum.apply(step=2)

    assert termination.end_effector_pos_error_threshold == pytest.approx(0.15)


def test_invalid_curriculum_configurations_raise_value_error() -> None:
    termination = _DummyTermination()

    with pytest.raises(ValueError, match="strictly increasing"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationThresholdScheduleCfg(
                        field=TerminationThresholdField.AnchorPosErrorThreshold,
                        points=[
                            CurriculumPointCfg(step=5, value=0.2),
                            CurriculumPointCfg(step=5, value=0.1),
                        ],
                    )
                ]
            ),
        )

    with pytest.raises(ValueError, match="non-negative"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationThresholdScheduleCfg(
                        field=TerminationThresholdField.AnchorPosErrorThreshold,
                        points=[CurriculumPointCfg(step=-1, value=0.2)],
                    )
                ]
            ),
        )

    with pytest.raises(ValueError, match="Duplicate curriculum schedule"):
        TerminationThresholdCurriculum(
            termination,
            TerminationCurriculumCfg(
                schedules=[
                    TerminationThresholdScheduleCfg(
                        field=TerminationThresholdField.AnchorPosErrorThreshold,
                        points=[CurriculumPointCfg(step=5, value=0.2)],
                    ),
                    TerminationThresholdScheduleCfg(
                        field=TerminationThresholdField.AnchorPosErrorThreshold,
                        points=[CurriculumPointCfg(step=10, value=0.1)],
                    ),
                ]
            ),
        )

import importlib
import sys
import types

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1:
        vec = vec.expand(quat.shape[:-1] + (3,))
    elif vec.shape[:-1] != quat.shape[:-1]:
        vec = torch.broadcast_to(vec, quat.shape[:-1] + (3,))

    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_conj = quat.clone()
    quat_conj[..., 1:] = -quat_conj[..., 1:]
    return _quat_apply(quat_conj, vec)


def _load_modules():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
        "ref2act.envs.motion_tracking.termination": sys.modules.get("ref2act.envs.motion_tracking.termination"),
        "ref2act.envs.motion_tracking.tracking_quality": sys.modules.get(
            "ref2act.envs.motion_tracking.tracking_quality"
        ),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply_inverse = _quat_apply_inverse

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    isaaclab.assets = assets_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.termination", None)
        sys.modules.pop("ref2act.envs.motion_tracking.tracking_quality", None)
        termination_mod = importlib.import_module("ref2act.envs.motion_tracking.termination")
        tracking_quality_mod = importlib.import_module("ref2act.envs.motion_tracking.tracking_quality")
        return termination_mod, tracking_quality_mod
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        for attr_name, previous_attr in previous_attrs.items():
            if previous_attr is sentinel:
                if hasattr(isaaclab, attr_name):
                    delattr(isaaclab, attr_name)
            else:
                setattr(isaaclab, attr_name, previous_attr)


def _make_sampler(num_envs: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        current_times=torch.zeros(num_envs, dtype=torch.float32),
        get_current_durations=lambda: torch.ones(num_envs, dtype=torch.float32),
    )


def _make_context(termination_mod, z_errors: list[float]):
    num_envs = len(z_errors)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor([[[0.0, 0.0, error]] for error in z_errors], dtype=torch.float32),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * num_envs, dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((num_envs, 1, 3), dtype=torch.float32),
        body_quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * num_envs, dtype=torch.float32),
        body_pos_relative=torch.zeros((num_envs, 1, 3), dtype=torch.float32),
    )
    return termination_mod.TerminationContext(
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        max_episode_length=torch.full((num_envs,), 100, dtype=torch.long),
        robot=robot,
        reference_motion=reference_motion,
        sampler=_make_sampler(num_envs),
    )


def test_quality_gate_classifies_ok_soft_and_recovery_without_hard_termination() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True),
        termination,
        num_envs=3,
        device=torch.device("cpu"),
    )

    result = gate.evaluate(_make_context(termination_mod, [0.5, 1.2, 2.0]))

    assert torch.equal(
        result.state,
        torch.tensor(
            [
                quality_mod.TrackingQuality.OK,
                quality_mod.TrackingQuality.SOFT_VIOLATION,
                quality_mod.TrackingQuality.RECOVERY_NEEDED,
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(result.hard_tracking_failure_mask, torch.tensor([False, False, False]))
    assert torch.equal(result.recovery_timeout_mask, torch.tensor([False, False, False]))
    assert torch.equal(result.record_failure_mask, torch.tensor([False, False, True]))


def test_quality_gate_recovery_hysteresis_requires_minimum_steps_before_exit() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(
            enabled=True,
            recovery_enter_threshold=1.8,
            recovery_exit_threshold=1.2,
            min_recovery_steps=2,
        ),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )

    first = gate.evaluate(_make_context(termination_mod, [2.0]))
    second = gate.evaluate(_make_context(termination_mod, [0.5]))
    third = gate.evaluate(_make_context(termination_mod, [0.5]))

    assert int(first.state[0]) == int(quality_mod.TrackingQuality.RECOVERY_NEEDED)
    assert int(second.state[0]) == int(quality_mod.TrackingQuality.RECOVERY_NEEDED)
    assert int(third.state[0]) == int(quality_mod.TrackingQuality.OK)


def test_quality_gate_hard_threshold_is_optional() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )

    no_hard_gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True, hard_tracking_threshold=None),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )
    hard_gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True, hard_tracking_threshold=4.0),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )

    no_hard = no_hard_gate.evaluate(_make_context(termination_mod, [10.0]))
    hard = hard_gate.evaluate(_make_context(termination_mod, [10.0]))

    assert torch.equal(no_hard.hard_tracking_failure_mask, torch.tensor([False]))
    assert torch.equal(hard.hard_tracking_failure_mask, torch.tensor([True]))
    assert torch.equal(hard.record_failure_mask, torch.tensor([True]))


def test_quality_gate_recovery_timeout_counts_current_recovery_step_and_resets() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(
            enabled=True,
            recovery_enter_threshold=1.8,
            max_recovery_steps=2,
        ),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )

    first = gate.evaluate(_make_context(termination_mod, [2.0]))
    second = gate.evaluate(_make_context(termination_mod, [2.0]))
    gate.reset(torch.tensor([0], dtype=torch.long))
    after_reset = gate.evaluate(_make_context(termination_mod, [2.0]))

    assert torch.equal(first.recovery_timeout_mask, torch.tensor([False]))
    assert torch.equal(second.recovery_timeout_mask, torch.tensor([True]))
    assert torch.equal(second.record_failure_mask, torch.tensor([True]))
    assert torch.equal(after_reset.recovery_timeout_mask, torch.tensor([False]))


def test_quality_gate_uses_current_rule_thresholds_for_scores() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )

    initial = gate.evaluate(_make_context(termination_mod, [1.0]))
    termination.get_failure_rule("anchor_position_failure").threshold = 0.5
    updated = gate.evaluate(_make_context(termination_mod, [1.0]))

    assert torch.allclose(initial.score, torch.tensor([1.0]))
    assert torch.allclose(updated.score, torch.tensor([2.0]))
    assert torch.allclose(initial.previous_score, torch.tensor([1.0]))
    assert torch.allclose(updated.previous_score, torch.tensor([1.0]))


def test_quality_gate_previous_score_resets_to_current_score_after_reset() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=1.0,
                    height_only=True,
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )

    first = gate.evaluate(_make_context(termination_mod, [2.0]))
    second = gate.evaluate(_make_context(termination_mod, [1.0]))
    gate.reset(torch.tensor([0], dtype=torch.long))
    after_reset = gate.evaluate(_make_context(termination_mod, [3.0]))

    assert torch.allclose(first.previous_score, torch.tensor([2.0]))
    assert torch.allclose(second.previous_score, torch.tensor([2.0]))
    assert torch.allclose(after_reset.previous_score, torch.tensor([3.0]))


def test_quality_gate_respects_end_effector_all_reduction_for_continuous_score() -> None:
    termination_mod, quality_mod = _load_modules()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.EndEffectorPositionFailureRuleCfg(
                    end_effector_body_indices=(0, 1),
                    threshold=1.0,
                    height_only=True,
                    reduction="all",
                ),
            ),
        )
    )
    gate = quality_mod.TrackingQualityGate(
        quality_mod.TrackingQualityGateCfg(enabled=True),
        termination,
        num_envs=1,
        device=torch.device("cpu"),
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]]], dtype=torch.float32),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((1, 2, 3), dtype=torch.float32),
        body_quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_pos_relative=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    context = termination_mod.TerminationContext(
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        max_episode_length=torch.full((1,), 100, dtype=torch.long),
        robot=robot,
        reference_motion=reference_motion,
        sampler=_make_sampler(1),
    )

    result = gate.evaluate(context)

    assert torch.allclose(result.score, torch.tensor([0.5]))

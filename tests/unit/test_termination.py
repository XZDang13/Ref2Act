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


def _load_termination_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
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
        return importlib.import_module("ref2act.envs.motion_tracking.termination")
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


def _make_context_inputs(num_envs: int = 1):
    return torch.zeros(num_envs, dtype=torch.long), torch.full((num_envs,), 100, dtype=torch.long), _make_sampler(num_envs)


def test_anchor_orientation_failure_rule_uses_body_quat_world() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorOrientationFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=0.1,
                ),
            ),
        )
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            body_link_quat_w=torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((1, 1, 3), dtype=torch.float32),
        body_quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        body_pos_relative=torch.zeros((1, 1, 3), dtype=torch.float32),
    )

    episode_length_buf, max_episode_length, sampler = _make_context_inputs()
    terminate, time_out = termination.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )

    assert torch.equal(terminate, torch.tensor([False]))
    assert torch.equal(time_out, torch.tensor([False]))


def test_probabilistic_threshold_policy_probability_is_monotonic_before_ramp_limit() -> None:
    termination_mod = _load_termination_module()
    policy = termination_mod.ThresholdPolicy(
        termination_mod.ThresholdPolicyCfg(probabilistic=True, ramp_multiplier=2.0, sigmoid_steepness=8.0)
    )

    probabilities = policy._error_to_termination_probability(
        torch.tensor([0.3, 0.375, 0.45], dtype=torch.float32),
        0.25,
    )

    assert torch.all(probabilities > 0.0)
    assert torch.all(probabilities < 1.0)
    assert torch.all(probabilities[1:] > probabilities[:-1])


def test_end_effector_position_rule_supports_full_3d_and_height_only_modes() -> None:
    termination_mod = _load_termination_module()
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.2, 0.0]]], dtype=torch.float32),
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

    full_3d = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.EndEffectorPositionFailureRuleCfg(
                    end_effector_body_indices=(1,),
                    threshold=0.25,
                    height_only=False,
                ),
            ),
        )
    )
    height_only = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.EndEffectorPositionFailureRuleCfg(
                    end_effector_body_indices=(1,),
                    threshold=0.25,
                    height_only=True,
                ),
            ),
        )
    )

    episode_length_buf, max_episode_length, sampler = _make_context_inputs()
    full_terminate, _ = full_3d.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )
    height_terminate, _ = height_only.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )

    assert torch.equal(full_terminate, torch.tensor([True]))
    assert torch.equal(height_terminate, torch.tensor([False]))


def test_runtime_threshold_update_takes_effect_without_recreating_termination() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=0.25,
                    height_only=True,
                ),
            ),
        )
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor([[[0.0, 0.0, 0.3]]], dtype=torch.float32),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((1, 1, 3), dtype=torch.float32),
        body_quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        body_pos_relative=torch.zeros((1, 1, 3), dtype=torch.float32),
    )

    episode_length_buf, max_episode_length, sampler = _make_context_inputs()
    initial_terminated, _ = termination.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )
    termination.get_failure_rule("anchor_position_failure").threshold = 0.35
    updated_terminated, _ = termination.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )

    assert torch.equal(initial_terminated, torch.tensor([True]))
    assert torch.equal(updated_terminated, torch.tensor([False]))


def test_failure_rules_expose_continuous_errors() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=0.25,
                    height_only=True,
                ),
                termination_mod.EndEffectorPositionFailureRuleCfg(
                    end_effector_body_indices=(1, 2),
                    threshold=0.25,
                    height_only=True,
                ),
            ),
        )
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.3], [0.0, 0.0, 0.1], [0.0, 0.0, 0.4]]],
                dtype=torch.float32,
            ),
            body_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((1, 3, 3), dtype=torch.float32),
        body_quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_pos_relative=torch.zeros((1, 3, 3), dtype=torch.float32),
    )
    episode_length_buf, max_episode_length, sampler = _make_context_inputs()
    context = termination.build_context(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )

    anchor_error = termination.get_failure_rule("anchor_position_failure").error(context)
    ee_error = termination.get_failure_rule("end_effector_position_failure").error(context)

    assert torch.allclose(anchor_error, torch.tensor([0.3]))
    assert torch.allclose(ee_error, torch.tensor([[0.1, 0.4]]))


def test_probabilistic_get_dones_samples_each_branch_independently(monkeypatch) -> None:
    termination_mod = _load_termination_module()
    policy = termination_mod.ThresholdPolicyCfg(probabilistic=True)
    termination = termination_mod.Termination(
        termination_mod.TerminationSpec(
            timeout_rules=(),
            failure_rules=(
                termination_mod.AnchorPositionFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=0.25,
                    height_only=True,
                    policy=policy,
                ),
                termination_mod.AnchorOrientationFailureRuleCfg(
                    anchor_body_index=0,
                    threshold=0.75,
                    policy=policy,
                ),
                termination_mod.EndEffectorPositionFailureRuleCfg(
                    end_effector_body_indices=(1, 2),
                    threshold=0.25,
                    height_only=True,
                    policy=policy,
                ),
            ),
        )
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.375], [0.0, 0.0, 0.375], [0.0, 0.0, 0.375]]],
                dtype=torch.float32,
            ),
            body_quat_w=torch.tensor(
                [[[0.70710677, 0.70710677, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_positions=torch.zeros((1, 3, 3), dtype=torch.float32),
        body_quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_pos_relative=torch.zeros((1, 3, 3), dtype=torch.float32),
    )

    rand_shapes: list[tuple[int, ...]] = []
    draws = [
        torch.tensor([0.6], dtype=torch.float32),
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([[0.6, 0.4]], dtype=torch.float32),
    ]

    def fake_rand_like(tensor: torch.Tensor) -> torch.Tensor:
        draw = draws[len(rand_shapes)].to(device=tensor.device, dtype=tensor.dtype)
        rand_shapes.append(tuple(tensor.shape))
        assert draw.shape == tensor.shape
        return draw

    monkeypatch.setattr(termination_mod.torch, "rand_like", fake_rand_like)

    episode_length_buf, max_episode_length, sampler = _make_context_inputs()
    terminate, time_out = termination.get_dones(
        episode_length_buf,
        max_episode_length,
        robot,
        reference_motion,
        sampler,
    )

    assert torch.equal(terminate, torch.tensor([True]))
    assert torch.equal(time_out, torch.tensor([False]))
    assert rand_shapes == [(1,), (1,), (1, 2)]
    assert torch.equal(termination.terminated_env_ids, torch.tensor([0]))

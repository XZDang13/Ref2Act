import importlib
import sys
import types

import torch


def _load_action_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    sys.modules["isaaclab.assets"] = assets_mod
    isaaclab.assets = assets_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.action", None)
        sys.modules.pop("ref2act.bridges.mujoco.action", None)
        return (
            importlib.import_module("ref2act.envs.motion_tracking.action"),
            importlib.import_module("ref2act.bridges.mujoco.action"),
        )
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


def _make_robot(num_envs: int = 2, num_joints: int = 3):
    joint_shape = (num_envs, num_joints)
    joint_limits = torch.stack(
        (
            -torch.ones(joint_shape, dtype=torch.float32),
            torch.ones(joint_shape, dtype=torch.float32),
        ),
        dim=-1,
    )
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            device=torch.device("cpu"),
            joint_pos=torch.zeros(joint_shape, dtype=torch.float32),
            default_joint_pos=torch.zeros(joint_shape, dtype=torch.float32),
            joint_pos_limits=joint_limits,
            joint_effort_limits=torch.full(joint_shape, 4.0, dtype=torch.float32),
            default_joint_stiffness=torch.full(joint_shape, 2.0, dtype=torch.float32),
        )
    )


def test_action_processor_resets_noise_and_target_from_structured_spec() -> None:
    action_mod, _ = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(
        robot,
        action_mod.ActionSpec(mode="median", buffer_length=3, noise_scale=0.2),
    )

    torch.manual_seed(0)
    processor.set_random_offset_noise([0, 1])
    processor.pre_process_action(torch.ones((2, 3), dtype=torch.float32))

    assert not torch.allclose(processor.target_joint_position[0], torch.zeros(3))

    processor.reset_action_buffer([0])

    assert torch.allclose(processor.applied_action[0], torch.zeros(3))
    assert torch.allclose(processor.previous_applied_action[0], torch.zeros(3))
    assert torch.allclose(processor.offset_noise[0], torch.zeros(3))
    assert torch.allclose(processor.target_joint_position[0], torch.zeros(3))
    assert not torch.allclose(processor.target_joint_position[1], torch.zeros(3))


def test_residual_mode_uses_reference_positions_and_clamps_targets() -> None:
    action_mod, _ = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(
        robot,
        action_mod.ActionSpec(mode="residual", buffer_length=2, noise_scale=0.0),
    )
    processor.set_reference_joint_position(
        torch.tensor(
            [
                [0.2, -0.2, 0.0],
                [0.4, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
    )

    processor.pre_process_action(
        torch.tensor(
            [
                [1.0, -1.0, 0.5],
                [-2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    assert torch.allclose(
        processor.target_joint_position,
        torch.tensor(
            [
                [0.7, -0.7, 0.25],
                [-0.6, 1.0, 0.4],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        processor.scale_action(torch.zeros((2, 3), dtype=torch.float32)),
        processor.reference_joint_position,
    )


def test_current_residual_mode_uses_current_joint_positions_and_clamps_targets() -> None:
    action_mod, _ = _load_action_module()
    robot = _make_robot()
    robot.data.joint_pos[:] = torch.tensor(
        [
            [0.2, -0.2, 0.0],
            [0.4, 0.4, 0.4],
        ],
        dtype=torch.float32,
    )
    processor = action_mod.ActionProcessor(
        robot,
        action_mod.ActionSpec(mode="current_residual", buffer_length=2, noise_scale=0.0),
    )
    processor.set_reference_joint_position(torch.full((2, 3), 0.9, dtype=torch.float32))

    processor.pre_process_action(
        torch.tensor(
            [
                [1.0, -1.0, 0.5],
                [-2.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )

    assert torch.allclose(
        processor.target_joint_position,
        torch.tensor(
            [
                [0.7, -0.7, 0.25],
                [-0.6, 1.0, 0.4],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        processor.scale_action(torch.zeros((2, 3), dtype=torch.float32)),
        robot.data.joint_pos,
    )


def test_env_and_mujoco_action_modes_share_the_same_target_semantics() -> None:
    action_mod, bridge_action_mod = _load_action_module()

    raw_action = torch.tensor([[0.8, -0.4, 0.2]], dtype=torch.float32)
    reference_joint_pos = torch.tensor([[0.25, -0.1, 0.5]], dtype=torch.float32)
    current_joint_pos = torch.tensor([[0.2, -0.2, 0.0]], dtype=torch.float32)

    for mode in ("median", "offset", "residual", "current_residual"):
        robot = _make_robot(num_envs=1, num_joints=3)
        robot.data.joint_pos[:] = current_joint_pos
        processor = action_mod.ActionProcessor(
            robot,
            action_mod.ActionSpec(mode=mode, buffer_length=1, noise_scale=0.0),
        )
        processor.set_reference_joint_position(reference_joint_pos)

        env_target = processor.scale_action(raw_action)

        builder = bridge_action_mod.IsaacLabMujocoAction()
        context = bridge_action_mod.MujocoActionContext(
            raw_action=raw_action.squeeze(0),
            action_mode=mode,
            action_scale=processor.scale,
            action_offset=processor.offset.squeeze(0) if processor.offset.ndim == 2 else processor.offset,
            joint_pos_limits_lower=processor.joint_low_limit,
            joint_pos_limits_upper=processor.joint_up_limit,
            current_joint_pos_loader=lambda current_joint_pos=current_joint_pos: current_joint_pos.squeeze(0),
            reference_joint_pos_loader=lambda reference_joint_pos=reference_joint_pos: reference_joint_pos.squeeze(0),
        )
        mujoco_target = builder.process_action(types.SimpleNamespace(), context).target_joint_pos.unsqueeze(0)

        assert torch.allclose(env_target, mujoco_target)

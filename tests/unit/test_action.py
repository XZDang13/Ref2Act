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
        return importlib.import_module("ref2act.envs.motion_tracking.action")
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


def test_reset_action_buffer_clears_noise_and_resets_target() -> None:
    action_mod = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(robot, action_buffer_length=3, noise_scale=0.2)
    processor.set_median_scale_offset(robot)

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


def test_random_offset_noise_updates_target_for_selected_envs() -> None:
    action_mod = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(robot, action_buffer_length=2, noise_scale=0.2)
    processor.set_median_scale_offset(robot)

    assert torch.allclose(processor.target_joint_position, torch.zeros((2, 3)))

    torch.manual_seed(1)
    processor.set_random_offset_noise(torch.tensor([1]))

    assert torch.allclose(processor.target_joint_position[0], torch.zeros(3))
    assert torch.allclose(processor.target_joint_position[1], processor.offset_noise[1])
    assert not torch.allclose(processor.target_joint_position[1], torch.zeros(3))


def test_residual_action_uses_reference_positions_and_clamps_targets() -> None:
    action_mod = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(robot, action_buffer_length=2, noise_scale=0.0, action_mod="residual")
    processor.set_residual_scale_offset(robot)
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


def test_residual_reference_updates_and_reset_restore_reference_baseline() -> None:
    action_mod = _load_action_module()
    robot = _make_robot()
    processor = action_mod.ActionProcessor(robot, action_buffer_length=2, noise_scale=0.2, action_mod="residual")
    processor.set_residual_scale_offset(robot)
    processor.set_reference_joint_position(
        torch.tensor(
            [
                [0.3, 0.0, -0.1],
                [0.0, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )
    )
    processor.pre_process_action(torch.ones((2, 3), dtype=torch.float32))

    processor.set_reference_joint_position(
        torch.tensor([[0.5, -0.5, 0.25]], dtype=torch.float32),
        env_ids=[0],
    )
    assert torch.allclose(processor.target_joint_position[0], torch.tensor([1.0, 0.0, 0.75], dtype=torch.float32))

    torch.manual_seed(2)
    processor.set_random_offset_noise([1])
    expected_target_with_noise = processor.reference_joint_position[1] + processor.scale + processor.offset_noise[1]
    assert torch.allclose(processor.target_joint_position[1], expected_target_with_noise)

    processor.reset_action_buffer([1])
    assert torch.allclose(processor.offset_noise[1], torch.zeros(3))
    assert torch.allclose(processor.applied_action[1], torch.zeros(3))
    assert torch.allclose(processor.previous_applied_action[1], torch.zeros(3))
    assert torch.allclose(processor.target_joint_position[1], processor.reference_joint_position[1])


def test_current_residual_action_uses_current_joint_positions_and_clamps_targets() -> None:
    action_mod = _load_action_module()
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
        action_buffer_length=2,
        noise_scale=0.0,
        action_mod="CurrentResidual",
    )
    processor.set_current_residual_scale_offset(robot)
    processor.set_reference_joint_position(
        torch.tensor(
            [
                [0.9, 0.9, 0.9],
                [-0.9, -0.9, -0.9],
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
        robot.data.joint_pos,
    )


def test_current_residual_reference_updates_and_reset_restore_current_joint_baseline() -> None:
    action_mod = _load_action_module()
    robot = _make_robot()
    robot.data.joint_pos[:] = torch.tensor(
        [
            [0.3, 0.0, -0.1],
            [0.0, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    processor = action_mod.ActionProcessor(
        robot,
        action_buffer_length=2,
        noise_scale=0.2,
        action_mod="current_residual",
    )
    processor.set_current_residual_scale_offset(robot)
    processor.pre_process_action(torch.ones((2, 3), dtype=torch.float32))

    robot.data.joint_pos[0] = torch.tensor([0.5, -0.5, 0.25], dtype=torch.float32)
    processor.set_reference_joint_position(
        torch.tensor([[0.9, 0.9, 0.9]], dtype=torch.float32),
        env_ids=[0],
    )
    assert torch.allclose(processor.target_joint_position[0], torch.tensor([1.0, 0.0, 0.75], dtype=torch.float32))

    torch.manual_seed(2)
    processor.set_random_offset_noise([1])
    expected_target_with_noise = robot.data.joint_pos[1] + processor.scale + processor.offset_noise[1]
    assert torch.allclose(processor.target_joint_position[1], expected_target_with_noise)

    robot.data.joint_pos[1] = torch.tensor([-0.2, 0.1, 0.0], dtype=torch.float32)
    processor.reset_action_buffer([1])
    assert torch.allclose(processor.offset_noise[1], torch.zeros(3))
    assert torch.allclose(processor.applied_action[1], torch.zeros(3))
    assert torch.allclose(processor.previous_applied_action[1], torch.zeros(3))
    assert torch.allclose(processor.target_joint_position[1], robot.data.joint_pos[1])

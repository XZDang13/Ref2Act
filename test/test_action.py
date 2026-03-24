from pathlib import Path
import importlib
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        sys.modules.pop("Ref2Act.action", None)
        return importlib.import_module("Ref2Act.action")
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

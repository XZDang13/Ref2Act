from pathlib import Path
import importlib
import sys
import types

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        "Ref2Act.sampler": sys.modules.get("Ref2Act.sampler"),
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

    sampler_mod = types.ModuleType("Ref2Act.sampler")
    sampler_mod.Sampler = type("Sampler", (), {})
    sampler_mod.ReferenceMotions = type("ReferenceMotions", (), {})

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["Ref2Act.sampler"] = sampler_mod
    isaaclab.assets = assets_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("Ref2Act.termination", None)
        return importlib.import_module("Ref2Act.termination")
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


def test_anchor_orientation_termination_uses_body_quat_world() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        anchor_body_index=0,
        end_effector_body_indices=[0],
        anchor_pos_error_threshold=1.0,
        anchor_ori_error_threshold=0.1,
        end_effector_pos_error_threshold=1.0,
        height_only=True,
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            body_link_quat_w=torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
    )

    terminated = termination.anchor_ori_error_terminate(robot, reference_motion)

    assert torch.equal(terminated, torch.tensor([False]))


def test_end_effector_termination_keeps_full_3d_error_when_height_only() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        anchor_body_index=0,
        end_effector_body_indices=[1],
        anchor_pos_error_threshold=1.0,
        anchor_ori_error_threshold=1.0,
        end_effector_pos_error_threshold=0.25,
        height_only=True,
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_pos_relative=torch.zeros((1, 2, 3), dtype=torch.float32),
    )

    terminated = termination.end_effector_pos_error_terminate(robot, reference_motion)

    assert torch.equal(terminated, torch.tensor([True]))


def test_end_effector_termination_can_use_height_only_mode() -> None:
    termination_mod = _load_termination_module()
    termination = termination_mod.Termination(
        anchor_body_index=0,
        end_effector_body_indices=[1],
        anchor_pos_error_threshold=1.0,
        anchor_ori_error_threshold=1.0,
        end_effector_pos_error_threshold=0.25,
        height_only=True,
        end_effector_height_only=True,
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.3, 0.0, 0.1]]],
                dtype=torch.float32,
            ),
        )
    )
    reference_motion = types.SimpleNamespace(
        body_pos_relative=torch.zeros((1, 2, 3), dtype=torch.float32),
    )

    terminated = termination.end_effector_pos_error_terminate(robot, reference_motion)

    assert torch.equal(terminated, torch.tensor([False]))

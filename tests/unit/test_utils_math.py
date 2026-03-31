import importlib
import math
import sys
import types

import torch

from ref2act.common.utils import slerp


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return _quat_conjugate(q) / torch.sum(q * q, dim=-1, keepdim=True)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _yaw_quat(q: torch.Tensor) -> torch.Tensor:
    yaw = torch.atan2(
        2.0 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
        1.0 - 2.0 * (q[..., 2] * q[..., 2] + q[..., 3] * q[..., 3]),
    )
    zeros = torch.zeros_like(yaw)
    return _quat_from_euler_xyz(zeros, zeros, yaw)


def _subtract_frame_transforms(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos = _quat_apply(_quat_inv(anchor_quat), key_pos - anchor_pos)
    rel_quat = _quat_mul(_quat_inv(anchor_quat), key_quat)
    return rel_pos, rel_quat


def _load_math_module():
    sentinel = object()
    previous_root = sys.modules.get("isaaclab", sentinel)
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
    }

    isaaclab_mod = previous_root if previous_root is not sentinel else types.ModuleType("isaaclab")
    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.subtract_frame_transforms = _subtract_frame_transforms
    math_mod.quat_apply = _quat_apply
    math_mod.quat_mul = _quat_mul
    math_mod.quat_conjugate = _quat_conjugate
    math_mod.yaw_quat = _yaw_quat
    math_mod.quat_inv = _quat_inv

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod

    sys.modules["isaaclab"] = isaaclab_mod
    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    isaaclab_mod.assets = assets_mod
    isaaclab_mod.utils = utils_mod

    try:
        sys.modules.pop("ref2act.common.math", None)
        return importlib.import_module("ref2act.common.math")
    finally:
        if previous_root is sentinel:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = previous_root

        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def test_slerp_is_stable_for_identical_quaternions() -> None:
    q0 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    blend = torch.tensor([0.0, 0.5], dtype=torch.float32)

    result = slerp(q0, q1=q0.clone(), blend=blend)

    assert torch.isfinite(result).all()
    assert torch.allclose(result, q0)


def test_get_relative_reference_motion_pose_uses_reference_anchor_orientation() -> None:
    math_mod = _load_math_module()

    robot_anchor_quat = _quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
    )
    reference_anchor_quat = _quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([-0.25], dtype=torch.float32),
    )
    reference_key_body_pos = torch.tensor([[[1.0, 0.0, 0.2]]], dtype=torch.float32)
    reference_key_body_quat = _quat_from_euler_xyz(
        torch.tensor([[0.0]], dtype=torch.float32),
        torch.tensor([[0.0]], dtype=torch.float32),
        torch.tensor([[0.1]], dtype=torch.float32),
    )
    robot_anchor_pos = torch.tensor([[0.5, -0.5, 1.0]], dtype=torch.float32)
    reference_anchor_pos = torch.tensor([[1.5, 0.25, 0.75]], dtype=torch.float32)

    relative_pos, relative_quat = math_mod.get_relative_reference_motion_pose(
        robot_anchor_pos,
        robot_anchor_quat,
        reference_anchor_pos,
        reference_anchor_quat,
        reference_key_body_pos,
        reference_key_body_quat,
    )

    delta_ori = _yaw_quat(_quat_mul(robot_anchor_quat[:, None, :], _quat_inv(reference_anchor_quat[:, None, :])))
    expected_pos = robot_anchor_pos[:, None, :].clone()
    expected_pos[..., 2] = reference_anchor_pos[:, None, 2]
    expected_pos = expected_pos + _quat_apply(delta_ori, reference_key_body_pos - reference_anchor_pos[:, None, :])
    expected_quat = _quat_mul(delta_ori, reference_key_body_quat)

    assert torch.allclose(relative_pos, expected_pos)
    assert torch.allclose(relative_quat, expected_quat)

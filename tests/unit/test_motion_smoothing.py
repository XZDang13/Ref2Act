import numpy as np
import torch

from ref2act.motion.processing.resample import _resample_motion_log
from ref2act.motion.smoothing import smooth_motion_trajectory


def _yaw_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((yaw.numel(), 4), dtype=torch.float32)
    result[:, 2] = torch.sin(yaw / 2.0)
    result[:, 3] = torch.cos(yaw / 2.0)
    return result


def test_smoothing_preserves_xyzw_unit_quaternions() -> None:
    frames = 31
    root_pos = torch.randn(frames, 3)
    root_rot = _yaw_xyzw(torch.linspace(-0.5, 0.5, frames))
    joint_pos = torch.randn(frames, 4)
    pos, quat, joints = smooth_motion_trajectory(root_pos, root_rot, joint_pos, fps=30.0)
    assert pos.shape == root_pos.shape
    assert quat.shape == root_rot.shape
    assert joints.shape == joint_pos.shape
    assert torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones(frames), atol=1.0e-5)


def test_resample_uses_current_retargeter_quaternion_field() -> None:
    quat = np.zeros((3, 1, 4), dtype=np.float32)
    quat[..., 3] = 1.0
    log = {
        "fps": 30.0,
        "joint_names": ["joint"],
        "body_names": ["pelvis"],
        "joint_pos": np.arange(3, dtype=np.float32)[:, None],
        "body_pos_w": np.zeros((3, 1, 3), dtype=np.float32),
        "body_quat_xyzw": quat,
    }
    output, feet = _resample_motion_log(
        log,
        np.zeros((3, 2), dtype=np.float32),
        source_fps=30,
        target_fps=60,
    )
    assert output["body_quat_xyzw"].shape == (6, 1, 4)
    assert output["body_lin_vel_w"].shape == (6, 1, 3)
    assert output["body_ang_vel_w"].shape == (6, 1, 3)
    assert feet.shape == (6, 2)

from pathlib import Path
import pickle
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ref2Act.convert_gmr_data import GMRMotionData, _resample_motion_log, build_parser
from Ref2Act.motion_smoothing import DEFAULT_SMOOTHING_PROFILE, smooth_motion_trajectory


def _yaw_quaternions(yaw_angles: torch.Tensor) -> torch.Tensor:
    half_angles = yaw_angles / 2.0
    quaternions = torch.zeros((yaw_angles.shape[0], 4), dtype=torch.float32)
    quaternions[:, 0] = torch.cos(half_angles)
    quaternions[:, 3] = torch.sin(half_angles)
    return quaternions


def _wxyz_to_xyzw(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternions[:, [1, 2, 3, 0]]


def _write_motion_pickle(path: Path, *, fps: float, root_pos: torch.Tensor, root_rot_wxyz: torch.Tensor, joint_pos: torch.Tensor) -> None:
    payload = {
        "fps": float(fps),
        "root_pos": root_pos.cpu().numpy(),
        "root_rot": _wxyz_to_xyzw(root_rot_wxyz).cpu().numpy(),
        "dof_pos": joint_pos.cpu().numpy(),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _mean_abs_second_difference(signal: torch.Tensor) -> float:
    second_difference = signal[2:] - 2.0 * signal[1:-1] + signal[:-2]
    return float(torch.mean(torch.abs(second_difference)).item())


def test_smoothing_preserves_shapes_dtype_and_unit_quaternions() -> None:
    num_frames = 21
    t = torch.linspace(0.0, 1.0, num_frames, dtype=torch.float32)
    root_pos = torch.stack((t, t.square(), torch.sin(t)), dim=-1)
    root_rot = _yaw_quaternions(0.25 * t)
    joint_pos = torch.stack((torch.sin(2.0 * t), torch.cos(3.0 * t)), dim=-1)

    smoothed_root_pos, smoothed_root_rot, smoothed_joint_pos = smooth_motion_trajectory(
        root_pos,
        root_rot,
        joint_pos,
        fps=30.0,
        profile="light",
    )

    assert smoothed_root_pos.shape == root_pos.shape
    assert smoothed_root_rot.shape == root_rot.shape
    assert smoothed_joint_pos.shape == joint_pos.shape
    assert smoothed_root_pos.dtype == root_pos.dtype
    assert smoothed_root_rot.dtype == root_rot.dtype
    assert smoothed_joint_pos.dtype == joint_pos.dtype
    assert torch.isfinite(smoothed_root_pos).all()
    assert torch.isfinite(smoothed_root_rot).all()
    assert torch.isfinite(smoothed_joint_pos).all()
    assert torch.allclose(torch.linalg.norm(smoothed_root_rot, dim=-1), torch.ones(num_frames), atol=1.0e-5)


def test_light_smoothing_reduces_pose_jitter() -> None:
    num_frames = 21
    t = torch.linspace(0.0, 1.0, num_frames, dtype=torch.float32)
    alternating = torch.where(torch.arange(num_frames) % 2 == 0, 1.0, -1.0).to(torch.float32)

    root_pos = torch.stack(
        (
            t + 0.04 * alternating,
            0.03 * alternating,
            torch.zeros_like(t),
        ),
        dim=-1,
    )
    root_rot = _yaw_quaternions(0.2 * t + 0.03 * alternating)
    joint_pos = torch.stack(
        (
            torch.sin(2.0 * t) + 0.05 * alternating,
            torch.cos(2.5 * t) - 0.05 * alternating,
        ),
        dim=-1,
    )

    smoothed_root_pos, _, smoothed_joint_pos = smooth_motion_trajectory(
        root_pos,
        root_rot,
        joint_pos,
        fps=30.0,
        profile="light",
    )

    assert _mean_abs_second_difference(smoothed_root_pos[:, 0]) < _mean_abs_second_difference(root_pos[:, 0])
    assert _mean_abs_second_difference(smoothed_joint_pos[:, 0]) < _mean_abs_second_difference(joint_pos[:, 0])


def test_smoothing_skips_very_short_clips() -> None:
    root_pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32)
    root_rot = _yaw_quaternions(torch.tensor([0.0, 0.1], dtype=torch.float32))
    joint_pos = torch.tensor([[0.0, 0.1], [0.2, 0.3]], dtype=torch.float32)

    smoothed_root_pos, smoothed_root_rot, smoothed_joint_pos = smooth_motion_trajectory(
        root_pos,
        root_rot,
        joint_pos,
        fps=30.0,
        profile="light",
    )

    assert torch.allclose(smoothed_root_pos, root_pos)
    assert torch.allclose(smoothed_root_rot, root_rot)
    assert torch.allclose(smoothed_joint_pos, joint_pos)


def test_parser_includes_motion_smoothing_flags() -> None:
    parser = build_parser()

    args = parser.parse_args(["--input_file", "motion.pkl"])
    assert args.smooth_motion is False
    assert args.smoothing_profile == DEFAULT_SMOOTHING_PROFILE
    assert args.target_fps is None

    args = parser.parse_args(
        [
            "--input_file",
            "motion.pkl",
            "--smooth-motion",
            "--smoothing-profile",
            "strong",
            "--target-fps",
            "100",
        ]
    )
    assert args.smooth_motion is True
    assert args.smoothing_profile == "strong"
    assert args.target_fps == 100


def test_resample_motion_log_updates_frequency_and_recomputes_velocities() -> None:
    joint_pos = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32).numpy()
    body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    ).numpy()
    body_quat_w = torch.zeros((3, 1, 4), dtype=torch.float32).numpy()
    body_quat_w[..., 0] = 1.0
    foot_heights = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=torch.float32,
    ).numpy()
    log = {
        "fps": 30.0,
        "joint_names": ["joint_0"],
        "body_names": ["body_0"],
        "joint_pos": joint_pos,
        "joint_vel": torch.zeros_like(torch.as_tensor(joint_pos)).numpy(),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": torch.zeros_like(torch.as_tensor(body_pos_w)).numpy(),
        "body_ang_vel_w": torch.zeros((3, 1, 3), dtype=torch.float32).numpy(),
    }

    resampled_log, resampled_foot_heights = _resample_motion_log(
        log,
        foot_heights,
        source_fps=30,
        target_fps=60,
    )

    expected_positions = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.0], dtype=torch.float32).numpy()
    expected_velocity = torch.gradient(
        torch.as_tensor(expected_positions).unsqueeze(-1),
        spacing=1.0 / 60.0,
        dim=0,
    )[0].numpy()

    assert resampled_log["fps"] == 60.0
    assert resampled_log["joint_pos"].shape == (6, 1)
    assert resampled_log["body_pos_w"].shape == (6, 1, 3)
    assert resampled_foot_heights.shape == (6, 2)
    assert torch.isfinite(torch.as_tensor(resampled_log["body_ang_vel_w"])).all()
    assert torch.allclose(torch.as_tensor(resampled_log["joint_pos"][:, 0]), torch.as_tensor(expected_positions))
    assert torch.allclose(torch.as_tensor(resampled_foot_heights[:, 0]), torch.as_tensor(expected_positions))
    assert torch.allclose(torch.as_tensor(resampled_log["joint_vel"]), torch.as_tensor(expected_velocity))
    assert torch.allclose(
        torch.as_tensor(resampled_log["body_lin_vel_w"])[:, 0, 0],
        torch.as_tensor(expected_velocity[:, 0]),
    )
    assert torch.count_nonzero(torch.as_tensor(resampled_log["body_ang_vel_w"])) == 0
    assert torch.allclose(
        torch.linalg.norm(torch.as_tensor(resampled_log["body_quat_w"]), dim=-1),
        torch.ones((6, 1), dtype=torch.float32),
        atol=1.0e-5,
    )


def test_gmr_motion_data_noop_when_smoothing_disabled(tmp_path: Path) -> None:
    root_pos = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.1, 0.0, 0.5],
            [0.2, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )
    root_rot = _yaw_quaternions(torch.tensor([0.0, 0.05, 0.1], dtype=torch.float32))
    joint_pos = torch.tensor(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    motion_file = tmp_path / "motion.pkl"
    _write_motion_pickle(motion_file, fps=30.0, root_pos=root_pos, root_rot_wxyz=root_rot, joint_pos=joint_pos)

    motion_data = GMRMotionData(
        str(motion_file),
        torch.device("cpu"),
        ["joint_0", "joint_1"],
        smooth_motion=False,
    )

    assert torch.allclose(motion_data.root_pos, root_pos)
    assert torch.allclose(motion_data.root_rot, root_rot)
    assert torch.allclose(motion_data.joint_pos, joint_pos)


def test_gmr_motion_data_handles_single_frame_clip(tmp_path: Path) -> None:
    root_pos = torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float32)
    root_rot = _yaw_quaternions(torch.tensor([0.0], dtype=torch.float32))
    joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    motion_file = tmp_path / "single_frame.pkl"
    _write_motion_pickle(motion_file, fps=30.0, root_pos=root_pos, root_rot_wxyz=root_rot, joint_pos=joint_pos)

    motion_data = GMRMotionData(
        str(motion_file),
        torch.device("cpu"),
        ["joint_0", "joint_1"],
        smooth_motion=True,
    )

    assert motion_data.root_lin_vel.shape == (1, 3)
    assert motion_data.root_ang_vel.shape == (1, 3)
    assert motion_data.joint_vel.shape == (1, 2)
    assert torch.count_nonzero(motion_data.root_lin_vel) == 0
    assert torch.count_nonzero(motion_data.root_ang_vel) == 0
    assert torch.count_nonzero(motion_data.joint_vel) == 0

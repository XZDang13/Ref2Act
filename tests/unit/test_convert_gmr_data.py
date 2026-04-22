from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ref2act.cli.convert import (
    ConversionOptions,
    SEGMENT_METHOD_ANCHOR,
    SEGMENT_METHOD_TIME,
    _MotionConversionState,
    _finalize_motion_slot,
    build_parser,
)

_BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]


def _build_motion_arrays(*, fps: float = 10.0, num_frames: int = 30, torso_x: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if torso_x is not None:
        torso_x = np.asarray(torso_x, dtype=np.float32)
        num_frames = int(torso_x.shape[0])
    dt = 1.0 / fps
    torso_x_values = (
        np.asarray(torso_x, dtype=np.float32).reshape(num_frames)
        if torso_x is not None
        else np.zeros(num_frames, dtype=np.float32)
    )

    joint_pos = np.zeros((num_frames, 2), dtype=np.float32)
    joint_pos[:, 0] = 0.1 * np.sin(np.linspace(0.0, 4.0 * np.pi, num_frames, dtype=np.float32))
    body_pos_w = np.zeros((num_frames, len(_BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = 1.0
    body_pos_w[:, 1, 0] = torso_x_values
    body_pos_w[:, 1, 2] = 2.0
    body_pos_w[:, 2, 1] = -0.1
    body_pos_w[:, 3, 1] = 0.1
    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    body_quat_w = np.zeros((num_frames, len(_BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0

    return {
        "joint_pos": joint_pos,
        "joint_vel": np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": np.zeros((num_frames, len(_BODY_NAMES), 3), dtype=np.float32),
    }


def _build_conversion_slot(tmp_path: Path, *, segment_method: str, torso_x: np.ndarray | None = None) -> Path:
    if torso_x is None:
        frame_index = np.arange(30, dtype=np.float32)
        torso_x = 0.1 + 0.05 * np.cos((2.0 * np.pi * frame_index) / 10.0)
    motion_arrays = _build_motion_arrays(torso_x=torso_x)
    output_file = tmp_path / f"{segment_method}.npz"
    slot = _MotionConversionState(
        env_id=0,
        input_file=tmp_path / "input.pkl",
        output_file=output_file,
        motion_data=SimpleNamespace(fps=10),
        log={
            "fps": 10.0,
            "joint_names": ["joint_0", "joint_1"],
            "body_names": list(_BODY_NAMES),
            "joint_pos": [frame.copy() for frame in motion_arrays["joint_pos"]],
            "joint_vel": [frame.copy() for frame in motion_arrays["joint_vel"]],
            "body_pos_w": [frame.copy() for frame in motion_arrays["body_pos_w"]],
            "body_quat_w": [frame.copy() for frame in motion_arrays["body_quat_w"]],
            "body_lin_vel_w": [frame.copy() for frame in motion_arrays["body_lin_vel_w"]],
            "body_ang_vel_w": [frame.copy() for frame in motion_arrays["body_ang_vel_w"]],
        },
        foot_height_frames=[frame.copy() for frame in motion_arrays["body_pos_w"][:, 2:4, 2]],
    )

    _finalize_motion_slot(
        slot,
        segment_bin_size=0.3,
        airborne_height_threshold=0.06,
        segment_method=segment_method,
        target_fps=None,
    )
    return output_file


def test_conversion_options_from_args_carries_segment_method() -> None:
    args = build_parser().parse_args(["--input_file", "motion.pkl", "--segment-method", "anchor"])
    options = ConversionOptions.from_args(args)

    assert options.segment_method == "anchor"


def test_finalize_motion_slot_anchor_mode_exports_anchor_metadata(tmp_path: Path) -> None:
    output_file = _build_conversion_slot(tmp_path, segment_method=SEGMENT_METHOD_ANCHOR)

    with np.load(output_file) as motion_data:
        assert {"segment_start_times", "segment_end_times", "segment_types"} <= set(motion_data.files)
        assert {"anchor_frame_labels", "anchor_segment_start_times", "anchor_segment_end_times"} <= set(motion_data.files)
        assert motion_data["anchor_selection_version"].dtype == np.int64
        assert motion_data["anchor_frame_labels"].dtype == np.int8
        assert motion_data["anchor_segment_labels"].dtype == np.int8
        assert motion_data["anchor_frame_indices"].dtype == np.int64
        assert motion_data["anchor_times"].dtype == np.float32
        assert motion_data["anchor_scores"].dtype == np.float32
        assert motion_data["anchor_support_modes"].dtype == np.int8
        assert motion_data["anchor_frame_indices"].shape[0] > 0
        if motion_data["anchor_times"].shape[0] > 1:
            assert float(motion_data["anchor_times"][1] - motion_data["anchor_times"][0]) >= 0.35 - 1.0e-6


def test_finalize_motion_slot_anchor_mode_inserts_bootstrap_and_trims_tail_anchor(tmp_path: Path, capsys) -> None:
    torso_x = np.zeros(30, dtype=np.float32)
    torso_x[:4] = 1.0
    output_file = _build_conversion_slot(
        tmp_path,
        segment_method=SEGMENT_METHOD_ANCHOR,
        torso_x=torso_x,
    )

    captured = capsys.readouterr()
    with np.load(output_file) as motion_data:
        anchor_frame_indices = motion_data["anchor_frame_indices"]
        anchor_times = motion_data["anchor_times"]
        frame_labels = motion_data["anchor_frame_labels"]
        segment_labels = motion_data["anchor_segment_labels"]
        duration = float(motion_data["joint_pos"].shape[0]) / float(np.asarray(motion_data["fps"]).item())

        assert anchor_frame_indices[0] == 0
        assert np.all(np.diff(anchor_frame_indices) > 0)
        assert anchor_frame_indices.dtype == np.int64
        assert frame_labels[0] != 2
        assert segment_labels[0] != 2
        assert np.all(anchor_times[1:] <= np.float32(duration - 0.30))

    assert "bootstrap_start=yes" in captured.out
    assert "tail_trimmed=1" in captured.out


def test_finalize_motion_slot_anchor_mode_logs_fallback_promotion(tmp_path: Path, capsys) -> None:
    torso_x = np.full(10, 0.8, dtype=np.float32)
    torso_x[4] = 0.0
    output_file = _build_conversion_slot(
        tmp_path,
        segment_method=SEGMENT_METHOD_ANCHOR,
        torso_x=torso_x,
    )

    captured = capsys.readouterr()
    with np.load(output_file) as motion_data:
        green_segment_durations = motion_data["anchor_segment_end_times"][
            motion_data["anchor_segment_labels"] == 2
        ] - motion_data["anchor_segment_start_times"][motion_data["anchor_segment_labels"] == 2]
        assert motion_data["anchor_frame_indices"].shape[0] > 0
        assert np.any(green_segment_durations > 0.15)
        assert motion_data["anchor_frame_indices"][0] == 0
    assert "strict=0" in captured.out
    assert "fallback_promotion=yes" in captured.out
    assert "bootstrap_start=no" in captured.out
    assert "tail_trimmed=1" in captured.out


def test_finalize_motion_slot_time_mode_omits_anchor_metadata(tmp_path: Path) -> None:
    output_file = _build_conversion_slot(tmp_path, segment_method=SEGMENT_METHOD_TIME)

    with np.load(output_file) as motion_data:
        assert {"segment_start_times", "segment_end_times", "segment_types"} <= set(motion_data.files)
        assert not any(key.startswith("anchor_") for key in motion_data.files)

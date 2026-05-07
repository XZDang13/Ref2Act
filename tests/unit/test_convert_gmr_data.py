from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ref2act.cli.convert import (
    ConversionOptions,
    SEGMENT_METHOD_ANCHOR,
    SEGMENT_METHOD_TIME,
    _MotionConversionState,
    _finalize_motion_slot,
    build_parser,
    split_motion_log_by_anchors,
)
from ref2act.motion import MotionLib

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


def _build_split_motion_log(
    *,
    fps: float = 10.0,
    num_frames: int,
    anchor_frames: tuple[int, ...],
) -> dict[str, object]:
    motion_arrays = _build_motion_arrays(fps=fps, num_frames=num_frames)
    duration = float(num_frames) / float(fps)
    anchor_frame_indices = np.asarray(anchor_frames, dtype=np.int64)
    num_anchors = int(anchor_frame_indices.shape[0])
    return {
        "fps": float(fps),
        "joint_names": ["joint_0", "joint_1"],
        "body_names": list(_BODY_NAMES),
        **motion_arrays,
        "segment_start_times": np.asarray([0.0], dtype=np.float32),
        "segment_end_times": np.asarray([duration], dtype=np.float32),
        "segment_types": np.asarray([0], dtype=np.int64),
        "anchor_selection_version": np.asarray(1, dtype=np.int64),
        "anchor_frame_labels": np.full(num_frames, 2, dtype=np.int8),
        "anchor_segment_start_times": np.asarray([0.0], dtype=np.float32),
        "anchor_segment_end_times": np.asarray([duration], dtype=np.float32),
        "anchor_segment_labels": np.asarray([2], dtype=np.int8),
        "anchor_frame_indices": anchor_frame_indices,
        "anchor_times": (anchor_frame_indices.astype(np.float32) / np.float32(fps)).astype(np.float32),
        "anchor_scores": np.ones(num_anchors, dtype=np.float32),
        "anchor_support_modes": np.full(num_anchors, 3, dtype=np.int8),
        "anchor_energy_norm": np.zeros(num_anchors, dtype=np.float32),
        "anchor_pose_extreme": np.zeros(num_anchors, dtype=np.float32),
        "anchor_torso_tilt_deg": np.zeros(num_anchors, dtype=np.float32),
        "anchor_joint_kinetic_energy": np.zeros(num_anchors, dtype=np.float32),
    }


def test_conversion_options_from_args_carries_segment_method() -> None:
    args = build_parser().parse_args(["--input_file", "motion.pkl", "--segment-method", "anchor"])
    options = ConversionOptions.from_args(args)

    assert options.segment_method == "anchor"
    assert options.split_by_anchors is False
    assert options.split_max_duration == 10.0
    assert options.split_min_duration == 2.0


def test_conversion_options_from_args_accepts_anchor_split_flags() -> None:
    args = build_parser().parse_args(
        [
            "--input_file",
            "motion.pkl",
            "--split-by-anchors",
            "--split-max-duration",
            "12.5",
            "--split-min-duration",
            "2.5",
        ]
    )
    options = ConversionOptions.from_args(args)

    assert options.split_by_anchors is True
    assert options.split_max_duration == 12.5
    assert options.split_min_duration == 2.5


def test_conversion_options_rejects_invalid_anchor_split_duration_order() -> None:
    args = build_parser().parse_args(
        [
            "--input_file",
            "motion.pkl",
            "--split-max-duration",
            "2.0",
            "--split-min-duration",
            "2.0",
        ]
    )

    with pytest.raises(ValueError, match="split-min-duration"):
        ConversionOptions.from_args(args)


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
        assert motion_data["anchor_joint_kinetic_energy"].dtype == np.float32
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


def test_finalize_motion_slot_anchor_mode_logs_hybrid_low_kinetic_anchor(tmp_path: Path, capsys) -> None:
    torso_x = np.full(10, 0.8, dtype=np.float32)
    torso_x[4] = 0.0
    output_file = _build_conversion_slot(
        tmp_path,
        segment_method=SEGMENT_METHOD_ANCHOR,
        torso_x=torso_x,
    )

    captured = capsys.readouterr()
    with np.load(output_file) as motion_data:
        assert np.array_equal(motion_data["anchor_frame_indices"], np.asarray([0, 4], dtype=np.int64))
        assert motion_data["anchor_frame_labels"][4] == 1
        assert motion_data["anchor_joint_kinetic_energy"].shape == motion_data["anchor_times"].shape
    assert "strict=0" in captured.out
    assert "fallback_promotion=no" in captured.out
    assert "bootstrap_start=yes" in captured.out
    assert "tail_trimmed=0" in captured.out


def test_finalize_motion_slot_time_mode_omits_anchor_metadata(tmp_path: Path) -> None:
    output_file = _build_conversion_slot(tmp_path, segment_method=SEGMENT_METHOD_TIME)

    with np.load(output_file) as motion_data:
        assert {"segment_start_times", "segment_end_times", "segment_types"} <= set(motion_data.files)
        assert not any(key.startswith("anchor_") for key in motion_data.files)


def test_split_motion_log_by_anchors_keeps_short_clip_unsplit(tmp_path: Path) -> None:
    log = _build_split_motion_log(num_frames=80, anchor_frames=(0, 30, 60))
    output_file = tmp_path / "motion.npz"

    written_files = split_motion_log_by_anchors(log, output_file, max_duration=10.0, min_duration=2.0)

    assert written_files == [output_file]
    assert output_file.exists()
    assert not (tmp_path / "motion_000.npz").exists()
    with np.load(output_file) as motion_data:
        assert motion_data["joint_pos"].shape[0] == 80


def test_split_motion_log_by_anchors_writes_numbered_parts_and_rebases_metadata(tmp_path: Path) -> None:
    log = _build_split_motion_log(num_frames=260, anchor_frames=(0, 90, 100, 190, 200))
    output_file = tmp_path / "motion.npz"

    written_files = split_motion_log_by_anchors(log, output_file, max_duration=10.0, min_duration=2.0)

    assert written_files == [
        tmp_path / "motion_000.npz",
        tmp_path / "motion_001.npz",
        tmp_path / "motion_002.npz",
    ]
    assert not output_file.exists()
    durations = []
    for path in written_files:
        with np.load(path) as motion_data:
            duration = motion_data["joint_pos"].shape[0] / float(np.asarray(motion_data["fps"]).item())
            durations.append(duration)
            assert duration >= 2.0
            assert motion_data["segment_start_times"][0] == np.float32(0.0)
            assert motion_data["segment_end_times"][-1] == np.float32(duration)
            assert motion_data["anchor_segment_start_times"][0] == np.float32(0.0)
            assert motion_data["anchor_segment_end_times"][-1] == np.float32(duration)

    assert durations == [10.0, 10.0, 6.0]
    with np.load(written_files[1]) as second_part:
        np.testing.assert_array_equal(second_part["anchor_frame_indices"], np.asarray([0, 90], dtype=np.int64))
        np.testing.assert_allclose(second_part["anchor_times"], np.asarray([0.0, 9.0], dtype=np.float32))

    motion_lib = MotionLib(written_files)
    assert motion_lib.num_motions == 3
    assert motion_lib.all_clips_have_anchor_segments


def test_split_motion_log_by_anchors_uses_next_anchor_when_no_anchor_is_below_max(tmp_path: Path) -> None:
    log = _build_split_motion_log(num_frames=250, anchor_frames=(0, 110, 220))

    written_files = split_motion_log_by_anchors(
        log,
        tmp_path / "motion.npz",
        max_duration=10.0,
        min_duration=2.0,
    )

    durations = []
    for path in written_files:
        with np.load(path) as motion_data:
            durations.append(motion_data["joint_pos"].shape[0] / float(np.asarray(motion_data["fps"]).item()))
    assert durations == [11.0, 11.0, 3.0]


def test_split_motion_log_by_anchors_merges_short_final_remainder(tmp_path: Path) -> None:
    log = _build_split_motion_log(num_frames=215, anchor_frames=(0, 100, 200))

    written_files = split_motion_log_by_anchors(
        log,
        tmp_path / "motion.npz",
        max_duration=10.0,
        min_duration=2.0,
    )

    assert written_files == [tmp_path / "motion_000.npz", tmp_path / "motion_001.npz"]
    durations = []
    for path in written_files:
        with np.load(path) as motion_data:
            durations.append(motion_data["joint_pos"].shape[0] / float(np.asarray(motion_data["fps"]).item()))
    assert durations == [10.0, 11.5]


def test_split_motion_log_by_anchors_fails_without_future_min_duration_anchor(tmp_path: Path) -> None:
    log = _build_split_motion_log(num_frames=130, anchor_frames=(0, 10))

    with pytest.raises(ValueError, match="could not find an anchor"):
        split_motion_log_by_anchors(
            log,
            tmp_path / "motion.npz",
            max_duration=10.0,
            min_duration=2.0,
        )

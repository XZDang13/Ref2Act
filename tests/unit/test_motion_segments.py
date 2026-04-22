from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib
from ref2act.motion.segments import (
    ANCHOR_FRAME_LABEL_GREEN,
    ANCHOR_FRAME_LABEL_RED,
    ANCHOR_FRAME_LABEL_YELLOW,
    SEGMENT_TYPE_AIR_MERGE,
    SEGMENT_TYPE_TIME_BIN,
    build_anchor_selection_diagnostics,
    build_anchor_selection_metadata,
    build_contact_segments,
    infer_ground_contact_from_foot_heights,
    _postprocess_anchor_frame_indices,
)

_ANCHOR_BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]


def _write_motion_file(
    path: Path,
    *,
    fps: float = 10.0,
    num_frames: int = 10,
    joint_pos: np.ndarray | None = None,
    segment_start_times: np.ndarray | None = None,
    segment_end_times: np.ndarray | None = None,
    segment_types: np.ndarray | None = None,
    anchor_segment_start_times: np.ndarray | None = None,
    anchor_segment_end_times: np.ndarray | None = None,
    anchor_segment_labels: np.ndarray | None = None,
    anchor_frame_indices: np.ndarray | None = None,
    anchor_times: np.ndarray | None = None,
) -> None:
    if joint_pos is None:
        joint_pos = np.zeros((num_frames, 1), dtype=np.float32)
    else:
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        num_frames = int(joint_pos.shape[0])
    joint_vel = np.zeros_like(joint_pos)
    body_pos_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, 1, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, 1, 3), dtype=np.float32)

    payload = {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(["joint_0"]),
        "body_names": np.asarray(["body_0"]),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }
    if segment_start_times is not None:
        payload["segment_start_times"] = np.asarray(segment_start_times, dtype=np.float32)
        payload["segment_end_times"] = np.asarray(segment_end_times, dtype=np.float32)
        payload["segment_types"] = np.asarray(segment_types, dtype=np.int64)
    if anchor_segment_start_times is not None:
        payload["anchor_segment_start_times"] = np.asarray(anchor_segment_start_times, dtype=np.float32)
    if anchor_segment_end_times is not None:
        payload["anchor_segment_end_times"] = np.asarray(anchor_segment_end_times, dtype=np.float32)
    if anchor_segment_labels is not None:
        payload["anchor_segment_labels"] = np.asarray(anchor_segment_labels, dtype=np.int64)
    if anchor_frame_indices is not None:
        payload["anchor_frame_indices"] = np.asarray(anchor_frame_indices, dtype=np.int64)
    if anchor_times is not None:
        payload["anchor_times"] = np.asarray(anchor_times, dtype=np.float32)

    np.savez(path, **payload)


def _build_anchor_log(
    *,
    fps: float = 10.0,
    num_frames: int = 30,
    pelvis_x: np.ndarray | None = None,
    torso_x: np.ndarray | None = None,
    left_foot_z: np.ndarray | None = None,
    right_foot_z: np.ndarray | None = None,
    joint_pos: np.ndarray | None = None,
) -> dict[str, object]:
    def _resolve(values: np.ndarray | None, *, default: float) -> np.ndarray:
        if values is None:
            return np.full(num_frames, default, dtype=np.float32)
        return np.asarray(values, dtype=np.float32).reshape(num_frames)

    dt = 1.0 / fps
    pelvis_x_values = _resolve(pelvis_x, default=0.0)
    torso_x_values = _resolve(torso_x, default=0.0)
    left_foot_z_values = _resolve(left_foot_z, default=0.0)
    right_foot_z_values = _resolve(right_foot_z, default=0.0)

    if joint_pos is None:
        joint_pos_values = np.zeros((num_frames, 2), dtype=np.float32)
    else:
        joint_pos_values = np.asarray(joint_pos, dtype=np.float32).reshape(num_frames, -1)

    body_pos_w = np.zeros((num_frames, len(_ANCHOR_BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 0] = pelvis_x_values
    body_pos_w[:, 0, 2] = 1.0
    body_pos_w[:, 1, 0] = torso_x_values
    body_pos_w[:, 1, 2] = 2.0
    body_pos_w[:, 2, 1] = -0.1
    body_pos_w[:, 2, 2] = left_foot_z_values
    body_pos_w[:, 3, 1] = 0.1
    body_pos_w[:, 3, 2] = right_foot_z_values

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0).astype(np.float32)
    joint_vel = np.gradient(joint_pos_values, dt, axis=0).astype(np.float32)
    body_quat_w = np.zeros((num_frames, len(_ANCHOR_BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0

    return {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(["joint_0", "joint_1"]),
        "body_names": np.asarray(_ANCHOR_BODY_NAMES),
        "joint_pos": joint_pos_values,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": np.zeros((num_frames, len(_ANCHOR_BODY_NAMES), 3), dtype=np.float32),
    }


def test_build_contact_segments_merges_airborne_jump_bins() -> None:
    has_ground_contact = np.asarray([True, True, False, False, False, True, True, True, True, True], dtype=bool)

    segment_start_times, segment_end_times, segment_types = build_contact_segments(
        has_ground_contact=has_ground_contact,
        dt=0.1,
        duration=1.0,
        bin_size=0.2,
    )

    assert np.allclose(segment_start_times, np.asarray([0.0, 0.8], dtype=np.float32))
    assert np.allclose(segment_end_times, np.asarray([0.8, 1.0], dtype=np.float32))
    assert np.array_equal(segment_types, np.asarray([SEGMENT_TYPE_AIR_MERGE, SEGMENT_TYPE_TIME_BIN]))
    assert (segment_end_times[0] - segment_start_times[0]) > 0.2


def test_build_contact_segments_keeps_grounded_walk_bins() -> None:
    has_ground_contact = np.ones(10, dtype=bool)

    segment_start_times, segment_end_times, segment_types = build_contact_segments(
        has_ground_contact=has_ground_contact,
        dt=0.1,
        duration=1.0,
        bin_size=0.2,
    )

    assert np.allclose(segment_start_times, np.asarray([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32))
    assert np.allclose(segment_end_times, np.asarray([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32))
    assert np.array_equal(segment_types, np.full(5, SEGMENT_TYPE_TIME_BIN, dtype=np.int64))


def test_infer_ground_contact_from_foot_heights_detects_airborne_frames() -> None:
    foot_heights = np.asarray(
        [
            [0.03, 0.02],
            [0.04, 0.03],
            [0.12, 0.11],
            [0.13, 0.12],
            [0.05, 0.03],
        ],
        dtype=np.float32,
    )

    ground_contact = infer_ground_contact_from_foot_heights(foot_heights, airborne_height_margin=0.06)

    assert np.array_equal(ground_contact, np.asarray([True, True, False, False, True], dtype=bool))


def test_infer_ground_contact_from_foot_heights_ignores_small_ground_drift() -> None:
    foot_heights = np.asarray(
        [
            [-0.02, -0.01],
            [0.00, 0.01],
            [0.03, 0.02],
            [0.12, 0.11],
            [0.13, 0.12],
            [0.02, 0.01],
        ],
        dtype=np.float32,
    )

    ground_contact = infer_ground_contact_from_foot_heights(foot_heights, airborne_height_margin=0.06)

    assert np.array_equal(ground_contact, np.asarray([True, True, True, False, False, True], dtype=bool))


def test_anchor_selection_marks_airborne_and_landing_impact_frames_red() -> None:
    left_foot_z = np.zeros(30, dtype=np.float32)
    right_foot_z = np.zeros(30, dtype=np.float32)
    left_foot_z[12:15] = 0.2
    right_foot_z[12:15] = 0.2
    diagnostics = build_anchor_selection_diagnostics(
        _build_anchor_log(left_foot_z=left_foot_z, right_foot_z=right_foot_z),
    )

    assert np.all(diagnostics.metadata.frame_labels[12:15] == ANCHOR_FRAME_LABEL_RED)
    assert np.all(diagnostics.metadata.frame_labels[diagnostics.near_landing_impact] == ANCHOR_FRAME_LABEL_RED)


def test_anchor_selection_marks_high_swing_single_support_apex_red() -> None:
    right_foot_z = np.zeros(30, dtype=np.float32)
    right_foot_z[10:13] = 0.36
    metadata = build_anchor_selection_metadata(
        _build_anchor_log(right_foot_z=right_foot_z),
    )

    assert metadata.frame_labels[11] == ANCHOR_FRAME_LABEL_RED


def test_anchor_selection_keeps_stable_low_energy_frames_green() -> None:
    pelvis_x = np.zeros(30, dtype=np.float32)
    pelvis_x[20:] = np.linspace(0.0, 4.5, 10, dtype=np.float32)
    joint_pos = np.zeros((30, 2), dtype=np.float32)
    joint_pos[20:, 0] = np.linspace(0.0, 3.0, 10, dtype=np.float32)
    metadata = build_anchor_selection_metadata(
        _build_anchor_log(pelvis_x=pelvis_x, torso_x=pelvis_x, joint_pos=joint_pos),
    )

    assert metadata.frame_labels[5] == ANCHOR_FRAME_LABEL_GREEN


def test_anchor_selection_keeps_walking_like_motion_resettable() -> None:
    num_frames = 40
    frame_index = np.arange(num_frames, dtype=np.float32)
    pelvis_x = np.linspace(0.0, 1.5, num_frames, dtype=np.float32)
    swing_wave = 0.06 * np.sin((2.0 * np.pi * frame_index) / 20.0) ** 2
    left_foot_z = np.where(frame_index < 20.0, swing_wave, 0.0).astype(np.float32)
    right_foot_z = np.where(frame_index >= 20.0, swing_wave, 0.0).astype(np.float32)
    diagnostics = build_anchor_selection_diagnostics(
        _build_anchor_log(
            fps=20.0,
            num_frames=num_frames,
            pelvis_x=pelvis_x,
            torso_x=pelvis_x,
            left_foot_z=left_foot_z,
            right_foot_z=right_foot_z,
        ),
    )

    assert diagnostics.metadata.frame_indices.shape[0] > 0
    assert np.any(diagnostics.metadata.frame_labels == ANCHOR_FRAME_LABEL_GREEN)


def test_anchor_selection_only_selects_anchors_inside_hard_safe_mask() -> None:
    left_foot_z = np.zeros(40, dtype=np.float32)
    right_foot_z = np.zeros(40, dtype=np.float32)
    left_foot_z[12:15] = 0.2
    right_foot_z[12:15] = 0.2
    right_foot_z[24:27] = 0.36
    diagnostics = build_anchor_selection_diagnostics(
        _build_anchor_log(
            fps=20.0,
            num_frames=40,
            right_foot_z=right_foot_z,
            left_foot_z=left_foot_z,
        ),
    )
    hard_safe_mask = ~(
        diagnostics.airborne
        | diagnostics.no_support
        | diagnostics.near_landing_impact
        | diagnostics.high_swing_pose
        | (diagnostics.torso_tilt_deg > 35.0)
    )

    non_bootstrap_anchor_indices = diagnostics.metadata.frame_indices[diagnostics.metadata.frame_indices != 0]
    assert np.all(hard_safe_mask[non_bootstrap_anchor_indices])


def test_anchor_selection_promotes_relaxed_fallback_when_strict_anchor_count_is_zero() -> None:
    torso_x = np.full(12, 0.8, dtype=np.float32)
    torso_x[5] = 0.0
    diagnostics = build_anchor_selection_diagnostics(_build_anchor_log(num_frames=12, torso_x=torso_x))
    metadata = diagnostics.metadata

    green_segment_durations = metadata.segment_end_times[metadata.segment_labels == ANCHOR_FRAME_LABEL_GREEN] - metadata.segment_start_times[
        metadata.segment_labels == ANCHOR_FRAME_LABEL_GREEN
    ]

    assert diagnostics.strict_anchor_frame_indices.shape[0] == 0
    assert diagnostics.used_fallback_promotion is True
    assert metadata.frame_indices.shape[0] > 0
    assert np.any(metadata.frame_labels == ANCHOR_FRAME_LABEL_GREEN)
    assert np.any(green_segment_durations > 0.15)


def test_anchor_selection_keeps_multiple_anchors_in_long_green_interval() -> None:
    frame_index = np.arange(30, dtype=np.float32)
    torso_x = 0.1 + 0.05 * np.cos((2.0 * np.pi * frame_index) / 10.0)
    metadata = build_anchor_selection_metadata(_build_anchor_log(torso_x=torso_x))

    assert metadata.frame_indices.shape[0] >= 3
    non_bootstrap_times = metadata.times[metadata.frame_indices != 0]
    assert np.all(np.diff(non_bootstrap_times) >= 0.35 - 1.0e-6)


def test_anchor_selection_inserts_bootstrap_start_anchor_without_promoting_opening_segment() -> None:
    torso_x = np.zeros(30, dtype=np.float32)
    torso_x[:4] = 1.0
    diagnostics = build_anchor_selection_diagnostics(_build_anchor_log(torso_x=torso_x))

    assert diagnostics.bootstrap_start_anchor_inserted is True
    assert diagnostics.metadata.frame_indices[0] == 0
    assert diagnostics.metadata.frame_labels[0] == ANCHOR_FRAME_LABEL_RED
    assert diagnostics.metadata.segment_labels[0] == ANCHOR_FRAME_LABEL_RED


def test_postprocess_anchor_frame_indices_matches_walk_like_bootstrap_case() -> None:
    frame_indices, bootstrap_start_anchor_inserted, num_tail_trimmed_anchors = _postprocess_anchor_frame_indices(
        np.asarray([18, 45, 71, 100, 125], dtype=np.int64),
        dt=0.02,
        duration=2.84,
        min_start_spacing_seconds=0.35,
        min_future_horizon_seconds=0.30,
    )

    assert bootstrap_start_anchor_inserted is True
    assert num_tail_trimmed_anchors == 0
    assert np.array_equal(frame_indices, np.asarray([0, 18, 45, 71, 100, 125], dtype=np.int64))


def test_postprocess_anchor_frame_indices_trims_tail_anchor_in_squat_like_case() -> None:
    frame_indices, bootstrap_start_anchor_inserted, num_tail_trimmed_anchors = _postprocess_anchor_frame_indices(
        np.asarray([12, 30, 146, 222, 244, 270, 296], dtype=np.int64),
        dt=0.02,
        duration=5.94,
        min_start_spacing_seconds=0.35,
        min_future_horizon_seconds=0.30,
    )

    assert bootstrap_start_anchor_inserted is True
    assert num_tail_trimmed_anchors == 1
    assert np.array_equal(frame_indices, np.asarray([0, 30, 146, 222, 244, 270], dtype=np.int64))


def test_postprocess_anchor_frame_indices_can_leave_only_bootstrap_anchor() -> None:
    frame_indices, bootstrap_start_anchor_inserted, num_tail_trimmed_anchors = _postprocess_anchor_frame_indices(
        np.asarray([9, 8, 9], dtype=np.int64),
        dt=0.1,
        duration=1.0,
        min_start_spacing_seconds=0.35,
        min_future_horizon_seconds=0.30,
    )

    assert bootstrap_start_anchor_inserted is True
    assert num_tail_trimmed_anchors == 2
    assert frame_indices.dtype == np.int64
    assert np.array_equal(frame_indices, np.asarray([0], dtype=np.int64))


def test_postprocess_anchor_frame_indices_drops_first_learned_anchor_when_too_close_to_bootstrap() -> None:
    frame_indices, bootstrap_start_anchor_inserted, num_tail_trimmed_anchors = _postprocess_anchor_frame_indices(
        np.asarray([1, 121, 152], dtype=np.int64),
        dt=0.02,
        duration=4.0,
        min_start_spacing_seconds=0.35,
        min_future_horizon_seconds=0.30,
    )

    assert bootstrap_start_anchor_inserted is True
    assert num_tail_trimmed_anchors == 0
    assert np.array_equal(frame_indices, np.asarray([0, 121, 152], dtype=np.int64))


def test_motion_lib_loads_segment_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.2, 0.8], dtype=np.float32),
        segment_end_times=np.asarray([0.2, 0.8, 1.0], dtype=np.float32),
        segment_types=np.asarray(
            [SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_AIR_MERGE, SEGMENT_TYPE_TIME_BIN],
            dtype=np.int64,
        ),
    )

    motion_lib = MotionLib([motion_file])
    clip = motion_lib.get_clip(0)

    assert clip.has_segments
    assert clip.num_segments == 3
    assert motion_lib.all_clips_have_segments
    assert motion_lib.motion_num_segments.tolist() == [3]
    assert np.allclose(clip.segment_start_times.cpu().numpy(), np.asarray([0.0, 0.2, 0.8], dtype=np.float32))


def test_motion_lib_loads_anchor_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_motion_file(
        motion_file,
        anchor_segment_start_times=np.asarray([0.0, 0.4, 0.7], dtype=np.float32),
        anchor_segment_end_times=np.asarray([0.4, 0.7, 1.0], dtype=np.float32),
        anchor_segment_labels=np.asarray(
            [ANCHOR_FRAME_LABEL_GREEN, ANCHOR_FRAME_LABEL_YELLOW, ANCHOR_FRAME_LABEL_RED],
            dtype=np.int64,
        ),
        anchor_frame_indices=np.asarray([1, 4, 7], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.4, 0.7], dtype=np.float32),
    )

    motion_lib = MotionLib([motion_file])
    clip = motion_lib.get_clip(0)

    assert clip.has_anchor_segments
    assert clip.num_anchor_segments == 3
    assert motion_lib.all_clips_have_anchor_segments
    assert motion_lib.motion_num_anchor_segments.tolist() == [3]
    assert np.allclose(clip.anchor_times.cpu().numpy(), np.asarray([0.1, 0.4, 0.7], dtype=np.float32))


def test_motion_lib_rejects_partial_anchor_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "partial_anchor_motion.npz"
    _write_motion_file(
        motion_file,
        anchor_segment_start_times=np.asarray([0.0, 0.4], dtype=np.float32),
        anchor_segment_end_times=np.asarray([0.4, 1.0], dtype=np.float32),
        anchor_segment_labels=np.asarray([ANCHOR_FRAME_LABEL_GREEN, ANCHOR_FRAME_LABEL_RED], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="missing part of the anchor metadata"):
        MotionLib([motion_file])


@pytest.mark.parametrize(
    ("anchor_segment_labels", "anchor_frame_indices", "anchor_times", "match"),
    [
        (
            np.asarray([ANCHOR_FRAME_LABEL_GREEN, 9], dtype=np.int64),
            np.asarray([1, 5], dtype=np.int64),
            np.asarray([0.1, 0.5], dtype=np.float32),
            "unknown anchor label ids",
        ),
        (
            np.asarray([ANCHOR_FRAME_LABEL_GREEN, ANCHOR_FRAME_LABEL_RED], dtype=np.int64),
            np.asarray([5, 4], dtype=np.int64),
            np.asarray([0.5, 0.4], dtype=np.float32),
            "anchor_frame_indices must be sorted",
        ),
        (
            np.asarray([ANCHOR_FRAME_LABEL_GREEN, ANCHOR_FRAME_LABEL_RED], dtype=np.int64),
            np.asarray([1, 5], dtype=np.int64),
            np.asarray([0.1, 1.2], dtype=np.float32),
            "outside the clip duration",
        ),
    ],
)
def test_motion_lib_rejects_invalid_anchor_selection_metadata(
    tmp_path: Path,
    anchor_segment_labels: np.ndarray,
    anchor_frame_indices: np.ndarray,
    anchor_times: np.ndarray,
    match: str,
) -> None:
    motion_file = tmp_path / "invalid_anchor_motion.npz"
    _write_motion_file(
        motion_file,
        anchor_segment_start_times=np.asarray([0.0, 0.4], dtype=np.float32),
        anchor_segment_end_times=np.asarray([0.4, 1.0], dtype=np.float32),
        anchor_segment_labels=anchor_segment_labels,
        anchor_frame_indices=anchor_frame_indices,
        anchor_times=anchor_times,
    )

    with pytest.raises(ValueError, match=match):
        MotionLib([motion_file])


def test_motion_lib_rejects_invalid_anchor_segment_partition(tmp_path: Path) -> None:
    motion_file = tmp_path / "invalid_anchor_partition.npz"
    _write_motion_file(
        motion_file,
        anchor_segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        anchor_segment_end_times=np.asarray([0.4, 1.0], dtype=np.float32),
        anchor_segment_labels=np.asarray([ANCHOR_FRAME_LABEL_GREEN, ANCHOR_FRAME_LABEL_RED], dtype=np.int64),
        anchor_frame_indices=np.asarray([1, 6], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.6], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="contiguous partition"):
        MotionLib([motion_file])


def test_motion_lib_samples_using_clip_fps_timeline(tmp_path: Path) -> None:
    motion_file = tmp_path / "sampled_motion.npz"
    _write_motion_file(
        motion_file,
        fps=30.0,
        joint_pos=np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
    )

    motion_lib = MotionLib([motion_file])
    sampled = motion_lib.sample_motion(
        motion_ids=torch.zeros(3, dtype=torch.long),
        times=torch.tensor([0.0, 1.0 / 60.0, 3.0 / 30.0], dtype=torch.float32),
    )

    assert np.allclose(sampled["joint_pos"].cpu().numpy().reshape(-1), np.asarray([0.0, 0.5, 2.0], dtype=np.float32))

from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib
from ref2act.motion.segments import (
    SEGMENT_TYPE_AIR_MERGE,
    SEGMENT_TYPE_TIME_BIN,
    build_anchor_selection_diagnostics,
    build_anchor_selection_metadata,
    build_contact_segments,
    infer_ground_contact_from_foot_heights,
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
    anchor_selection_version: int | None = 3,
    anchor_frame_indices: np.ndarray | None = None,
    anchor_times: np.ndarray | None = None,
    anchor_joint_kinetic_energy: np.ndarray | None = None,
    extra_payload: dict[str, np.ndarray] | None = None,
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
    if anchor_frame_indices is not None and anchor_selection_version is not None:
        payload["anchor_selection_version"] = np.asarray(anchor_selection_version, dtype=np.int64)
    if anchor_frame_indices is not None:
        payload["anchor_frame_indices"] = np.asarray(anchor_frame_indices, dtype=np.int64)
    if anchor_times is not None:
        payload["anchor_times"] = np.asarray(anchor_times, dtype=np.float32)
    if anchor_joint_kinetic_energy is not None:
        payload["anchor_joint_kinetic_energy"] = np.asarray(anchor_joint_kinetic_energy, dtype=np.float32)
    elif anchor_frame_indices is not None:
        payload["anchor_joint_kinetic_energy"] = np.zeros(
            np.asarray(anchor_frame_indices).reshape(-1).shape,
            dtype=np.float32,
        )
    if extra_payload is not None:
        payload.update(extra_payload)

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
    joint_vel: np.ndarray | None = None,
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
    if joint_vel is None:
        joint_vel_values = np.gradient(joint_pos_values, dt, axis=0).astype(np.float32)
    else:
        joint_vel_values = np.asarray(joint_vel, dtype=np.float32).reshape(num_frames, -1)
    body_quat_w = np.zeros((num_frames, len(_ANCHOR_BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0

    return {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(["joint_0", "joint_1"]),
        "body_names": np.asarray(_ANCHOR_BODY_NAMES),
        "joint_pos": joint_pos_values,
        "joint_vel": joint_vel_values,
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


def test_build_contact_segments_avoids_roundoff_extra_bin_at_exact_duration_multiple() -> None:
    has_ground_contact = np.ones(63, dtype=bool)

    segment_start_times, segment_end_times, segment_types = build_contact_segments(
        has_ground_contact=has_ground_contact,
        dt=1.0 / 30.0,
        duration=63.0 / 30.0,
        bin_size=0.3,
    )

    assert np.all(segment_end_times > segment_start_times)
    assert np.isclose(segment_end_times[-1], np.float32(2.1))
    assert segment_start_times.shape == (7,)
    assert np.array_equal(segment_types, np.full(7, SEGMENT_TYPE_TIME_BIN, dtype=np.int64))


def test_build_contact_segments_merges_unrepresentable_float32_tail_bin() -> None:
    duration = 10.0 + 2.0e-7
    has_ground_contact = np.ones(100, dtype=bool)

    segment_start_times, segment_end_times, segment_types = build_contact_segments(
        has_ground_contact=has_ground_contact,
        dt=duration / has_ground_contact.shape[0],
        duration=duration,
        bin_size=10.0,
    )

    assert np.all(segment_end_times > segment_start_times)
    assert np.array_equal(segment_start_times, np.asarray([0.0], dtype=np.float32))
    assert np.array_equal(segment_end_times, np.asarray([np.float32(duration)], dtype=np.float32))
    assert np.array_equal(segment_types, np.asarray([SEGMENT_TYPE_TIME_BIN], dtype=np.int64))


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


def test_anchor_selection_uses_safe_local_kinetic_minima() -> None:
    energy = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 0.5, 2.0, 3.0], dtype=np.float32)
    metadata = build_anchor_selection_metadata(_build_anchor_log(num_frames=11, joint_vel=energy[:, None]))

    assert np.array_equal(metadata.frame_indices, np.asarray([0, 4, 8], dtype=np.int64))
    assert np.allclose(metadata.times, np.asarray([0.0, 0.4, 0.8], dtype=np.float32))
    assert np.allclose(metadata.joint_kinetic_energy, np.asarray([5.0, 1.0, 0.5], dtype=np.float32))


def test_anchor_selection_skips_unsafe_local_minima() -> None:
    energy = np.asarray([5.0, 1.0, 5.0, 4.0, 0.5, 4.0, 5.0], dtype=np.float32)
    left_foot_z = np.zeros(7, dtype=np.float32)
    right_foot_z = np.zeros(7, dtype=np.float32)
    left_foot_z[1] = 0.2
    right_foot_z[1] = 0.2

    diagnostics = build_anchor_selection_diagnostics(
        _build_anchor_log(
            num_frames=7,
            left_foot_z=left_foot_z,
            right_foot_z=right_foot_z,
            joint_vel=energy[:, None],
        )
    )

    assert not diagnostics.safe_mask[1]
    assert diagnostics.safe_mask[4]
    assert np.array_equal(diagnostics.metadata.frame_indices, np.asarray([0, 4], dtype=np.int64))


def test_anchor_selection_falls_back_to_lowest_energy_safe_frame_when_no_safe_minimum() -> None:
    energy = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    torso_x = np.ones(8, dtype=np.float32)
    torso_x[5] = 0.0

    diagnostics = build_anchor_selection_diagnostics(
        _build_anchor_log(num_frames=8, torso_x=torso_x, joint_vel=energy[:, None])
    )

    assert diagnostics.used_lowest_energy_fallback is True
    assert np.array_equal(diagnostics.metadata.frame_indices, np.asarray([0, 5], dtype=np.int64))


def test_anchor_selection_keeps_frame_zero_when_no_safe_frames() -> None:
    torso_x = np.ones(5, dtype=np.float32)

    diagnostics = build_anchor_selection_diagnostics(_build_anchor_log(num_frames=5, torso_x=torso_x))

    assert not np.any(diagnostics.safe_mask)
    assert diagnostics.used_lowest_energy_fallback is False
    assert np.array_equal(diagnostics.metadata.frame_indices, np.asarray([0], dtype=np.int64))


def test_anchor_selection_spacing_keeps_lower_energy_minimum() -> None:
    energy = np.asarray([9.0, 8.0, 7.0, 6.0, 5.0, 1.0, 5.0, 0.2, 5.0, 6.0], dtype=np.float32)
    metadata = build_anchor_selection_metadata(_build_anchor_log(num_frames=10, joint_vel=energy[:, None]))

    assert np.array_equal(metadata.frame_indices, np.asarray([0, 7], dtype=np.int64))


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
        anchor_frame_indices=np.asarray([1, 4, 7], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.4, 0.7], dtype=np.float32),
        anchor_joint_kinetic_energy=np.asarray([3.0, 1.0, 2.0], dtype=np.float32),
    )

    motion_lib = MotionLib([motion_file])
    clip = motion_lib.get_clip(0)

    assert clip.has_anchor_segments
    assert clip.num_anchor_segments == 3
    assert motion_lib.all_clips_have_anchor_segments
    assert motion_lib.motion_num_anchor_segments.tolist() == [3]
    assert np.array_equal(clip.anchor_frame_indices.cpu().numpy(), np.asarray([1, 4, 7], dtype=np.int64))
    assert np.allclose(clip.anchor_times.cpu().numpy(), np.asarray([0.1, 0.4, 0.7], dtype=np.float32))


def test_motion_lib_rejects_partial_anchor_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "partial_anchor_motion.npz"
    _write_motion_file(
        motion_file,
        anchor_selection_version=None,
        anchor_frame_indices=np.asarray([1, 4], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.4], dtype=np.float32),
        anchor_joint_kinetic_energy=np.asarray([1.0, 0.5], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing part of the anchor metadata"):
        MotionLib([motion_file])


def test_motion_lib_rejects_legacy_anchor_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "legacy_anchor_motion.npz"
    _write_motion_file(
        motion_file,
        extra_payload={
            "anchor_selection_version": np.asarray(2, dtype=np.int64),
            "anchor_frame_labels": np.zeros(10, dtype=np.int8),
            "anchor_segment_start_times": np.asarray([0.0], dtype=np.float32),
            "anchor_segment_end_times": np.asarray([1.0], dtype=np.float32),
            "anchor_segment_labels": np.asarray([2], dtype=np.int8),
        },
    )

    with pytest.raises(ValueError, match="legacy anchor metadata"):
        MotionLib([motion_file])


@pytest.mark.parametrize(
    ("anchor_selection_version", "anchor_frame_indices", "anchor_times", "anchor_joint_kinetic_energy", "match"),
    [
        (
            2,
            np.asarray([1, 5], dtype=np.int64),
            np.asarray([0.1, 0.5], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            "unsupported anchor_selection_version",
        ),
        (
            3,
            np.asarray([5, 4], dtype=np.int64),
            np.asarray([0.5, 0.4], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            "anchor_frame_indices must be sorted",
        ),
        (
            3,
            np.asarray([1, 5], dtype=np.int64),
            np.asarray([0.1, 1.2], dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            "outside the clip duration",
        ),
        (
            3,
            np.asarray([1, 5], dtype=np.int64),
            np.asarray([0.1, 0.5], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            "anchor_joint_kinetic_energy",
        ),
    ],
)
def test_motion_lib_rejects_invalid_anchor_selection_metadata(
    tmp_path: Path,
    anchor_selection_version: int,
    anchor_frame_indices: np.ndarray,
    anchor_times: np.ndarray,
    anchor_joint_kinetic_energy: np.ndarray,
    match: str,
) -> None:
    motion_file = tmp_path / "invalid_anchor_motion.npz"
    _write_motion_file(
        motion_file,
        anchor_selection_version=anchor_selection_version,
        anchor_frame_indices=anchor_frame_indices,
        anchor_times=anchor_times,
        anchor_joint_kinetic_energy=anchor_joint_kinetic_energy,
    )

    with pytest.raises(ValueError, match=match):
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

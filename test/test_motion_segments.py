from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ref2Act.motion_lib import MotionLib
from Ref2Act.motion_segments import (
    SEGMENT_TYPE_AIR_MERGE,
    SEGMENT_TYPE_TIME_BIN,
    build_contact_segments,
    infer_ground_contact_from_foot_heights,
)


def _write_motion_file(
    path: Path,
    *,
    fps: float = 10.0,
    num_frames: int = 10,
    segment_start_times: np.ndarray | None = None,
    segment_end_times: np.ndarray | None = None,
    segment_types: np.ndarray | None = None,
) -> None:
    joint_pos = np.zeros((num_frames, 1), dtype=np.float32)
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

    np.savez(path, **payload)


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

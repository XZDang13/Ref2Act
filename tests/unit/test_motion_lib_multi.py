from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib


TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "motions"


def _write_motion_file(
    path: Path,
    *,
    fps: float,
    joint_pos: np.ndarray,
    name: str | None = None,
    include_segments: bool = False,
) -> None:
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
    if name is not None:
        payload["name"] = np.asarray(name)
    if include_segments:
        payload["segment_start_times"] = np.asarray([0.0], dtype=np.float32)
        payload["segment_end_times"] = np.asarray([num_frames / fps], dtype=np.float32)
        payload["segment_types"] = np.asarray([0], dtype=np.int64)
    np.savez(path, **payload)


def test_motion_lib_supports_mixed_motion_batches() -> None:
    motion_lib = MotionLib(
        [
            TEST_DATA_DIR / "jab.npz",
            TEST_DATA_DIR / "pick_up.npz",
        ]
    )

    motion_ids = torch.tensor([0, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.0], dtype=torch.float32)
    samples = motion_lib.sample_motion(motion_ids=motion_ids, times=times)

    jab_clip = motion_lib.get_clip(0)
    pick_clip = motion_lib.get_clip(1)

    assert motion_lib.num_motions == 2
    assert motion_lib.motion_names == ["jab", "pick_up"]
    assert motion_lib.get_duration(motion_ids)[0] != motion_lib.get_duration(motion_ids)[1]
    assert torch.allclose(samples["joint_pos"][0], jab_clip.joint_pos[0])
    assert torch.allclose(samples["joint_pos"][1], pick_clip.joint_pos[0])


def test_motion_lib_applies_offsets_per_selected_motion() -> None:
    motion_lib = MotionLib(
        [
            TEST_DATA_DIR / "jab.npz",
            TEST_DATA_DIR / "pick_up.npz",
        ]
    )

    motion_ids = torch.tensor([0, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.0], dtype=torch.float32)
    position_offsets = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=torch.float32)
    samples = motion_lib.sample_motion(
        motion_ids=motion_ids,
        times=times,
        position_offsets=position_offsets,
    )

    jab_clip = motion_lib.get_clip(0)
    pick_clip = motion_lib.get_clip(1)

    assert torch.allclose(
        samples["body_positions"][0],
        jab_clip.body_positions[0] + position_offsets[0].view(1, 3),
    )
    assert torch.allclose(
        samples["body_positions"][1],
        pick_clip.body_positions[0] + position_offsets[1].view(1, 3),
    )


def test_motion_lib_packed_sampling_matches_grouped_fractional_mixed_batch() -> None:
    motion_lib = MotionLib(
        [
            TEST_DATA_DIR / "jab.npz",
            TEST_DATA_DIR / "pick_up.npz",
            TEST_DATA_DIR / "walk.npz",
        ]
    )

    motion_ids = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long)
    times = torch.tensor([0.0, 0.25 / 30.0, 1.5 / 30.0, 10.25 / 30.0, 20.5 / 30.0, 83.75 / 30.0])
    position_offsets = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [-1.0, 0.5, 2.0],
            [0.25, -0.5, 0.75],
            [2.0, -1.0, 0.0],
            [-0.25, -0.75, 1.5],
        ],
        dtype=torch.float32,
    )

    assert motion_lib._packed_sampling_enabled
    packed_samples = motion_lib.sample_motion(
        motion_ids=motion_ids,
        times=times,
        position_offsets=position_offsets,
    )
    grouped_samples = motion_lib._sample_motion_grouped(
        motion_ids=motion_ids,
        times=times,
        position_offsets=position_offsets,
    )

    for key, packed_value in packed_samples.items():
        assert torch.allclose(packed_value, grouped_samples[key], atol=1.0e-5, rtol=1.0e-5)


def test_motion_lib_packed_sampling_handles_single_motion_batch() -> None:
    motion_lib = MotionLib([TEST_DATA_DIR / "jab.npz"])
    motion_ids = torch.zeros(4, dtype=torch.long)
    times = torch.tensor([0.0, 0.5 / 30.0, 5.25 / 30.0, 97.0 / 30.0], dtype=torch.float32)

    assert motion_lib._packed_sampling_enabled
    packed_samples = motion_lib.sample_motion(motion_ids=motion_ids, times=times)
    grouped_samples = motion_lib._sample_motion_grouped(motion_ids=motion_ids, times=times)

    for key, packed_value in packed_samples.items():
        assert torch.allclose(packed_value, grouped_samples[key], atol=1.0e-5, rtol=1.0e-5)


def test_motion_lib_packed_sampling_uses_per_motion_fps_and_frame_count(tmp_path: Path) -> None:
    motion_a = tmp_path / "slow.npz"
    motion_b = tmp_path / "fast.npz"
    _write_motion_file(
        motion_a,
        fps=10.0,
        joint_pos=np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32),
        name="slow",
    )
    _write_motion_file(
        motion_b,
        fps=20.0,
        joint_pos=np.asarray([[100.0], [120.0], [140.0], [160.0]], dtype=np.float32),
        name="fast",
    )

    motion_lib = MotionLib([motion_a, motion_b])
    motion_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    times = torch.tensor([0.05, 0.025, 1.0, 1.0], dtype=torch.float32)
    samples = motion_lib.sample_motion(motion_ids=motion_ids, times=times)

    assert motion_lib._packed_sampling_enabled
    assert torch.allclose(
        samples["joint_pos"].reshape(-1),
        torch.tensor([5.0, 110.0, 20.0, 160.0], dtype=torch.float32),
    )


def test_motion_lib_packed_sampling_uses_flat_frame_storage(tmp_path: Path) -> None:
    motion_a = tmp_path / "short.npz"
    motion_b = tmp_path / "long.npz"
    _write_motion_file(
        motion_a,
        fps=10.0,
        joint_pos=np.asarray([[0.0], [1.0]], dtype=np.float32),
        name="short",
    )
    _write_motion_file(
        motion_b,
        fps=10.0,
        joint_pos=np.asarray([[10.0], [11.0], [12.0], [13.0], [14.0]], dtype=np.float32),
        name="long",
    )

    motion_lib = MotionLib([motion_a, motion_b])

    assert motion_lib._packed_sampling_enabled
    assert motion_lib._packed_sampling_tensors["joint_pos"].shape[0] == 7
    assert torch.equal(motion_lib._packed_frame_offsets, torch.tensor([0, 2], dtype=torch.long))


def test_motion_lib_compact_storage_samples_like_non_compact(tmp_path: Path) -> None:
    motion_a = tmp_path / "short.npz"
    motion_b = tmp_path / "long.npz"
    _write_motion_file(
        motion_a,
        fps=10.0,
        joint_pos=np.asarray([[0.0], [1.0]], dtype=np.float32),
        name="short",
        include_segments=True,
    )
    _write_motion_file(
        motion_b,
        fps=10.0,
        joint_pos=np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32),
        name="long",
        include_segments=True,
    )

    full_motion_lib = MotionLib([motion_a, motion_b])
    compact_motion_lib = MotionLib([motion_a, motion_b], compact_after_packing=True)
    motion_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.05, 0.1, 0.2], dtype=torch.float32)

    full_samples = full_motion_lib.sample_motion(motion_ids=motion_ids, times=times)
    compact_samples = compact_motion_lib.sample_motion(motion_ids=motion_ids, times=times)

    assert compact_motion_lib._packed_sampling_enabled
    assert compact_motion_lib.get_clip(0).has_segments
    assert compact_motion_lib.get_clip(0).segment_start_times is not None
    with pytest.raises(RuntimeError, match="compact_after_packing=True"):
        _ = compact_motion_lib.get_clip(0).joint_pos
    for key, full_value in full_samples.items():
        assert torch.allclose(compact_samples[key], full_value, atol=1.0e-6, rtol=1.0e-6)


def test_motion_lib_default_storage_keeps_clip_frame_tensors(tmp_path: Path) -> None:
    motion_file = tmp_path / "motion.npz"
    _write_motion_file(
        motion_file,
        fps=10.0,
        joint_pos=np.asarray([[0.0], [1.0]], dtype=np.float32),
        name="motion",
    )

    motion_lib = MotionLib([motion_file])

    assert torch.equal(motion_lib.get_clip(0).joint_pos.reshape(-1), torch.tensor([0.0, 1.0]))

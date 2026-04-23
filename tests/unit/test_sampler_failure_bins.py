import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib, SegmentSource
from ref2act.motion.segments import (
    ANCHOR_FRAME_LABEL_GREEN,
    ANCHOR_FRAME_LABEL_RED,
    SEGMENT_TYPE_AIR_MERGE,
    SEGMENT_TYPE_TIME_BIN,
)


def _write_motion_file(
    path: Path,
    *,
    fps: float = 10.0,
    num_frames: int = 10,
    segment_start_times: np.ndarray | None = None,
    segment_end_times: np.ndarray | None = None,
    segment_types: np.ndarray | None = None,
    anchor_segment_start_times: np.ndarray | None = None,
    anchor_segment_end_times: np.ndarray | None = None,
    anchor_segment_labels: np.ndarray | None = None,
    anchor_frame_indices: np.ndarray | None = None,
    anchor_times: np.ndarray | None = None,
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


def _load_sampler_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.scene": sys.modules.get("isaaclab.scene"),
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
    }
    previous_attrs = {
        "scene": getattr(isaaclab, "scene", sentinel),
        "assets": getattr(isaaclab, "assets", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
    }

    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveScene = type("InteractiveScene", (), {})

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("Quaternion math should not run in failure-bin unit tests.")

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_mul = _unexpected_call
    math_mod.quat_inv = _unexpected_call
    math_mod.quat_apply = _unexpected_call
    math_mod.yaw_quat = _unexpected_call
    math_mod.quat_from_euler_xyz = _unexpected_call

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod

    sys.modules["isaaclab.scene"] = scene_mod
    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    isaaclab.scene = scene_mod
    isaaclab.assets = assets_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("ref2act.motion.sampling", None)
        return importlib.import_module("ref2act.motion.sampling")
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


def _write_anchor_sampler_motion_file(path: Path) -> None:
    _write_motion_file(
        path,
        anchor_segment_start_times=np.asarray([0.0], dtype=np.float32),
        anchor_segment_end_times=np.asarray([1.0], dtype=np.float32),
        anchor_segment_labels=np.asarray([ANCHOR_FRAME_LABEL_GREEN], dtype=np.int64),
        anchor_frame_indices=np.asarray([1, 5, 9], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
    )


def test_sampler_maps_times_inside_merged_segment_to_same_bin(tmp_path: Path) -> None:
    motion_file = tmp_path / "jump_segmented.npz"
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
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        bin_size=0.2,
        device=torch.device("cpu"),
    )

    motion_ids = torch.zeros(5, dtype=torch.long)
    times = torch.tensor([0.2, 0.25, 0.5, 0.79, 0.8], dtype=torch.float32)
    bin_indices = sampler._times_to_bins(motion_ids, times)

    assert torch.equal(bin_indices, torch.tensor([1, 1, 1, 1, 2], dtype=torch.long))


def test_sampler_failure_weighted_sampling_stays_inside_segment_bounds(tmp_path: Path) -> None:
    motion_file = tmp_path / "jump_segmented.npz"
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
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        bin_size=0.2,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=None,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])
    motion_ids = torch.zeros(256, dtype=torch.long)
    times = sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.FailureWeighted,
    )

    assert torch.all(times >= 0.2)
    assert torch.all(times < 0.8)


def test_failure_weighted_motion_selection_prefers_harder_motion(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    for motion_file in (motion_a, motion_b):
        _write_motion_file(
            motion_file,
            segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
            segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
            segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
        )

    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=4096,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([8.0, 8.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([10.0, 10.0])
    sampler.bin_fail_counts[1][:] = torch.tensor([2.0, 2.0])
    sampler.bin_sample_counts[1][:] = torch.tensor([10.0, 10.0])

    torch.manual_seed(0)
    motion_ids = sampler._sample_failure_weighted_motion_ids(torch.arange(4096))
    counts = torch.bincount(motion_ids, minlength=2)

    assert counts[0] > counts[1]


def test_failure_weighted_motion_selection_guard_keeps_all_motions_live(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    for motion_file in (motion_a, motion_b):
        _write_motion_file(
            motion_file,
            segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
            segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
            segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
        )

    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=4096,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.2,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([10.0, 10.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([10.0, 10.0])
    sampler.bin_fail_counts[1][:] = torch.zeros(2, dtype=torch.float32)
    sampler.bin_sample_counts[1][:] = torch.tensor([10.0, 10.0])

    torch.manual_seed(0)
    motion_ids = sampler._sample_failure_weighted_motion_ids(torch.arange(4096))
    counts = torch.bincount(motion_ids, minlength=2)

    assert counts[0] > counts[1]
    assert counts[1] > 200


def test_failure_weighted_motion_selection_probability_cap_limits_max_share(tmp_path: Path) -> None:
    motion_files = [tmp_path / f"motion_{motion_idx}.npz" for motion_idx in range(4)]
    for motion_file in motion_files:
        _write_motion_file(
            motion_file,
            segment_start_times=np.asarray([0.0], dtype=np.float32),
            segment_end_times=np.asarray([1.0], dtype=np.float32),
            segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
        )

    motion_lib = MotionLib(motion_files)
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=2.5,
        device=torch.device("cpu"),
    )

    motion_fail_counts = torch.tensor([100.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    motion_sample_counts = torch.ones(4, dtype=torch.float32)
    motion_probs = sampler._build_guarded_sampling_probabilities(
        motion_fail_counts,
        motion_sample_counts,
        temperature=1.0,
    )

    max_allowed_prob = sampler.failure_weight_max_uniform_ratio / 4.0
    assert bool(torch.isclose(motion_probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert motion_probs[0] > motion_probs[1]
    assert float(motion_probs.max().item()) <= max_allowed_prob + 1.0e-6


def test_failure_weighted_bin_guard_keeps_all_bins_live(tmp_path: Path) -> None:
    motion_file = tmp_path / "jump_segmented.npz"
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
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        bin_size=0.2,
        failure_weight_uniform_mix=0.2,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])

    torch.manual_seed(0)
    motion_ids = torch.zeros(4096, dtype=torch.long)
    _, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(motion_ids)
    counts = torch.bincount(target_bin_indices, minlength=3)

    assert torch.all(counts > 0)


def test_failure_weighted_bin_probability_cap_limits_max_share(tmp_path: Path) -> None:
    motion_file = tmp_path / "jump_segmented.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.25, 0.5, 0.75], dtype=np.float32),
        segment_end_times=np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN] * 4, dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=2.5,
        device=torch.device("cpu"),
    )

    bin_probs = sampler._build_guarded_sampling_probabilities(
        torch.tensor([100.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
        temperature=1.0,
    )

    max_allowed_prob = sampler.failure_weight_max_uniform_ratio / 4.0
    assert bool(torch.isclose(bin_probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert bin_probs[0] > bin_probs[1]
    assert float(bin_probs.max().item()) <= max_allowed_prob + 1.0e-6


def test_random_strategy_keeps_uniform_motion_selection(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    for motion_file in (motion_a, motion_b):
        _write_motion_file(
            motion_file,
            segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
            segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
            segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
        )

    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=4096,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.2,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([10.0, 10.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([10.0, 10.0])
    sampler.bin_fail_counts[1][:] = torch.zeros(2, dtype=torch.float32)
    sampler.bin_sample_counts[1][:] = torch.tensor([10.0, 10.0])

    torch.manual_seed(0)
    reset_sample = sampler.reset(torch.arange(4096), strategy=sampler_mod.SamplingStrategy.Random)
    counts = torch.bincount(reset_sample.motion_ids, minlength=2)

    assert counts[0] > 1500
    assert counts[1] > 1500


def test_sampler_random_sampling_uses_segment_start_times(tmp_path: Path) -> None:
    motion_file = tmp_path / "jump_segmented.npz"
    expected_start_times = torch.tensor([0.0, 0.2, 0.8], dtype=torch.float32)
    _write_motion_file(
        motion_file,
        segment_start_times=expected_start_times.numpy(),
        segment_end_times=np.asarray([0.2, 0.8, 1.0], dtype=np.float32),
        segment_types=np.asarray(
            [SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_AIR_MERGE, SEGMENT_TYPE_TIME_BIN],
            dtype=np.int64,
        ),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )

    torch.manual_seed(0)
    motion_ids = torch.zeros(256, dtype=torch.long)
    times = sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.Random,
    )

    unique_times = torch.unique(times)
    assert torch.equal(unique_times, expected_start_times)


def test_sampler_failure_weighted_sampling_requires_segment_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "legacy_motion.npz"
    _write_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        bin_size=0.2,
        device=torch.device("cpu"),
    )

    motion_ids = torch.zeros(4, dtype=torch.long)

    assert not sampler.supports_failure_weighted_sampling
    assert torch.equal(
        sampler.sample_times_for_motion_ids(motion_ids, strategy=sampler_mod.SamplingStrategy.Start),
        torch.zeros(4, dtype=torch.float32),
    )
    assert sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.Random,
    ).shape == motion_ids.shape
    with pytest.raises(RuntimeError, match="segment metadata"):
        sampler.sample_times_for_motion_ids(
            motion_ids,
            strategy=sampler_mod.SamplingStrategy.FailureWeighted,
        )


def test_sampler_exports_segment_source_default(tmp_path: Path) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )

    assert SegmentSource.Time is not None
    assert sampler.segment_source == sampler_mod.SegmentSource.Time
    assert SegmentSource.Time.name == "Time"


def test_anchor_random_sampling_uses_unique_reset_anchors(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    torch.manual_seed(0)
    motion_ids = torch.zeros(4096, dtype=torch.long)
    times, target_bin_indices = sampler._sample_rand_times_for_motion_ids(motion_ids)
    counts = torch.bincount(target_bin_indices, minlength=3)

    assert torch.equal(sampler.bin_reset_times[0], torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32))
    assert torch.all(counts > 0)
    assert torch.allclose(times, sampler.bin_reset_times[0][target_bin_indices])


def test_anchor_failure_weighted_sampling_uses_anchor_weights_and_reset_time(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=None,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])
    motion_ids = torch.zeros(128, dtype=torch.long)
    times, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(
        motion_ids,
    )

    assert torch.equal(target_bin_indices, torch.full((128,), 1, dtype=torch.long))
    assert torch.allclose(times, torch.full((128,), 0.5, dtype=torch.float32))


def test_anchor_failure_weighted_guard_keeps_all_anchors_live(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=2.5,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])

    torch.manual_seed(0)
    motion_ids = torch.zeros(4096, dtype=torch.long)
    times, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(motion_ids)
    counts = torch.bincount(target_bin_indices, minlength=3)

    assert torch.all(sampler.bin_reset_eligible[0])
    assert torch.all(counts > 0)
    assert torch.all((times == 0.1) | (times == 0.5) | (times == 0.9))


def test_anchor_failure_weighted_probability_cap_limits_anchor_mass(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        failure_weight_uniform_mix=0.0,
        failure_weight_max_uniform_ratio=2.5,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    probs = sampler._build_guarded_sampling_probabilities(
        torch.tensor([500.0, 100.0, 0.0], dtype=torch.float32),
        torch.ones(3, dtype=torch.float32),
        temperature=1.0,
        eligible_mask=sampler.bin_reset_eligible[0],
    )

    eligible_mask = sampler.bin_reset_eligible[0]
    num_eligible = int(eligible_mask.sum().item())
    max_allowed_prob = sampler.failure_weight_max_uniform_ratio / float(num_eligible)
    assert torch.all(eligible_mask)
    assert bool(torch.isclose(probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert float(probs[eligible_mask].max().item()) <= max_allowed_prob + 1.0e-6


def test_anchor_sample_and_fail_counts_accumulate_on_nearest_previous_anchor(tmp_path: Path, monkeypatch) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        sampler,
        "sample_motion_ids",
        lambda env_ids=None: torch.zeros(1, dtype=torch.long, device=sampler.device),
    )
    monkeypatch.setattr(
        sampler,
        "_sample_anchor_source_random_times_for_motion_ids",
        lambda motion_ids: (
            torch.tensor([0.5], dtype=torch.float32, device=sampler.device),
            torch.tensor([1], dtype=torch.long, device=sampler.device),
        ),
    )

    reset_sample = sampler.reset(torch.tensor([0]), strategy=sampler_mod.SamplingStrategy.Random)
    sampler.current_times[0] = 0.95
    sampler.record_failures(torch.tensor([0]))

    assert torch.equal(reset_sample.target_bin_indices, torch.tensor([1], dtype=torch.long))
    assert torch.allclose(reset_sample.times, torch.tensor([0.5], dtype=torch.float32))
    assert torch.equal(sampler.bin_sample_counts[0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(sampler.bin_fail_counts[0], torch.tensor([0.0, 0.0, 1.0]))


def test_anchor_source_requires_anchor_metadata(tmp_path: Path) -> None:
    motion_file = tmp_path / "legacy_motion.npz"
    _write_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(RuntimeError, match="segment-method anchor"):
        sampler_mod.Sampler(
            num_envs=1,
            motion_lib=motion_lib,
            dt=0.05,
            anchor_body_index=0,
            segment_source=sampler_mod.SegmentSource.Anchor,
            device=torch.device("cpu"),
        )


def test_anchor_source_requires_at_least_one_reset_anchor(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_without_reset_anchor.npz"
    _write_motion_file(
        motion_file,
        anchor_segment_start_times=np.asarray([0.0], dtype=np.float32),
        anchor_segment_end_times=np.asarray([1.0], dtype=np.float32),
        anchor_segment_labels=np.asarray([ANCHOR_FRAME_LABEL_RED], dtype=np.int64),
        anchor_frame_indices=np.asarray([], dtype=np.int64),
        anchor_times=np.asarray([], dtype=np.float32),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(RuntimeError, match="reset anchor"):
        sampler_mod.Sampler(
            num_envs=1,
            motion_lib=motion_lib,
            dt=0.05,
            anchor_body_index=0,
            segment_source=sampler_mod.SegmentSource.Anchor,
            device=torch.device("cpu"),
        )


def test_sampling_strategy_start_is_unchanged_for_anchor_source(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    motion_ids = torch.zeros(8, dtype=torch.long)
    times = sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.Start,
    )

    assert torch.equal(times, torch.zeros(8, dtype=torch.float32))


@pytest.mark.parametrize("uniform_mix", [-0.1, 1.1])
def test_sampler_rejects_invalid_failure_weight_uniform_mix(tmp_path: Path, uniform_mix: float) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(ValueError, match="failure_weight_uniform_mix"):
        sampler_mod.Sampler(
            num_envs=1,
            motion_lib=motion_lib,
            dt=0.05,
            anchor_body_index=0,
            failure_weight_uniform_mix=uniform_mix,
            device=torch.device("cpu"),
        )

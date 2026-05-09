import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib, SegmentSource
from ref2act.motion.segments import (
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
    anchor_selection_version: int | None = 3,
    anchor_frame_indices: np.ndarray | None = None,
    anchor_times: np.ndarray | None = None,
    anchor_joint_kinetic_energy: np.ndarray | None = None,
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
        anchor_frame_indices=np.asarray([1, 5, 9], dtype=np.int64),
        anchor_times=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        anchor_joint_kinetic_energy=np.asarray([0.3, 0.1, 0.2], dtype=np.float32),
    )


def _write_empty_anchor_motion_file(path: Path) -> None:
    _write_motion_file(
        path,
        anchor_frame_indices=np.asarray([], dtype=np.int64),
        anchor_times=np.asarray([], dtype=np.float32),
        anchor_joint_kinetic_energy=np.asarray([], dtype=np.float32),
    )


def _write_anchor_motion_file(path: Path, anchor_times: np.ndarray) -> None:
    _write_motion_file(
        path,
        num_frames=max(int(anchor_times.shape[0]) * 2, 10),
        anchor_frame_indices=np.arange(anchor_times.shape[0], dtype=np.int64),
        anchor_times=np.asarray(anchor_times, dtype=np.float32),
        anchor_joint_kinetic_energy=np.zeros(anchor_times.shape[0], dtype=np.float32),
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
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
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
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        device=torch.device("cpu"),
    )

    sampler.motion_fail_counts[:] = torch.tensor([16.0, 4.0])
    sampler.motion_sample_counts[:] = torch.tensor([20.0, 20.0])
    sampler._global_step = 1

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
        weight_fail=0.8,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        device=torch.device("cpu"),
    )

    sampler.motion_fail_counts[:] = torch.tensor([20.0, 0.0])
    sampler.motion_sample_counts[:] = torch.tensor([20.0, 20.0])
    sampler._global_step = 1

    torch.manual_seed(0)
    motion_ids = sampler._sample_failure_weighted_motion_ids(torch.arange(4096))
    counts = torch.bincount(motion_ids, minlength=2)

    assert counts[0] > counts[1]
    assert counts[1] > 200


def test_failure_weighted_motion_selection_explores_unvisited_motion(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    for motion_file in (motion_a, motion_b):
        _write_motion_file(
            motion_file,
            segment_start_times=np.asarray([0.0], dtype=np.float32),
            segment_end_times=np.asarray([1.0], dtype=np.float32),
            segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
        )

    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=4096,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        weight_fail=0.0,
        weight_novel=1.0,
        adaptive_uniform_ratio=0.0,
        device=torch.device("cpu"),
    )

    sampler.motion_assigned_counts[:] = torch.tensor([100.0, 0.0])
    sampler._global_step = 1

    torch.manual_seed(0)
    motion_ids = sampler._sample_failure_weighted_motion_ids(torch.arange(4096))
    counts = torch.bincount(motion_ids, minlength=2)

    assert counts[1] > counts[0]


def test_failure_weighted_motion_selection_cap_beta_limits_extreme_fail_rates(tmp_path: Path) -> None:
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
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        cap_beta=1.0,
        device=torch.device("cpu"),
    )

    sampler.motion_fail_counts[:] = torch.tensor([100.0, 10.0, 10.0, 10.0], dtype=torch.float32)
    sampler.motion_sample_counts[:] = torch.ones(4, dtype=torch.float32)
    sampler._global_step = 1
    motion_probs = sampler._build_motion_sampling_probabilities()

    assert bool(torch.isclose(motion_probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert motion_probs[0] > motion_probs[1]
    assert float(motion_probs[0].item()) <= 0.53


def test_mosaic_bin_sampling_ignores_sample_counts_without_failures(tmp_path: Path) -> None:
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
        adaptive_uniform_ratio=0.0,
        device=torch.device("cpu"),
    )

    probs = sampler._build_bin_sampling_probabilities(
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 100.0], dtype=torch.float32),
    )

    assert torch.allclose(probs, torch.tensor([0.5, 0.5], dtype=torch.float32))


def test_mosaic_bin_sampling_uses_failure_counts_not_bin_novelty(tmp_path: Path) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.33, 0.66], dtype=np.float32),
        segment_end_times=np.asarray([0.33, 0.66, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN] * 3, dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        adaptive_uniform_ratio=0.0,
        device=torch.device("cpu"),
    )

    probs = sampler._build_bin_sampling_probabilities(
        torch.tensor([3.0, 0.0, 0.0], dtype=torch.float32),
        torch.tensor([3.0, 100.0, 0.0], dtype=torch.float32),
    )

    assert torch.allclose(probs, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))


def test_mosaic_motion_warmup_ramp_returns_uniform_then_adaptive(tmp_path: Path) -> None:
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
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        adaptive_alpha=1.0,
        motion_sampling_warmup_s=0.2,
        motion_sampling_ramp_s=0.2,
        motion_sampling_schedule="linear",
        device=torch.device("cpu"),
    )

    sampler.motion_fail_counts[:] = torch.tensor([10.0, 0.0], dtype=torch.float32)
    sampler.motion_sample_counts[:] = torch.ones(2, dtype=torch.float32)
    warmup_probs = sampler._build_motion_sampling_probabilities()
    sampler.current_times[0] = 0.75
    sampler.record_failures(torch.tensor([0]))

    sampler._global_step = 8
    adaptive_probs = sampler._build_motion_sampling_probabilities()

    assert torch.allclose(warmup_probs, torch.tensor([0.5, 0.5], dtype=torch.float32))
    assert torch.equal(sampler.bin_fail_counts[0], torch.zeros(2, dtype=torch.float32))
    assert adaptive_probs[0] > 0.99
    assert adaptive_probs[1] < 0.01


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
        weight_fail=0.8,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.2,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])

    torch.manual_seed(0)
    motion_ids = torch.zeros(4096, dtype=torch.long)
    _, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(motion_ids)
    counts = torch.bincount(target_bin_indices, minlength=3)

    assert torch.all(counts > 0)


def test_failure_weighted_bin_kernel_smoothing_preserves_valid_mask(tmp_path: Path) -> None:
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
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        adaptive_kernel_size=3,
        adaptive_lambda=0.5,
        device=torch.device("cpu"),
    )

    eligible_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
    bin_probs = sampler._build_bin_sampling_probabilities(
        torch.tensor([100.0, 50.0, 0.0, 0.0], dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
        eligible_mask=eligible_mask,
    )

    assert bool(torch.isclose(bin_probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert float(bin_probs[1].item()) == 0.0
    assert torch.all(bin_probs[eligible_mask] >= 0.0)


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


def test_sampler_failure_weighted_sampling_uses_generated_time_bins_without_segment_metadata(tmp_path: Path) -> None:
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

    assert sampler.supports_failure_weighted_sampling
    assert sampler.num_bins_per_motion[0].item() == 5
    assert torch.equal(
        sampler.sample_times_for_motion_ids(motion_ids, strategy=sampler_mod.SamplingStrategy.Start),
        torch.zeros(4, dtype=torch.float32),
    )
    assert sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.Random,
    ).shape == motion_ids.shape
    times = sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.FailureWeighted,
    )
    assert times.shape == motion_ids.shape
    assert torch.all(times >= 0.0)
    assert torch.all(times <= motion_lib.motion_durations[0])


def test_sampler_can_skip_failure_bin_initialization(tmp_path: Path) -> None:
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
        enable_failure_bins=False,
        device=torch.device("cpu"),
    )

    assert not sampler._has_failure_bins()
    assert not sampler.supports_failure_weighted_sampling
    assert torch.equal(
        sampler.sample_times_for_motion_ids(
            torch.zeros(2, dtype=torch.long),
            strategy=sampler_mod.SamplingStrategy.Start,
        ),
        torch.zeros(2, dtype=torch.float32),
    )


def test_sampler_vectorized_times_to_bins_matches_segment_bounds(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_file(
        motion_a,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    _write_motion_file(
        motion_b,
        segment_start_times=np.asarray([0.0, 0.2, 0.4, 0.8], dtype=np.float32),
        segment_end_times=np.asarray([0.2, 0.4, 0.8, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN] * 4, dtype=np.int64),
    )
    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )

    motion_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.5, 0.19, 0.2, 0.99], dtype=torch.float32)

    assert torch.equal(sampler._times_to_bins(motion_ids, times), torch.tensor([0, 1, 0, 1, 3]))


def test_sampler_vectorized_bin_count_accumulation_preserves_list_views(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_file(
        motion_a,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    _write_motion_file(
        motion_b,
        segment_start_times=np.asarray([0.0, 0.25, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN] * 3, dtype=np.int64),
    )
    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )

    motion_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    bin_indices = torch.tensor([0, 1, 0, 2, 2], dtype=torch.long)
    sampler._accumulate_bin_counts(sampler.bin_sample_counts, motion_ids, bin_indices)

    assert torch.equal(sampler.bin_sample_counts[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(sampler.bin_sample_counts[1], torch.tensor([1.0, 0.0, 2.0]))
    sampler.bin_sample_counts[1][1] = 7.0
    assert sampler._bin_sample_counts_padded[1, 1].item() == 7.0


def test_sampler_failure_weighted_reset_returns_valid_time_bins(tmp_path: Path) -> None:
    motion_a = tmp_path / "motion_a.npz"
    motion_b = tmp_path / "motion_b.npz"
    _write_motion_file(
        motion_a,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    _write_motion_file(
        motion_b,
        segment_start_times=np.asarray([0.0, 0.2, 0.8], dtype=np.float32),
        segment_end_times=np.asarray([0.2, 0.8, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN] * 3, dtype=np.int64),
    )
    motion_lib = MotionLib([motion_a, motion_b])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=64,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        device=torch.device("cpu"),
    )

    reset_sample = sampler.reset(
        torch.arange(64, dtype=torch.long),
        strategy=sampler_mod.SamplingStrategy.FailureWeighted,
    )
    bin_starts = sampler._bin_start_times_padded[reset_sample.motion_ids, reset_sample.target_bin_indices]
    bin_ends = sampler._bin_end_times_padded[reset_sample.motion_ids, reset_sample.target_bin_indices]

    assert torch.all(reset_sample.target_bin_indices >= 0)
    assert torch.all(reset_sample.target_bin_indices < sampler.num_bins_per_motion[reset_sample.motion_ids])
    assert torch.all(reset_sample.times >= bin_starts)
    assert torch.all(reset_sample.times <= bin_ends)


def test_sampler_failure_weighted_reset_returns_valid_anchor_bins(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=32,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    reset_sample = sampler.reset(
        torch.arange(32, dtype=torch.long),
        strategy=sampler_mod.SamplingStrategy.FailureWeighted,
    )

    assert torch.all(reset_sample.target_bin_indices >= 0)
    assert torch.all(reset_sample.target_bin_indices < sampler.num_bins_per_motion[0])
    assert torch.allclose(reset_sample.times, sampler.bin_reset_times[0][reset_sample.target_bin_indices])


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


def test_anchor_random_sampling_uses_capped_bins_with_duplicate_reset_anchors(tmp_path: Path) -> None:
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
    counts = torch.bincount(target_bin_indices, minlength=5)

    assert torch.allclose(sampler.bin_start_times[0], torch.tensor([0.0, 0.3, 0.5, 0.8, 0.9], dtype=torch.float32))
    assert torch.allclose(sampler.bin_end_times[0], torch.tensor([0.3, 0.5, 0.8, 0.9, 1.0], dtype=torch.float32))
    assert torch.allclose(sampler.bin_reset_times[0], torch.tensor([0.1, 0.1, 0.5, 0.5, 0.9], dtype=torch.float32))
    assert torch.all(counts > 0)
    assert torch.allclose(times, sampler.bin_reset_times[0][target_bin_indices])


def test_anchor_source_makes_all_v3_anchor_bins_reset_eligible(
    tmp_path: Path,
) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_motion_file(
        motion_file,
        num_frames=20,
        anchor_frame_indices=np.asarray([0, 6], dtype=np.int64),
        anchor_times=np.asarray([0.0, 0.6], dtype=np.float32),
        anchor_joint_kinetic_energy=np.asarray([0.2, 0.1], dtype=np.float32),
    )
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

    assert torch.all(sampler.bin_end_times[0] - sampler.bin_start_times[0] <= 0.3 + 1.0e-6)
    assert torch.all(sampler.bin_reset_eligible[0])

    motion_ids = torch.zeros(256, dtype=torch.long)
    times, target_bin_indices = sampler._sample_rand_times_for_motion_ids(motion_ids)

    assert torch.all(target_bin_indices >= 0)
    assert torch.allclose(times, sampler.bin_reset_times[0][target_bin_indices])
    assert torch.all((times == 0.0) | (times == 0.6))


def test_anchor_random_sampling_skips_motion_clips_without_reset_anchors(tmp_path: Path, capsys) -> None:
    empty_motion = tmp_path / "empty_anchor.npz"
    valid_motion = tmp_path / "valid_anchor.npz"
    _write_empty_anchor_motion_file(empty_motion)
    _write_anchor_sampler_motion_file(valid_motion)
    motion_lib = MotionLib([empty_motion, valid_motion])
    sampler_mod = _load_sampler_module()

    sampler = sampler_mod.Sampler(
        num_envs=512,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    captured = capsys.readouterr()
    assert "anchor reset sampling will skip clips without eligible reset anchors" in captured.out
    assert torch.equal(sampler.motion_reset_eligible, torch.tensor([False, True], dtype=torch.bool))
    assert torch.all(sampler.bin_reset_eligible[0] == torch.zeros_like(sampler.bin_reset_eligible[0]))
    assert sampler.num_bins_per_motion[0].item() == 0

    torch.manual_seed(0)
    reset_sample = sampler.reset(torch.arange(512), strategy=sampler_mod.SamplingStrategy.Random)

    assert torch.equal(torch.unique(reset_sample.motion_ids), torch.tensor([1], dtype=torch.long))
    with pytest.raises(RuntimeError, match="without eligible reset anchors"):
        sampler.sample_times_for_motion_ids(
            torch.tensor([0], dtype=torch.long),
            strategy=sampler_mod.SamplingStrategy.Random,
        )


def test_anchor_failure_weighted_sampling_skips_motion_clips_without_reset_anchors(tmp_path: Path) -> None:
    empty_motion = tmp_path / "empty_anchor.npz"
    valid_motion = tmp_path / "valid_anchor.npz"
    _write_empty_anchor_motion_file(empty_motion)
    _write_anchor_sampler_motion_file(valid_motion)
    motion_lib = MotionLib([empty_motion, valid_motion])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=512,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    sampler.motion_fail_counts[:] = torch.tensor([100.0, 1.0])
    sampler.motion_sample_counts[:] = torch.ones(2, dtype=torch.float32)

    torch.manual_seed(0)
    reset_sample = sampler.reset(torch.arange(512), strategy=sampler_mod.SamplingStrategy.FailureWeighted)

    assert torch.equal(torch.unique(reset_sample.motion_ids), torch.tensor([1], dtype=torch.long))
    with pytest.raises(RuntimeError, match="without eligible reset anchors"):
        sampler.sample_times_for_motion_ids(
            torch.tensor([0], dtype=torch.long),
            strategy=sampler_mod.SamplingStrategy.FailureWeighted,
        )


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
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 0.0, 25.0, 0.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.ones(5, dtype=torch.float32)
    motion_ids = torch.zeros(128, dtype=torch.long)
    times, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(
        motion_ids,
    )

    assert torch.equal(target_bin_indices, torch.full((128,), 2, dtype=torch.long))
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
        weight_fail=0.8,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.2,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 0.0, 25.0, 0.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.ones(5, dtype=torch.float32)

    torch.manual_seed(0)
    motion_ids = torch.zeros(4096, dtype=torch.long)
    times, target_bin_indices = sampler._sample_failure_weighted_times_for_motion_ids(motion_ids)
    counts = torch.bincount(target_bin_indices, minlength=5)

    assert torch.all(sampler.bin_reset_eligible[0])
    assert torch.all(counts > 0)
    assert torch.all((times == 0.1) | (times == 0.5) | (times == 0.9))


def test_anchor_failure_weighted_kernel_smoothing_preserves_anchor_ineligibility(tmp_path: Path) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        weight_fail=1.0,
        weight_novel=0.0,
        adaptive_uniform_ratio=0.0,
        adaptive_kernel_size=3,
        adaptive_lambda=0.5,
        segment_source=sampler_mod.SegmentSource.Anchor,
        device=torch.device("cpu"),
    )
    sampler.bin_reset_eligible[0][1] = False
    sampler._bin_reset_eligible_padded[0, 1] = False

    probs = sampler._build_bin_sampling_probabilities(
        torch.tensor([500.0, 100.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        torch.ones(5, dtype=torch.float32),
        eligible_mask=sampler.bin_reset_eligible[0],
    )

    eligible_mask = sampler.bin_reset_eligible[0]
    assert bool(torch.isclose(probs.sum(), torch.tensor(1.0, dtype=torch.float32)).item())
    assert float(probs[1].item()) == 0.0
    assert torch.all(probs[eligible_mask] >= 0.0)


def test_anchor_failure_counts_accumulate_on_nearest_previous_anchor_for_failure_weighted(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
        "_sample_failure_weighted_times_for_motion_ids",
        lambda motion_ids: (
            torch.tensor([0.5], dtype=torch.float32, device=sampler.device),
            torch.tensor([2], dtype=torch.long, device=sampler.device),
        ),
    )

    reset_sample = sampler.reset(torch.tensor([0]), strategy=sampler_mod.SamplingStrategy.FailureWeighted)
    sampler.current_times[0] = 0.95
    sampler._global_step = 1
    sampler.record_failures(torch.tensor([0]))

    assert torch.equal(reset_sample.target_bin_indices, torch.tensor([2], dtype=torch.long))
    assert torch.allclose(reset_sample.times, torch.tensor([0.5], dtype=torch.float32))
    assert torch.equal(sampler.bin_sample_counts[0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(sampler.bin_fail_counts[0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.001]))


def test_sampler_records_failures_for_non_failure_weighted_reset(tmp_path: Path, monkeypatch) -> None:
    motion_file = tmp_path / "anchor_motion.npz"
    _write_anchor_sampler_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()
    sampler = sampler_mod.Sampler(
        num_envs=1,
        motion_lib=motion_lib,
        dt=0.05,
        anchor_body_index=0,
        adaptive_alpha=1.0,
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
            torch.tensor([2], dtype=torch.long, device=sampler.device),
        ),
    )

    sampler.reset(torch.tensor([0]), strategy=sampler_mod.SamplingStrategy.Random)
    sampler.current_times[0] = 0.95
    sampler._global_step = 1
    sampler.record_failures(torch.tensor([0]))

    assert torch.equal(sampler.bin_sample_counts[0], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(sampler.bin_fail_counts[0], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


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


def test_anchor_source_requires_at_least_one_motion_with_reset_anchor(tmp_path: Path) -> None:
    motion_file = tmp_path / "empty_anchor.npz"
    _write_empty_anchor_motion_file(motion_file)
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(RuntimeError, match="at least one motion clip with an eligible reset anchor"):
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


@pytest.mark.parametrize(
    ("kwarg_name", "invalid_value"),
    [
        ("weight_fail", -0.1),
        ("weight_novel", float("inf")),
        ("cap_beta", -0.1),
        ("adaptive_uniform_ratio", -0.1),
        ("adaptive_alpha", 1.1),
        ("adaptive_kernel_size", 0),
        ("adaptive_lambda", -0.1),
        ("motion_sampling_warmup_s", -0.1),
        ("motion_sampling_ramp_s", -0.1),
    ],
)
def test_sampler_rejects_invalid_mosaic_sampler_knobs(
    tmp_path: Path,
    kwarg_name: str,
    invalid_value: float,
) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(ValueError, match=kwarg_name):
        sampler_mod.Sampler(
            num_envs=1,
            motion_lib=motion_lib,
            dt=0.05,
            anchor_body_index=0,
            **{kwarg_name: invalid_value},
            device=torch.device("cpu"),
        )


def test_sampler_rejects_invalid_motion_sampling_schedule(tmp_path: Path) -> None:
    motion_file = tmp_path / "segmented_motion.npz"
    _write_motion_file(
        motion_file,
        segment_start_times=np.asarray([0.0, 0.5], dtype=np.float32),
        segment_end_times=np.asarray([0.5, 1.0], dtype=np.float32),
        segment_types=np.asarray([SEGMENT_TYPE_TIME_BIN, SEGMENT_TYPE_TIME_BIN], dtype=np.int64),
    )
    motion_lib = MotionLib([motion_file])
    sampler_mod = _load_sampler_module()

    with pytest.raises(ValueError, match="motion_sampling_schedule"):
        sampler_mod.Sampler(
            num_envs=1,
            motion_lib=motion_lib,
            dt=0.05,
            anchor_body_index=0,
            motion_sampling_schedule="step",
            device=torch.device("cpu"),
        )

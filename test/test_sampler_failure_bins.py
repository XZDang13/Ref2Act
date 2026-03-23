from pathlib import Path
import importlib
import sys
import types

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ref2Act.motion_lib import MotionLib
from Ref2Act.motion_segments import SEGMENT_TYPE_AIR_MERGE, SEGMENT_TYPE_TIME_BIN


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
        sys.modules.pop("Ref2Act.sampler", None)
        return importlib.import_module("Ref2Act.sampler")
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
        device=torch.device("cpu"),
    )

    sampler.bin_fail_counts[0][:] = torch.tensor([0.0, 25.0, 0.0])
    sampler.bin_sample_counts[0][:] = torch.tensor([1.0, 1.0, 1.0])
    motion_ids = torch.zeros(256, dtype=torch.long)
    times = sampler.sample_times_for_motion_ids(
        motion_ids,
        strategy=sampler_mod.SamplingStrategy.FailureWeighted,
        min_weight=0.0,
    )

    assert torch.all(times >= 0.2)
    assert torch.all(times < 0.8)


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

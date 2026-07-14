import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib, MotionSampler, SamplingStrategy, SegmentSource


def _write_motion(directory: Path, anchors: list[tuple[int, float]] | None = None) -> Path:
    directory.mkdir()
    frames = 10
    quat = np.zeros((frames, 1, 4), dtype=np.float32)
    quat[..., 3] = 1.0
    np.savez(
        directory / "final_motion.npz",
        fps=np.asarray(10.0, dtype=np.float32),
        robot=np.asarray("g1"),
        joint_names=np.asarray(["joint"]),
        body_names=np.asarray(["pelvis"]),
        joint_pos=np.zeros((frames, 1), dtype=np.float32),
        joint_vel=np.zeros((frames, 1), dtype=np.float32),
        body_pos_w=np.zeros((frames, 1, 3), dtype=np.float32),
        body_quat_xyzw=quat,
        body_lin_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    )
    if anchors is not None:
        payload = {"enabled": True, "anchors": [{"frame": f, "time_s": t} for f, t in anchors]}
        (directory / "reset_anchors.json").write_text(json.dumps(payload))
    return directory


def test_time_mode_builds_fixed_bins_only_from_bin_size(tmp_path: Path) -> None:
    lib = MotionLib(_write_motion(tmp_path / "motion", anchors=[(2, 0.2), (7, 0.7)]))
    sampler = MotionSampler(2, lib, 0.02, bin_size=0.3, segment_source=SegmentSource.Time)
    assert torch.allclose(sampler.bin_start_times[0], torch.tensor([0.0, 0.3, 0.6, 0.9]))
    assert torch.allclose(sampler.bin_reset_times[0], sampler.bin_start_times[0])


def test_anchor_mode_resets_only_at_json_anchor_times(tmp_path: Path) -> None:
    lib = MotionLib(_write_motion(tmp_path / "motion", anchors=[(2, 0.2), (7, 0.7)]))
    sampler = MotionSampler(64, lib, 0.02, bin_size=0.3, segment_source=SegmentSource.Anchor)
    sample = sampler.reset(strategy=SamplingStrategy.Random)
    assert all(any(value == pytest.approx(expected) for expected in (0.2, 0.7)) for value in sample.times.tolist())
    assert not torch.any(sample.times == 0.0)


def test_anchor_mode_rejects_motion_without_enabled_sidecar(tmp_path: Path) -> None:
    lib = MotionLib(_write_motion(tmp_path / "motion"))
    with pytest.raises(RuntimeError, match="reset_anchors.json"):
        MotionSampler(1, lib, 0.02, segment_source=SegmentSource.Anchor)

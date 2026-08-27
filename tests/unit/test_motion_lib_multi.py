import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ref2act.motion import MotionLib


def _write_motion(
    directory: Path,
    *,
    offset: float = 0.0,
    frames: int = 4,
    fps: float = 10.0,
    anchors: list[tuple[int, float]] | None = None,
) -> Path:
    directory.mkdir()
    joint_pos = np.arange(frames, dtype=np.float32)[:, None] + offset
    quat = np.zeros((frames, 1, 4), dtype=np.float32)
    quat[..., 3] = 1.0
    np.savez(
        directory / "final_motion.npz",
        fps=np.asarray(fps, dtype=np.float32),
        robot=np.asarray("g1"),
        joint_names=np.asarray(["joint"]),
        body_names=np.asarray(["pelvis"]),
        joint_pos=joint_pos,
        joint_vel=np.ones_like(joint_pos),
        body_pos_w=np.repeat(joint_pos[:, None], 3, axis=2),
        body_quat_xyzw=quat,
        body_lin_vel_w=np.ones((frames, 1, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 1, 3), dtype=np.float32),
    )
    if anchors is not None:
        (directory / "reset_anchors.json").write_text(
            json.dumps({"enabled": True, "anchors": [{"frame": f, "time_s": t} for f, t in anchors]})
        )
    return directory


def test_accepts_directory_direct_file_and_explicit_multi_motion(tmp_path: Path) -> None:
    first = _write_motion(tmp_path / "first", offset=0.0)
    second = _write_motion(tmp_path / "second", offset=10.0)
    lib = MotionLib([first, second / "final_motion.npz"])
    sample = lib.sample_motion(torch.tensor([0, 1]), torch.tensor([0.1, 0.1]))
    assert lib.num_motions == 2
    assert torch.allclose(sample["joint_pos"][:, 0], torch.tensor([1.0, 11.0]))


def test_applies_position_offsets_per_selected_motion(tmp_path: Path) -> None:
    first = _write_motion(tmp_path / "first")
    second = _write_motion(tmp_path / "second", offset=10.0)
    lib = MotionLib([first, second])
    offsets = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    sample = lib.sample_motion(torch.tensor([0, 1]), torch.zeros(2), position_offsets=offsets)
    assert torch.allclose(sample["body_positions"][:, 0], torch.tensor([[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]]))


def test_rejects_arbitrary_npz_and_dataset_root(tmp_path: Path) -> None:
    arbitrary = tmp_path / "motion.npz"
    arbitrary.touch()
    with pytest.raises(ValueError, match="final_motion.npz"):
        MotionLib(arbitrary)
    with pytest.raises(FileNotFoundError, match="final_motion.npz"):
        MotionLib(tmp_path)


def test_rejects_legacy_or_incomplete_npz(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir()
    np.savez(directory / "final_motion.npz", body_quat_w=np.zeros((2, 1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="missing required fields"):
        MotionLib(directory)


def test_rejects_non_finite_data_and_multi_motion_topology_mismatch(tmp_path: Path) -> None:
    non_finite = _write_motion(tmp_path / "non_finite")
    with np.load(non_finite / "final_motion.npz") as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["joint_vel"] = payload["joint_vel"].copy()
    payload["joint_vel"][0, 0] = np.nan
    np.savez(non_finite / "final_motion.npz", **payload)
    with pytest.raises(ValueError, match="NaN or inf"):
        MotionLib(non_finite)

    first = _write_motion(tmp_path / "first")
    second = _write_motion(tmp_path / "second")
    with np.load(second / "final_motion.npz") as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["joint_names"] = np.asarray(["different_joint"])
    np.savez(second / "final_motion.npz", **payload)
    with pytest.raises(ValueError, match="Joint names do not match"):
        MotionLib([first, second])


def test_validates_anchor_order_range_and_time(tmp_path: Path) -> None:
    directory = _write_motion(tmp_path / "motion", anchors=[(1, 0.1), (2, 0.25)])
    with pytest.raises(ValueError, match="inconsistent"):
        MotionLib(directory)
    (directory / "reset_anchors.json").write_text(
        json.dumps({"enabled": True, "anchors": [{"frame": 2, "time_s": 0.2}, {"frame": 1, "time_s": 0.1}]})
    )
    with pytest.raises(ValueError, match="sorted"):
        MotionLib(directory)


def test_disabled_anchor_sidecar_means_no_anchors(tmp_path: Path) -> None:
    directory = _write_motion(tmp_path / "motion")
    (directory / "reset_anchors.json").write_text(json.dumps({"enabled": False, "anchors": [{"frame": 0, "time_s": 0.0}]}))
    lib = MotionLib(directory)
    assert lib.clips[0].anchor_times is None


def test_packed_cache_invalidates_when_sidecar_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("REF2ACT_MOTION_PACK_CACHE_DIR", str(cache_dir))
    directory = _write_motion(tmp_path / "motion", anchors=[(1, 0.1)])
    first = MotionLib(directory, compact_after_packing=True)
    assert first.packed_cache_path is not None
    assert first.packed_cache_path.is_file()
    cache_files_before = set(cache_dir.iterdir())
    sidecar = directory / "reset_anchors.json"
    sidecar.write_text(json.dumps({"enabled": True, "anchors": [{"frame": 2, "time_s": 0.2}]}))
    stat = sidecar.stat()
    os.utime(sidecar, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = MotionLib(directory, compact_after_packing=True)
    assert set(cache_dir.iterdir()) != cache_files_before
    assert first.clips[0].anchor_times.item() == pytest.approx(0.1)
    assert second.clips[0].anchor_times.item() == pytest.approx(0.2)

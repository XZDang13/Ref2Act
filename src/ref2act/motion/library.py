from collections.abc import Sequence
import gc
import hashlib
import json
from math import ceil
import os
from os import PathLike
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from dataclasses import dataclass, field

from ref2act.common.utils import compute_frame_blend_from_fps, interpolate, slerp

MotionFileInput = str | PathLike[str] | Sequence[str | PathLike[str]]
_MOTION_LOAD_PROGRESS_STEPS = 10
_MOTION_LOAD_PROGRESS_FILES = int(os.getenv("REF2ACT_MOTION_LOAD_PROGRESS_FILES", "100"))
_MOTION_LOAD_PROGRESS_SECONDS = float(os.getenv("REF2ACT_MOTION_LOAD_PROGRESS_SECONDS", "5.0"))
_MOTION_SLOW_CLIP_SECONDS = float(os.getenv("REF2ACT_MOTION_SLOW_CLIP_SECONDS", "2.0"))
_MOTION_PACK_CACHE_VERSION = 3


def _progress_interval(total: int) -> int:
    if total <= 0:
        return 1
    if _MOTION_LOAD_PROGRESS_FILES > 0:
        return max(1, min(total, _MOTION_LOAD_PROGRESS_FILES))
    return max(1, min(500, ceil(total / _MOTION_LOAD_PROGRESS_STEPS)))


def _motion_file_group(motion_file: str | PathLike[str]) -> str:
    path = Path(motion_file)
    parts = path.parts
    if "mocap_data" in parts:
        mocap_index = parts.index("mocap_data")
        if mocap_index + 1 < len(parts):
            return parts[mocap_index + 1]
    parent_name = path.parent.name
    return parent_name or "."


def _validate_anchor_selection_arrays(
    frame_indices: np.ndarray,
    times: np.ndarray,
    *,
    num_frames: int,
    duration: float,
    atol: float = 1.0e-5,
) -> None:
    resolved_frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    resolved_times = np.asarray(times, dtype=np.float64).reshape(-1)
    if resolved_frame_indices.shape != resolved_times.shape:
        raise ValueError("anchor_frame_indices and anchor_times must have the same shape.")
    if np.any(resolved_frame_indices[1:] < resolved_frame_indices[:-1]):
        raise ValueError("anchor_frame_indices must be sorted in non-decreasing order.")
    if np.any(resolved_times[1:] + atol < resolved_times[:-1]):
        raise ValueError("anchor_times must be sorted in non-decreasing order.")
    if np.any(resolved_frame_indices < 0) or np.any(resolved_frame_indices >= num_frames):
        raise ValueError("anchor_frame_indices contain values outside the clip frame range.")
    if np.any(resolved_times < -atol) or np.any(resolved_times > duration + atol):
        raise ValueError("anchor_times contain values outside the clip duration.")


@dataclass
class MotionClip:
    _FRAME_TENSOR_FIELDS = frozenset(
        (
            "joint_pos",
            "joint_vel",
            "body_positions",
            "body_quaternions",
            "body_linear_velocities",
            "body_angular_velocities",
        )
    )

    motion_id: int
    name: str
    source: str
    fps: float
    dt: float
    num_frames: int
    duration: float
    joint_pos: torch.Tensor | None
    joint_vel: torch.Tensor | None
    body_positions: torch.Tensor | None
    body_quaternions: torch.Tensor | None
    body_linear_velocities: torch.Tensor | None
    body_angular_velocities: torch.Tensor | None
    anchor_frame_indices: torch.Tensor | None = None
    anchor_times: torch.Tensor | None = None
    joint_names: list[str] = field(default_factory=list)
    body_names: list[str] = field(default_factory=list)

    def __getattribute__(self, name: str):
        value = object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_FRAME_TENSOR_FIELDS") and value is None:
            raise RuntimeError(
                f"Motion clip frame tensor '{name}' is unavailable because its MotionLib was loaded "
                "with compact_after_packing=True. Use MotionLib.sample_motion(...) instead."
            )
        return value

    def clear_frame_tensors(self) -> None:
        for field_name in self._FRAME_TENSOR_FIELDS:
            object.__setattr__(self, field_name, None)

    @property
    def has_anchor_segments(self) -> bool:
        if self.anchor_frame_indices is None or self.anchor_times is None:
            return False
        return int(self.anchor_frame_indices.numel()) > 0

    @property
    def num_anchor_segments(self) -> int:
        if self.anchor_frame_indices is None:
            return 0
        return int(self.anchor_frame_indices.shape[0])


class MotionLib:
    def __init__(
        self,
        motion_files: MotionFileInput,
        device: torch.device = torch.device("cpu"),
        *,
        compact_after_packing: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.compact_after_packing = bool(compact_after_packing)
        self._frame_load_device = (
            torch.device("cpu")
            if self.compact_after_packing and self.device.type == "cuda"
            else self.device
        )
        self.motion_files = self._normalize_motion_files(motion_files)
        load_start_time = perf_counter()
        total_motion_files = len(self.motion_files)
        print(
            "loading motion data: "
            f"{total_motion_files} clip(s), target_device={self.device}, "
            f"frame_load_device={self._frame_load_device}, compact_after_packing={self.compact_after_packing}",
            flush=True,
        )
        packed_cache_path = self._packed_cache_path()
        if self.compact_after_packing and self._try_load_packed_cache(packed_cache_path, load_start_time):
            return

        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
            print("disabled Python GC during motion bulk load", flush=True)
        progress_interval = _progress_interval(total_motion_files)
        self.clips = []
        loaded_frames = 0
        last_progress_time = load_start_time
        current_group = None
        try:
            for motion_id, motion_file in enumerate(self.motion_files):
                motion_group = _motion_file_group(motion_file)
                loaded_count = motion_id + 1
                should_log_start = (
                    total_motion_files > 8
                    and (
                        loaded_count == 1
                        or loaded_count == total_motion_files
                        or loaded_count % progress_interval == 1
                    )
                )
                if should_log_start:
                    print(
                        "loading motion file: "
                        f"index={loaded_count}/{total_motion_files}, group={motion_group}, path={motion_file}",
                        flush=True,
                    )
                if motion_group != current_group:
                    current_group = motion_group
                    print(
                        "loading motion group: "
                        f"group={motion_group}, index={loaded_count}/{total_motion_files}, path={motion_file}",
                        flush=True,
                    )
                clip_start_time = perf_counter()
                clip = self._load_clip(motion_id, motion_file)
                clip_elapsed = perf_counter() - clip_start_time
                if _MOTION_SLOW_CLIP_SECONDS > 0.0 and clip_elapsed >= _MOTION_SLOW_CLIP_SECONDS:
                    try:
                        size_mb = Path(motion_file).stat().st_size / 1024**2
                    except OSError:
                        size_mb = 0.0
                    print(
                        "slow motion file load: "
                        f"index={loaded_count}/{total_motion_files}, elapsed={clip_elapsed:.1f}s, "
                        f"size_mb={size_mb:.1f}, frames={clip.num_frames}, path={motion_file}",
                        flush=True,
                    )
                self.clips.append(clip)
                loaded_frames += int(clip.num_frames)
                now = perf_counter()
                should_log_by_count = loaded_count == total_motion_files or loaded_count % progress_interval == 0
                should_log_by_time = (
                    _MOTION_LOAD_PROGRESS_SECONDS > 0.0
                    and now - last_progress_time >= _MOTION_LOAD_PROGRESS_SECONDS
                )
                if (
                    total_motion_files > 8
                    and (should_log_by_count or should_log_by_time)
                ):
                    print(
                        "loading motion data: "
                        f"{loaded_count}/{total_motion_files} clip(s), "
                        f"frames={loaded_frames}, elapsed={now - load_start_time:.1f}s, "
                        f"group={motion_group}, last_path={motion_file}",
                        flush=True,
                    )
                    last_progress_time = now
            if not self.clips:
                raise ValueError("No motion clips were loaded.")
            load_elapsed = perf_counter() - load_start_time

            self._finalize_clip_metadata(validate_frame_shapes=True)
            self._packed_sampling_tensors: dict[str, torch.Tensor] = {}
            self._packed_frame_offsets: torch.Tensor | None = None
            pack_start_time = perf_counter()
            print("packing motion sampling tensors", flush=True)
            self._packed_sampling_enabled = self._build_packed_sampling_tensors()
            pack_elapsed = perf_counter() - pack_start_time
            print(
                f"packed motion sampling tensors: enabled={self._packed_sampling_enabled}, elapsed={pack_elapsed:.1f}s",
                flush=True,
            )
            if self.compact_after_packing:
                if not self._packed_sampling_enabled:
                    raise RuntimeError(
                        "compact_after_packing=True requires compatible clips so packed sampling can be enabled."
                    )
                self._save_packed_cache(packed_cache_path)
                compact_start_time = perf_counter()
                print("clearing per-clip frame tensors after packing", flush=True)
                self._clear_clip_frame_tensors()
                print(
                    f"cleared per-clip frame tensors after packing: elapsed={perf_counter() - compact_start_time:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            print(
                "motion bulk load failed: "
                f"loaded_clips={len(self.clips)}/{total_motion_files}, "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        finally:
            if gc_was_enabled:
                gc.enable()
                print("restored Python GC after motion bulk load", flush=True)
        duration_values = [clip.duration for clip in self.clips]
        if self.num_motions <= 8:
            duration_summary = f"durations={[round(duration, 3) for duration in duration_values]} s"
        else:
            total_duration = sum(duration_values)
            duration_summary = (
                f"duration_s=min:{min(duration_values):.3f} "
                f"mean:{total_duration / self.num_motions:.3f} "
                f"max:{max(duration_values):.3f} total:{total_duration:.3f}"
            )
        print(
            f"motion data loaded: {self.num_motions} clip(s), "
            f"frames={sum(clip.num_frames for clip in self.clips)}, "
            f"{duration_summary}, load_elapsed={load_elapsed:.1f}s, "
            f"total_elapsed={perf_counter() - load_start_time:.1f}s",
            flush=True,
        )

    def _finalize_clip_metadata(self, *, validate_frame_shapes: bool) -> None:
        if validate_frame_shapes:
            self._validate_clip_compatibility()
        else:
            self._validate_clip_names()

        self.joint_names = self.clips[0].joint_names
        self.body_names = self.clips[0].body_names
        self.num_motions = len(self.clips)
        self.motion_names = [clip.name for clip in self.clips]
        self.motion_durations = torch.tensor(
            [clip.duration for clip in self.clips], dtype=torch.float32, device=self.device
        )
        self.motion_num_frames = torch.tensor(
            [clip.num_frames for clip in self.clips], dtype=torch.long, device=self.device
        )
        self.motion_fps = torch.tensor(
            [clip.fps for clip in self.clips], dtype=torch.float32, device=self.device
        )
        self.motion_has_anchor_segments = torch.tensor(
            [clip.has_anchor_segments for clip in self.clips], dtype=torch.bool, device=self.device
        )
        self.motion_num_anchor_segments = torch.tensor(
            [clip.num_anchor_segments for clip in self.clips], dtype=torch.long, device=self.device
        )
        self.motion_anchor_frame_indices = [clip.anchor_frame_indices for clip in self.clips]
        self.motion_anchor_times = [clip.anchor_times for clip in self.clips]
        self.all_clips_have_anchor_segments = bool(torch.all(self.motion_has_anchor_segments).item())

    @staticmethod
    def _optional_tensor_to_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.detach().to(device="cpu")

    def _optional_tensor_to_device(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return torch.as_tensor(value).to(device=self.device)

    def _motion_file_stats(self) -> list[tuple[str, int, int]]:
        stats = []
        for motion_file in self.motion_files:
            path = Path(motion_file).expanduser().resolve()
            stat = path.stat()
            stats.append((str(path), int(stat.st_size), int(stat.st_mtime_ns)))
            anchor_path = path.with_name("reset_anchors.json")
            if anchor_path.exists():
                anchor_stat = anchor_path.stat()
                stats.append((str(anchor_path), int(anchor_stat.st_size), int(anchor_stat.st_mtime_ns)))
            else:
                stats.append((str(anchor_path), -1, -1))
        return stats

    def _packed_cache_path(self) -> Path | None:
        cache_dir = os.getenv("REF2ACT_MOTION_PACK_CACHE_DIR")
        if not cache_dir:
            return None

        try:
            stats = self._motion_file_stats()
        except OSError as exc:
            print(f"skipping motion pack cache: failed to stat motion files: {exc}", flush=True)
            return None

        digest = hashlib.sha256()
        digest.update(str(_MOTION_PACK_CACHE_VERSION).encode("utf-8"))
        for path, size, mtime_ns in stats:
            digest.update(path.encode("utf-8"))
            digest.update(str(size).encode("ascii"))
            digest.update(str(mtime_ns).encode("ascii"))
        return Path(cache_dir).expanduser().resolve() / f"motionlib_{digest.hexdigest()[:24]}.pt"

    @property
    def packed_cache_path(self) -> Path | None:
        """Resolved fingerprinted cache path for this ordered motion set."""

        return self._packed_cache_path()

    def _clip_cache_payload(self, clip: MotionClip) -> dict[str, object]:
        return {
            "motion_id": int(clip.motion_id),
            "name": str(clip.name),
            "source": str(clip.source),
            "fps": float(clip.fps),
            "dt": float(clip.dt),
            "num_frames": int(clip.num_frames),
            "duration": float(clip.duration),
            "joint_names": list(clip.joint_names),
            "body_names": list(clip.body_names),
            "anchor_frame_indices": self._optional_tensor_to_cpu(clip.anchor_frame_indices),
            "anchor_times": self._optional_tensor_to_cpu(clip.anchor_times),
        }

    def _clip_from_cache_payload(self, payload: dict[str, object]) -> MotionClip:
        return MotionClip(
            motion_id=int(payload["motion_id"]),
            name=str(payload["name"]),
            source=str(payload["source"]),
            fps=float(payload["fps"]),
            dt=float(payload["dt"]),
            num_frames=int(payload["num_frames"]),
            duration=float(payload["duration"]),
            joint_pos=None,
            joint_vel=None,
            body_positions=None,
            body_quaternions=None,
            body_linear_velocities=None,
            body_angular_velocities=None,
            anchor_frame_indices=self._optional_tensor_to_device(payload["anchor_frame_indices"]),
            anchor_times=self._optional_tensor_to_device(payload["anchor_times"]),
            joint_names=list(payload["joint_names"]),
            body_names=list(payload["body_names"]),
        )

    def _try_load_packed_cache(self, cache_path: Path | None, start_time: float) -> bool:
        if cache_path is None or not cache_path.exists():
            return False

        cache_start_time = perf_counter()
        print(f"loading packed motion cache: {cache_path}", flush=True)
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if int(payload.get("version", -1)) != _MOTION_PACK_CACHE_VERSION:
                print("packed motion cache version mismatch; rebuilding", flush=True)
                return False
            if payload.get("file_stats") != self._motion_file_stats():
                print("packed motion cache file stats mismatch; rebuilding", flush=True)
                return False

            self.clips = [self._clip_from_cache_payload(item) for item in payload["clips"]]
            if not self.clips:
                return False
            self._finalize_clip_metadata(validate_frame_shapes=False)
            self._packed_sampling_tensors = {
                str(name): torch.as_tensor(tensor).to(device=self.device)
                for name, tensor in payload["packed_sampling_tensors"].items()
            }
            self._packed_frame_offsets = torch.as_tensor(
                payload["packed_frame_offsets"], dtype=torch.long, device=self.device
            )
            self._packed_sampling_enabled = True
        except Exception as exc:
            print(f"failed to load packed motion cache; rebuilding: {exc}", flush=True)
            return False

        print(
            f"packed motion cache loaded: {self.num_motions} clip(s), "
            f"frames={int(self.motion_num_frames.sum().item())}, "
            f"cache_elapsed={perf_counter() - cache_start_time:.1f}s, "
            f"total_elapsed={perf_counter() - start_time:.1f}s",
            flush=True,
        )
        return True

    def _save_packed_cache(self, cache_path: Path | None) -> None:
        if cache_path is None:
            return

        cache_start_time = perf_counter()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        print(f"saving packed motion cache: {cache_path}", flush=True)
        try:
            payload = {
                "version": _MOTION_PACK_CACHE_VERSION,
                "file_stats": self._motion_file_stats(),
                "clips": [self._clip_cache_payload(clip) for clip in self.clips],
                "packed_sampling_tensors": {
                    name: tensor.detach().to(device="cpu")
                    for name, tensor in self._packed_sampling_tensors.items()
                },
                "packed_frame_offsets": self._packed_frame_offsets.detach().to(device="cpu"),
            }
            torch.save(payload, temporary_path)
            os.replace(temporary_path, cache_path)
        except Exception as exc:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"failed to save packed motion cache: {exc}", flush=True)
            return

        print(
            f"packed motion cache saved: elapsed={perf_counter() - cache_start_time:.1f}s",
            flush=True,
        )

    def _clear_clip_frame_tensors(self) -> None:
        for clip in self.clips:
            clip.clear_frame_tensors()
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize_motion_files(motion_files: MotionFileInput) -> list[str]:
        if isinstance(motion_files, (str, PathLike)):
            paths = [motion_files]
        else:
            paths = list(motion_files)
        if not paths:
            raise ValueError("motion_files must contain at least one motion clip path.")
        resolved_paths: list[str] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if path.is_dir():
                path = path / "final_motion.npz"
            elif path.name != "final_motion.npz":
                raise ValueError(
                    f"Motion input must be a directory containing final_motion.npz or a direct "
                    f"final_motion.npz path, got: {path}"
                )
            if not path.is_file():
                raise FileNotFoundError(f"Retargeter motion file was not found: {path}")
            resolved_paths.append(str(path.resolve()))
        return resolved_paths

    def _load_joint_names(self, motion_file: str) -> list[str]:
        with np.load(motion_file) as motion_data:
            return motion_data["joint_names"].tolist()

    def _load_body_names(self, motion_file: str) -> list[str]:
        with np.load(motion_file) as motion_data:
            return motion_data["body_names"].tolist()

    def _load_clip(self, motion_id: int, motion_file: str) -> MotionClip:
        with np.load(motion_file) as motion_data:
            required_keys = {
                "fps",
                "robot",
                "joint_names",
                "body_names",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_xyzw",
                "body_lin_vel_w",
                "body_ang_vel_w",
            }
            missing_keys = sorted(required_keys.difference(motion_data.files))
            if missing_keys:
                raise ValueError(
                    f"Retargeter motion {motion_file} is missing required fields: {missing_keys}. "
                    "Legacy Ref2Act NPZ files are not supported."
                )
            legacy_keys = {
                "body_quat_w",
                "anchor_selection_version",
                "anchor_frame_indices",
                "anchor_times",
                "segment_start_times",
                "segment_end_times",
                "segment_types",
            }
            present_legacy_keys = sorted(legacy_keys.intersection(motion_data.files))
            if present_legacy_keys:
                raise ValueError(
                    f"Retargeter motion {motion_file} contains unsupported legacy fields: {present_legacy_keys}"
                )
            fps = float(np.asarray(motion_data["fps"]).item())
            if not np.isfinite(fps) or fps <= 0.0:
                raise ValueError(f"Motion clip {motion_file} has a non-positive fps: {fps}")
            dt = 1.0 / fps
            joint_names = np.asarray(motion_data["joint_names"]).tolist()
            body_names = np.asarray(motion_data["body_names"]).tolist()
            robot_name = str(np.asarray(motion_data["robot"]).item())
            if not robot_name:
                raise ValueError(f"Retargeter motion {motion_file} has an empty robot field.")
            for field_name, names in (("joint_names", joint_names), ("body_names", body_names)):
                if not names or not all(isinstance(name, str) and name for name in names):
                    raise ValueError(f"Retargeter motion {motion_file} field {field_name} must contain names.")
                if len(set(names)) != len(names):
                    raise ValueError(f"Retargeter motion {motion_file} field {field_name} contains duplicates.")
            arrays = {
                "joint_pos": np.asarray(motion_data["joint_pos"]),
                "joint_vel": np.asarray(motion_data["joint_vel"]),
                "body_pos_w": np.asarray(motion_data["body_pos_w"]),
                "body_quat_xyzw": np.asarray(motion_data["body_quat_xyzw"]),
                "body_lin_vel_w": np.asarray(motion_data["body_lin_vel_w"]),
                "body_ang_vel_w": np.asarray(motion_data["body_ang_vel_w"]),
            }
            for key, array in arrays.items():
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"Retargeter motion {motion_file} field {key} contains NaN or inf values.")
            joint_pos_np = arrays["joint_pos"]
            if joint_pos_np.ndim != 2 or joint_pos_np.shape[1] != len(joint_names):
                raise ValueError("joint_pos must have shape [T, len(joint_names)].")
            num_frames = int(joint_pos_np.shape[0])
            if num_frames < 1:
                raise ValueError(f"Retargeter motion {motion_file} must contain at least one frame.")
            expected_shapes = {
                "joint_vel": (num_frames, len(joint_names)),
                "body_pos_w": (num_frames, len(body_names), 3),
                "body_quat_xyzw": (num_frames, len(body_names), 4),
                "body_lin_vel_w": (num_frames, len(body_names), 3),
                "body_ang_vel_w": (num_frames, len(body_names), 3),
            }
            for key, expected_shape in expected_shapes.items():
                if arrays[key].shape != expected_shape:
                    raise ValueError(
                        f"Retargeter motion {motion_file} field {key} must have shape {expected_shape}, "
                        f"got {arrays[key].shape}."
                    )
            quat_norms = np.linalg.norm(arrays["body_quat_xyzw"], axis=-1)
            if not np.allclose(quat_norms, 1.0, atol=1.0e-3):
                raise ValueError(f"Retargeter motion {motion_file} contains non-unit body_quat_xyzw values.")

            joint_pos = torch.as_tensor(joint_pos_np, dtype=torch.float32, device=self._frame_load_device)
            duration = dt * num_frames
            anchor_frame_indices = None
            anchor_times = None

            anchor_path = Path(motion_file).with_name("reset_anchors.json")
            if anchor_path.exists():
                try:
                    anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Failed to read Retargeter anchor file {anchor_path}: {exc}") from exc
                if not isinstance(anchor_payload, dict):
                    raise ValueError(f"Retargeter anchor file {anchor_path} must contain a JSON object.")
                enabled = anchor_payload.get("enabled", False)
                if not isinstance(enabled, bool):
                    raise ValueError(f"Retargeter anchor file {anchor_path} field enabled must be boolean.")
                if enabled:
                    anchors = anchor_payload.get("anchors")
                    if not isinstance(anchors, list) or not anchors:
                        raise ValueError(f"Enabled Retargeter anchor file {anchor_path} must contain anchors.")
                    try:
                        anchor_frame_indices_np = np.asarray([item["frame"] for item in anchors], dtype=np.int64)
                        anchor_times_np = np.asarray([item["time_s"] for item in anchors], dtype=np.float32)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid anchor entry in {anchor_path}: {exc}") from exc
                    _validate_anchor_selection_arrays(
                        anchor_frame_indices_np,
                        anchor_times_np,
                        num_frames=num_frames,
                        duration=duration,
                    )
                    if np.any(anchor_frame_indices_np[1:] <= anchor_frame_indices_np[:-1]):
                        raise ValueError(f"Anchor frames in {anchor_path} must be strictly increasing and unique.")
                    expected_times = anchor_frame_indices_np.astype(np.float64) / fps
                    if not np.allclose(anchor_times_np, expected_times, atol=1.0e-5):
                        raise ValueError(f"Anchor frame/time values in {anchor_path} are inconsistent with fps={fps}.")
                    anchor_frame_indices = torch.as_tensor(
                        anchor_frame_indices_np, dtype=torch.long, device=self.device
                    )
                    anchor_times = torch.as_tensor(anchor_times_np, dtype=torch.float32, device=self.device)

            return MotionClip(
                motion_id=motion_id,
                name=Path(motion_file).parent.name,
                source=motion_file,
                joint_names=joint_names,
                body_names=body_names,
                fps=fps,
                dt=dt,
                num_frames=num_frames,
                duration=duration,
                joint_pos=joint_pos,
                joint_vel=torch.as_tensor(arrays["joint_vel"], dtype=torch.float32, device=self._frame_load_device),
                body_positions=torch.as_tensor(
                    arrays["body_pos_w"], dtype=torch.float32, device=self._frame_load_device
                ),
                body_quaternions=torch.as_tensor(
                    arrays["body_quat_xyzw"], dtype=torch.float32, device=self._frame_load_device
                ),
                body_linear_velocities=torch.as_tensor(
                    arrays["body_lin_vel_w"], dtype=torch.float32, device=self._frame_load_device
                ),
                body_angular_velocities=torch.as_tensor(
                    arrays["body_ang_vel_w"], dtype=torch.float32, device=self._frame_load_device
                ),
                anchor_frame_indices=anchor_frame_indices,
                anchor_times=anchor_times,
            )

    def _validate_clip_compatibility(self) -> None:
        self._validate_clip_names()
        reference_joint_dim = self.clips[0].joint_pos.shape[1:]
        reference_body_dim = self.clips[0].body_positions.shape[1:]

        for clip in self.clips[1:]:
            if clip.joint_pos.shape[1:] != reference_joint_dim:
                raise ValueError(f"Joint tensor shape mismatch across motion clips: {clip.source}")
            if clip.body_positions.shape[1:] != reference_body_dim:
                raise ValueError(f"Body tensor shape mismatch across motion clips: {clip.source}")

    def _validate_clip_names(self) -> None:
        reference_joint_names = self.clips[0].joint_names
        reference_body_names = self.clips[0].body_names

        for clip in self.clips[1:]:
            if clip.joint_names != reference_joint_names:
                raise ValueError(f"Joint names do not match across motion clips: {clip.source}")
            if clip.body_names != reference_body_names:
                raise ValueError(f"Body names do not match across motion clips: {clip.source}")

    def _require_single_motion(self, attribute_name: str) -> MotionClip:
        if self.num_motions != 1:
            raise RuntimeError(f"{attribute_name} is only defined for a single loaded motion clip.")
        return self.clips[0]

    @property
    def fps(self) -> float:
        return self._require_single_motion("fps").fps

    @property
    def dt(self) -> float:
        return self._require_single_motion("dt").dt

    @property
    def num_frames(self) -> int:
        return self._require_single_motion("num_frames").num_frames

    @property
    def duration(self) -> float:
        return self._require_single_motion("duration").duration

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._require_single_motion("joint_pos").joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._require_single_motion("joint_vel").joint_vel

    @property
    def body_positions(self) -> torch.Tensor:
        return self._require_single_motion("body_positions").body_positions

    @property
    def body_quaternions(self) -> torch.Tensor:
        return self._require_single_motion("body_quaternions").body_quaternions

    @property
    def body_linear_velocities(self) -> torch.Tensor:
        return self._require_single_motion("body_linear_velocities").body_linear_velocities

    @property
    def body_angular_velocities(self) -> torch.Tensor:
        return self._require_single_motion("body_angular_velocities").body_angular_velocities

    def get_clip(self, motion_id: int) -> MotionClip:
        motion_id = int(motion_id)
        if motion_id < 0 or motion_id >= self.num_motions:
            raise IndexError(f"motion_id out of range: {motion_id}")
        return self.clips[motion_id]

    def get_duration(self, motion_ids: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if motion_ids.numel() == 0:
            return torch.empty_like(motion_ids, dtype=torch.float32)
        if validate:
            self._validate_motion_ids(motion_ids)
        return self.motion_durations[motion_ids]

    def _validate_motion_ids(self, motion_ids: torch.Tensor) -> None:
        if torch.any(motion_ids < 0) or torch.any(motion_ids >= self.num_motions):
            raise IndexError("motion_ids contain values outside the loaded motion clip range.")

    def _build_packed_sampling_tensors(self) -> bool:
        sample_fields = {
            "joint_pos": "joint_pos",
            "joint_vel": "joint_vel",
            "body_positions": "body_positions",
            "body_quaternions": "body_quaternions",
            "body_linear_velocities": "body_linear_velocities",
            "body_angular_velocities": "body_angular_velocities",
        }

        reference_device = self.clips[0].joint_pos.device
        packed_tensors: dict[str, torch.Tensor] = {}
        frame_offsets: list[int] = []
        next_frame_offset = 0
        for clip in self.clips:
            frame_offsets.append(next_frame_offset)
            next_frame_offset += clip.num_frames

        for output_name, clip_attribute in sample_fields.items():
            clip_tensors = [getattr(clip, clip_attribute) for clip in self.clips]
            reference_shape = clip_tensors[0].shape[1:]
            incompatible_clip = any(
                tensor.device != reference_device or tensor.shape[1:] != reference_shape
                for tensor in clip_tensors
            )
            if incompatible_clip:
                self._packed_sampling_tensors = {}
                self._packed_frame_offsets = None
                return False

            field_start_time = perf_counter()
            print(f"packing motion sampling tensor: {output_name}", flush=True)
            packed_tensors[output_name] = torch.cat(clip_tensors, dim=0).to(device=self.device)
            print(
                f"packed motion sampling tensor: {output_name}, "
                f"shape={tuple(packed_tensors[output_name].shape)}, "
                f"elapsed={perf_counter() - field_start_time:.1f}s",
                flush=True,
            )

        self._packed_sampling_tensors = packed_tensors
        self._packed_frame_offsets = torch.as_tensor(frame_offsets, dtype=torch.long, device=self.device)
        return True

    def _sample_clip(
        self,
        clip: MotionClip,
        times: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        index_0, index_1, blend = compute_frame_blend_from_fps(times, clip.fps, clip.num_frames)
        index_0 = index_0.to(device=self.device)
        index_1 = index_1.to(device=self.device)
        blend = blend.to(device=self.device, dtype=torch.float32)

        joint_pos = interpolate(clip.joint_pos, b=clip.joint_pos, blend=blend, start=index_0, end=index_1)
        joint_vel = interpolate(clip.joint_vel, b=clip.joint_vel, blend=blend, start=index_0, end=index_1)
        body_positions = interpolate(
            clip.body_positions, b=clip.body_positions, blend=blend, start=index_0, end=index_1
        )

        if position_offsets is not None:
            if position_offsets.ndim == 2:
                position_offsets = position_offsets.unsqueeze(1)
            body_positions = body_positions + position_offsets

        body_quaternions = slerp(
            clip.body_quaternions, q1=clip.body_quaternions, blend=blend, start=index_0, end=index_1
        )
        body_linear_velocities = interpolate(
            clip.body_linear_velocities,
            b=clip.body_linear_velocities,
            blend=blend,
            start=index_0,
            end=index_1,
        )
        body_angular_velocities = interpolate(
            clip.body_angular_velocities,
            b=clip.body_angular_velocities,
            blend=blend,
            start=index_0,
            end=index_1,
        )

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_positions": body_positions,
            "body_quaternions": body_quaternions,
            "body_linear_velocities": body_linear_velocities,
            "body_angular_velocities": body_angular_velocities,
        }

    def _sample_motion_grouped(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = motion_ids.shape[0]
        template_clip = self.clips[0]
        motions = {
            "joint_pos": torch.empty(
                (batch_size, *template_clip.joint_pos.shape[1:]), dtype=torch.float32, device=self.device
            ),
            "joint_vel": torch.empty(
                (batch_size, *template_clip.joint_vel.shape[1:]), dtype=torch.float32, device=self.device
            ),
            "body_positions": torch.empty(
                (batch_size, *template_clip.body_positions.shape[1:]), dtype=torch.float32, device=self.device
            ),
            "body_quaternions": torch.empty(
                (batch_size, *template_clip.body_quaternions.shape[1:]), dtype=torch.float32, device=self.device
            ),
            "body_linear_velocities": torch.empty(
                (batch_size, *template_clip.body_linear_velocities.shape[1:]),
                dtype=torch.float32,
                device=self.device,
            ),
            "body_angular_velocities": torch.empty(
                (batch_size, *template_clip.body_angular_velocities.shape[1:]),
                dtype=torch.float32,
                device=self.device,
            ),
        }

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            group_mask = motion_ids == motion_id
            group_position_offsets = position_offsets[group_mask] if position_offsets is not None else None
            sampled_clip = self._sample_clip(
                self.clips[motion_id],
                times[group_mask],
                position_offsets=group_position_offsets,
            )
            for key, value in sampled_clip.items():
                motions[key][group_mask] = value

        return motions

    @staticmethod
    def _interpolate_packed_frames(
        values: torch.Tensor,
        index_0: torch.Tensor,
        index_1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        return interpolate(values[index_0], b=values[index_1], blend=blend)

    def _sample_motion_packed(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        frame = times * self.motion_fps[motion_ids]
        max_frame = self.motion_num_frames[motion_ids].to(dtype=torch.float32) - 1.0
        frame = torch.minimum(torch.clamp(frame, min=0.0), max_frame)
        index_0 = torch.floor(frame).long()
        index_1 = torch.minimum(index_0 + 1, self.motion_num_frames[motion_ids] - 1)
        blend = frame - index_0.to(dtype=frame.dtype)
        if self._packed_frame_offsets is None:
            raise RuntimeError("Packed sampling is enabled but packed frame offsets are unavailable.")
        frame_offsets = self._packed_frame_offsets[motion_ids]
        packed_index_0 = frame_offsets + index_0
        packed_index_1 = frame_offsets + index_1

        packed = self._packed_sampling_tensors
        joint_pos = self._interpolate_packed_frames(packed["joint_pos"], packed_index_0, packed_index_1, blend)
        joint_vel = self._interpolate_packed_frames(packed["joint_vel"], packed_index_0, packed_index_1, blend)
        body_positions = self._interpolate_packed_frames(
            packed["body_positions"],
            packed_index_0,
            packed_index_1,
            blend,
        )

        if position_offsets is not None:
            if position_offsets.ndim == 2:
                position_offsets = position_offsets.unsqueeze(1)
            body_positions = body_positions + position_offsets

        body_quaternions = slerp(
            packed["body_quaternions"][packed_index_0],
            q1=packed["body_quaternions"][packed_index_1],
            blend=blend,
        )
        body_linear_velocities = self._interpolate_packed_frames(
            packed["body_linear_velocities"],
            packed_index_0,
            packed_index_1,
            blend,
        )
        body_angular_velocities = self._interpolate_packed_frames(
            packed["body_angular_velocities"],
            packed_index_0,
            packed_index_1,
            blend,
        )

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_positions": body_positions,
            "body_quaternions": body_quaternions,
            "body_linear_velocities": body_linear_velocities,
            "body_angular_velocities": body_angular_velocities,
        }

    def sample_motion(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
        *,
        validate: bool = True,
    ) -> dict[str, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1)
        if motion_ids.shape != times.shape:
            raise ValueError("motion_ids and times must have the same batch shape.")
        if position_offsets is not None:
            position_offsets = torch.as_tensor(position_offsets, dtype=torch.float32, device=self.device)
            if position_offsets.shape[0] != times.shape[0]:
                raise ValueError("position_offsets must have the same batch size as motion_ids and times.")
        if validate:
            self._validate_motion_ids(motion_ids)

        if self._packed_sampling_enabled:
            return self._sample_motion_packed(
                motion_ids=motion_ids,
                times=times,
                position_offsets=position_offsets,
            )
        return self._sample_motion_grouped(
            motion_ids=motion_ids,
            times=times,
            position_offsets=position_offsets,
        )


class MotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    """

    def __init__(
        self,
        motion_file: MotionFileInput,
        device: torch.device | str = "cpu",
        render_scene: bool = False,
        motion_id: int = 0,
    ) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.
            render_scene: Whether the scene (space occupied by the skeleton during movement)
                is rendered instead of a reduced view of the skeleton.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        self.figure = None
        self.figure_axes = None
        self.render_scene = render_scene

        # load motions
        self.motion_lib = MotionLib(motion_files=motion_file, device=device)
        self.clip = self.motion_lib.get_clip(motion_id)

        self.num_frames = self.clip.num_frames
        self.current_frame = 0
        self.body_positions = self.clip.body_positions.cpu().numpy()
        self.body_colors = self._build_body_colors(self.motion_lib.body_names)

        print("\nBody")
        for i, name in enumerate(self.motion_lib.body_names):
            minimum = np.min(self.body_positions[:, i], axis=0).round(decimals=2)
            maximum = np.max(self.body_positions[:, i], axis=0).round(decimals=2)
            print(f"  |-- [{name}] minimum position: {minimum}, maximum position: {maximum}")

    @staticmethod
    def _build_body_colors(body_names: list[str]) -> list[str]:
        leg_keywords = ("hip", "knee", "ankle", "leg", "foot")
        arm_keywords = ("shoulder", "elbow", "hand", "arm", "wrist")
        body_keywords = ("pelvis", "torso", "chest", "spine", "waist")

        colors = []
        for name in body_names:
            lower = name.lower()
            if any(key in lower for key in leg_keywords):
                colors.append("tab:blue")
            elif any(key in lower for key in arm_keywords):
                colors.append("tab:orange")
            elif any(key in lower for key in body_keywords):
                colors.append("tab:green")
            else:
                colors.append("gray")
        return colors

    def drawing_callback(self, frame: int) -> None:
        """Drawing callback called each frame"""
        # get current motion frame
        # get data
        vertices = self.body_positions[self.current_frame]
        # draw skeleton state
        self.figure_axes.clear()
        self.figure_axes.scatter(*vertices.T, color=self.body_colors, depthshade=False)
        # adjust exes according to motion view
        # - scene
        if self.render_scene:
            # compute axes limits
            minimum = np.min(self.body_positions.reshape(-1, 3), axis=0)
            maximum = np.max(self.body_positions.reshape(-1, 3), axis=0)
            center = 0.5 * (maximum + minimum)
            diff = 0.75 * (maximum - minimum)
        # - skeleton
        else:
            # compute axes limits
            minimum = np.min(vertices, axis=0)
            maximum = np.max(vertices, axis=0)
            center = 0.5 * (maximum + minimum)
            diff = np.array([0.75 * np.max(maximum - minimum).item()] * 3)
        # scale view
        self.figure_axes.set_xlim((center[0] - diff[0], center[0] + diff[0]))
        self.figure_axes.set_ylim((center[1] - diff[1], center[1] + diff[1]))
        self.figure_axes.set_zlim((center[2] - diff[2], center[2] + diff[2]))
        self.figure_axes.set_box_aspect(aspect=diff / diff[0])
        # plot ground plane
        x, y = np.meshgrid([center[0] - diff[0], center[0] + diff[0]], [center[1] - diff[1], center[1] + diff[1]])
        self.figure_axes.plot_surface(x, y, np.zeros_like(x), color="green", alpha=0.2)
        # print metadata
        self.figure_axes.set_xlabel("X")
        self.figure_axes.set_ylabel("Y")
        self.figure_axes.set_zlabel("Z")
        self.figure_axes.set_title(f"frame: {self.current_frame}/{self.num_frames}")
        # increase frame counter
        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            self.current_frame = 0

    def show(self) -> None:
        """Show motion"""
        # create a 3D figure
        self.figure = plt.figure()
        self.figure_axes = self.figure.add_subplot(projection="3d")
        # matplotlib animation (the instance must live as long as the animation will run)
        self.animation = matplotlib.animation.FuncAnimation(
            fig=self.figure,
            func=self.drawing_callback,
            frames=self.num_frames,
            interval=1000 * self.clip.dt,
        )
        plt.show()

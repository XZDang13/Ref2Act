from collections.abc import Sequence
from os import PathLike
from pathlib import Path
import numpy as np
import torch
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from dataclasses import dataclass

from ref2act.common.utils import compute_frame_blend_from_fps, interpolate, slerp

from .segments import (
    ANCHOR_FRAME_LABEL_GREEN,
    ANCHOR_FRAME_LABEL_RED,
    ANCHOR_FRAME_LABEL_YELLOW,
    validate_segment_arrays,
)


MotionFileInput = str | PathLike[str] | Sequence[str | PathLike[str]]
_ANCHOR_SEGMENT_LABELS = (
    int(ANCHOR_FRAME_LABEL_RED),
    int(ANCHOR_FRAME_LABEL_YELLOW),
    int(ANCHOR_FRAME_LABEL_GREEN),
)


def _validate_anchor_segment_arrays(
    start_times: np.ndarray,
    end_times: np.ndarray,
    labels: np.ndarray,
    *,
    duration: float,
) -> None:
    validate_segment_arrays(start_times, end_times, duration=duration)
    resolved_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if resolved_labels.shape != np.asarray(start_times).reshape(-1).shape:
        raise ValueError("anchor_segment_labels must have the same shape as anchor_segment_start_times.")
    if np.any(~np.isin(resolved_labels, _ANCHOR_SEGMENT_LABELS)):
        raise ValueError("anchor_segment_labels contains unknown anchor label ids.")


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
    motion_id: int
    name: str
    source: str
    fps: float
    dt: float
    num_frames: int
    duration: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_positions: torch.Tensor
    body_quaternions: torch.Tensor
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor
    segment_start_times: torch.Tensor | None = None
    segment_end_times: torch.Tensor | None = None
    segment_types: torch.Tensor | None = None
    anchor_segment_start_times: torch.Tensor | None = None
    anchor_segment_end_times: torch.Tensor | None = None
    anchor_segment_labels: torch.Tensor | None = None
    anchor_frame_indices: torch.Tensor | None = None
    anchor_times: torch.Tensor | None = None

    @property
    def has_segments(self) -> bool:
        return self.segment_start_times is not None

    @property
    def num_segments(self) -> int:
        if self.segment_start_times is None:
            return 0
        return int(self.segment_start_times.shape[0])

    @property
    def has_anchor_segments(self) -> bool:
        return self.anchor_segment_start_times is not None

    @property
    def num_anchor_segments(self) -> int:
        if self.anchor_segment_start_times is None:
            return 0
        return int(self.anchor_segment_start_times.shape[0])


class MotionLib:
    def __init__(
        self,
        motion_files: MotionFileInput,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.motion_files = self._normalize_motion_files(motion_files)
        self.clips = [self._load_clip(motion_id, motion_file) for motion_id, motion_file in enumerate(self.motion_files)]
        if not self.clips:
            raise ValueError("No motion clips were loaded.")

        self.joint_names = self._load_joint_names(self.motion_files[0])
        self.body_names = self._load_body_names(self.motion_files[0])
        self._validate_clip_compatibility()

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
        self.motion_has_segments = torch.tensor(
            [clip.has_segments for clip in self.clips], dtype=torch.bool, device=self.device
        )
        self.motion_num_segments = torch.tensor(
            [clip.num_segments for clip in self.clips], dtype=torch.long, device=self.device
        )
        self.motion_segment_start_times = [clip.segment_start_times for clip in self.clips]
        self.motion_segment_end_times = [clip.segment_end_times for clip in self.clips]
        self.motion_segment_types = [clip.segment_types for clip in self.clips]
        self.motion_has_anchor_segments = torch.tensor(
            [clip.has_anchor_segments for clip in self.clips], dtype=torch.bool, device=self.device
        )
        self.motion_num_anchor_segments = torch.tensor(
            [clip.num_anchor_segments for clip in self.clips], dtype=torch.long, device=self.device
        )
        self.motion_anchor_segment_start_times = [clip.anchor_segment_start_times for clip in self.clips]
        self.motion_anchor_segment_end_times = [clip.anchor_segment_end_times for clip in self.clips]
        self.motion_anchor_segment_labels = [clip.anchor_segment_labels for clip in self.clips]
        self.motion_anchor_frame_indices = [clip.anchor_frame_indices for clip in self.clips]
        self.motion_anchor_times = [clip.anchor_times for clip in self.clips]
        self.all_clips_have_segments = bool(torch.all(self.motion_has_segments).item())
        self.all_clips_have_anchor_segments = bool(torch.all(self.motion_has_anchor_segments).item())
        self._packed_sampling_tensors: dict[str, torch.Tensor] = {}
        self._packed_sampling_enabled = self._build_packed_sampling_tensors()
        print(
            f"motion data loaded: {self.num_motions} clip(s), "
            f"durations={[round(float(duration), 3) for duration in self.motion_durations.tolist()]} s"
        )

    @staticmethod
    def _normalize_motion_files(motion_files: MotionFileInput) -> list[str]:
        if isinstance(motion_files, (str, PathLike)):
            paths = [motion_files]
        else:
            paths = list(motion_files)
        if not paths:
            raise ValueError("motion_files must contain at least one motion clip path.")
        return [str(Path(path)) for path in paths]

    def _load_joint_names(self, motion_file: str) -> list[str]:
        with np.load(motion_file) as motion_data:
            return motion_data["joint_names"].tolist()

    def _load_body_names(self, motion_file: str) -> list[str]:
        with np.load(motion_file) as motion_data:
            return motion_data["body_names"].tolist()

    def _load_clip(self, motion_id: int, motion_file: str) -> MotionClip:
        with np.load(motion_file) as motion_data:
            fps = float(np.asarray(motion_data["fps"]).item())
            if fps <= 0.0:
                raise ValueError(f"Motion clip {motion_file} has a non-positive fps: {fps}")
            dt = 1.0 / fps
            joint_pos = torch.as_tensor(motion_data["joint_pos"], dtype=torch.float32, device=self.device)
            num_frames = int(joint_pos.shape[0])
            duration = dt * num_frames
            segment_start_times = None
            segment_end_times = None
            segment_types = None
            anchor_segment_start_times = None
            anchor_segment_end_times = None
            anchor_segment_labels = None
            anchor_frame_indices = None
            anchor_times = None

            segment_keys = ("segment_start_times", "segment_end_times", "segment_types")
            has_segment_keys = [key in motion_data for key in segment_keys]
            if any(has_segment_keys):
                if not all(has_segment_keys):
                    raise ValueError(
                        f"Motion clip {motion_file} is missing part of the segment metadata: {segment_keys}"
                    )
                segment_start_times = torch.as_tensor(
                    motion_data["segment_start_times"], dtype=torch.float32, device=self.device
                ).reshape(-1)
                segment_end_times = torch.as_tensor(
                    motion_data["segment_end_times"], dtype=torch.float32, device=self.device
                ).reshape(-1)
                segment_types = torch.as_tensor(
                    motion_data["segment_types"], dtype=torch.long, device=self.device
                ).reshape(-1)
                validate_segment_arrays(
                    segment_start_times.detach().cpu().numpy(),
                    segment_end_times.detach().cpu().numpy(),
                    duration=duration,
                    segment_types=segment_types.detach().cpu().numpy(),
                )

            anchor_keys = (
                "anchor_segment_start_times",
                "anchor_segment_end_times",
                "anchor_segment_labels",
                "anchor_frame_indices",
                "anchor_times",
            )
            has_anchor_keys = [key in motion_data for key in anchor_keys]
            if any(has_anchor_keys):
                if not all(has_anchor_keys):
                    raise ValueError(
                        f"Motion clip {motion_file} is missing part of the anchor metadata: {anchor_keys}"
                    )
                anchor_segment_start_times = torch.as_tensor(
                    motion_data["anchor_segment_start_times"], dtype=torch.float32, device=self.device
                ).reshape(-1)
                anchor_segment_end_times = torch.as_tensor(
                    motion_data["anchor_segment_end_times"], dtype=torch.float32, device=self.device
                ).reshape(-1)
                anchor_segment_labels = torch.as_tensor(
                    motion_data["anchor_segment_labels"], dtype=torch.long, device=self.device
                ).reshape(-1)
                anchor_frame_indices = torch.as_tensor(
                    motion_data["anchor_frame_indices"], dtype=torch.long, device=self.device
                ).reshape(-1)
                anchor_times = torch.as_tensor(
                    motion_data["anchor_times"], dtype=torch.float32, device=self.device
                ).reshape(-1)
                _validate_anchor_segment_arrays(
                    anchor_segment_start_times.detach().cpu().numpy(),
                    anchor_segment_end_times.detach().cpu().numpy(),
                    anchor_segment_labels.detach().cpu().numpy(),
                    duration=duration,
                )
                _validate_anchor_selection_arrays(
                    anchor_frame_indices.detach().cpu().numpy(),
                    anchor_times.detach().cpu().numpy(),
                    num_frames=num_frames,
                    duration=duration,
                )

            return MotionClip(
                motion_id=motion_id,
                name=str(motion_data["name"].item()) if "name" in motion_data else Path(motion_file).stem,
                source=motion_file,
                fps=fps,
                dt=dt,
                num_frames=num_frames,
                duration=duration,
                joint_pos=joint_pos,
                joint_vel=torch.as_tensor(motion_data["joint_vel"], dtype=torch.float32, device=self.device),
                body_positions=torch.as_tensor(motion_data["body_pos_w"], dtype=torch.float32, device=self.device),
                body_quaternions=torch.as_tensor(motion_data["body_quat_w"], dtype=torch.float32, device=self.device),
                body_linear_velocities=torch.as_tensor(
                    motion_data["body_lin_vel_w"], dtype=torch.float32, device=self.device
                ),
                body_angular_velocities=torch.as_tensor(
                    motion_data["body_ang_vel_w"], dtype=torch.float32, device=self.device
                ),
                segment_start_times=segment_start_times,
                segment_end_times=segment_end_times,
                segment_types=segment_types,
                anchor_segment_start_times=anchor_segment_start_times,
                anchor_segment_end_times=anchor_segment_end_times,
                anchor_segment_labels=anchor_segment_labels,
                anchor_frame_indices=anchor_frame_indices,
                anchor_times=anchor_times,
            )

    def _validate_clip_compatibility(self) -> None:
        reference_joint_names = self._load_joint_names(self.motion_files[0])
        reference_body_names = self._load_body_names(self.motion_files[0])
        reference_joint_dim = self.clips[0].joint_pos.shape[1:]
        reference_body_dim = self.clips[0].body_positions.shape[1:]

        for motion_file, clip in zip(self.motion_files[1:], self.clips[1:], strict=False):
            joint_names = self._load_joint_names(motion_file)
            body_names = self._load_body_names(motion_file)
            if joint_names != reference_joint_names:
                raise ValueError(f"Joint names do not match across motion clips: {motion_file}")
            if body_names != reference_body_names:
                raise ValueError(f"Body names do not match across motion clips: {motion_file}")
            if clip.joint_pos.shape[1:] != reference_joint_dim:
                raise ValueError(f"Joint tensor shape mismatch across motion clips: {motion_file}")
            if clip.body_positions.shape[1:] != reference_body_dim:
                raise ValueError(f"Body tensor shape mismatch across motion clips: {motion_file}")

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

    def get_duration(self, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        if motion_ids.numel() == 0:
            return torch.empty_like(motion_ids, dtype=torch.float32)
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
        max_num_frames = int(self.motion_num_frames.max().item())

        for output_name, clip_attribute in sample_fields.items():
            clip_tensors = [getattr(clip, clip_attribute) for clip in self.clips]
            reference_shape = clip_tensors[0].shape[1:]
            if any(tensor.device != reference_device or tensor.shape[1:] != reference_shape for tensor in clip_tensors):
                self._packed_sampling_tensors = {}
                return False

            packed_shape = (self.num_motions, max_num_frames, *reference_shape)
            packed = torch.empty(packed_shape, dtype=torch.float32, device=self.device)
            for motion_id, tensor in enumerate(clip_tensors):
                packed[motion_id, : tensor.shape[0]] = tensor
            packed_tensors[output_name] = packed

        self._packed_sampling_tensors = packed_tensors
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
        motion_ids: torch.Tensor,
        index_0: torch.Tensor,
        index_1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        return interpolate(values[motion_ids, index_0], b=values[motion_ids, index_1], blend=blend)

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

        packed = self._packed_sampling_tensors
        joint_pos = self._interpolate_packed_frames(packed["joint_pos"], motion_ids, index_0, index_1, blend)
        joint_vel = self._interpolate_packed_frames(packed["joint_vel"], motion_ids, index_0, index_1, blend)
        body_positions = self._interpolate_packed_frames(
            packed["body_positions"],
            motion_ids,
            index_0,
            index_1,
            blend,
        )

        if position_offsets is not None:
            if position_offsets.ndim == 2:
                position_offsets = position_offsets.unsqueeze(1)
            body_positions = body_positions + position_offsets

        body_quaternions = slerp(
            packed["body_quaternions"][motion_ids, index_0],
            q1=packed["body_quaternions"][motion_ids, index_1],
            blend=blend,
        )
        body_linear_velocities = self._interpolate_packed_frames(
            packed["body_linear_velocities"],
            motion_ids,
            index_0,
            index_1,
            blend,
        )
        body_angular_velocities = self._interpolate_packed_frames(
            packed["body_angular_velocities"],
            motion_ids,
            index_0,
            index_1,
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
    ) -> dict[str, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1)
        if motion_ids.shape != times.shape:
            raise ValueError("motion_ids and times must have the same batch shape.")
        if position_offsets is not None:
            position_offsets = torch.as_tensor(position_offsets, dtype=torch.float32, device=self.device)
            if position_offsets.shape[0] != times.shape[0]:
                raise ValueError("position_offsets must have the same batch size as motion_ids and times.")
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

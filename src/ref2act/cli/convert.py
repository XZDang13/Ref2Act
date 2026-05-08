import argparse
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import torch
from isaaclab.app import AppLauncher

from ref2act.common.utils import compute_frame_blend_from_fps, interpolate, slerp
from ref2act.motion.segments import (
    DEFAULT_AIRBORNE_HEIGHT_MARGIN,
    build_anchor_selection_diagnostics,
    build_contact_segments,
    infer_ground_contact_from_foot_heights,
    validate_segment_arrays,
)
from ref2act.motion.smoothing import DEFAULT_SMOOTHING_PROFILE, SMOOTHING_PROFILES, smooth_motion_trajectory

SEGMENT_METHOD_TIME = "time"
SEGMENT_METHOD_ANCHOR = "anchor"
DEFAULT_SPLIT_MAX_DURATION = 10.0
DEFAULT_SPLIT_MIN_DURATION = 2.0
_FRAME_ALIGNED_KEYS = {
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}
_SEGMENT_TIME_GROUPS = (
    ("segment_start_times", "segment_end_times", "segment_types"),
)
_ANCHOR_ALIGNED_KEYS = {
    "anchor_frame_indices",
    "anchor_times",
    "anchor_joint_kinetic_energy",
}


@dataclass(frozen=True)
class ConversionOptions:
    device: str | torch.device
    height_offset: float
    segment_bin_size: float
    airborne_height_threshold: float
    segment_method: Literal["time", "anchor"]
    smooth_motion: bool
    smoothing_profile: str
    target_fps: int | None
    split_by_anchors: bool
    split_max_duration: float
    split_min_duration: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ConversionOptions":
        if args.split_min_duration >= args.split_max_duration:
            raise ValueError("--split-min-duration must be less than --split-max-duration")
        return cls(
            device=args.device,
            height_offset=args.height_offset,
            segment_bin_size=args.segment_bin_size,
            airborne_height_threshold=args.airborne_height_threshold,
            segment_method=args.segment_method,
            smooth_motion=args.smooth_motion,
            smoothing_profile=args.smoothing_profile,
            target_fps=args.target_fps,
            split_by_anchors=args.split_by_anchors,
            split_max_duration=args.split_max_duration,
            split_min_duration=args.split_min_duration,
        )


@dataclass
class ConversionRuntime:
    sim: object
    scene: object
    joint_order: list[str]
    fps: int
    render_interval: int
    num_agents: int


@dataclass(frozen=True)
class ConversionFailure:
    input_file: Path
    output_file: Path
    error: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def add_conversion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--height_offset",
        type=float,
        default=0.0,
        help="Offset to root z position.",
    )
    parser.add_argument(
        "--segment-bin-size",
        type=float,
        default=0.3,
        help="Base time-bin size in seconds used to build segment-aware failure bins.",
    )
    parser.add_argument(
        "--airborne-height-threshold",
        type=float,
        default=DEFAULT_AIRBORNE_HEIGHT_MARGIN,
        help="Height above each foot's baseline required for both feet to be treated as airborne.",
    )
    parser.add_argument(
        "--segment-method",
        type=str,
        choices=(SEGMENT_METHOD_TIME, SEGMENT_METHOD_ANCHOR),
        default=SEGMENT_METHOD_TIME,
        help="Segment export method. 'time' preserves the current segment bins, 'anchor' adds anchor metadata.",
    )
    parser.add_argument(
        "--smooth-motion",
        action="store_true",
        help="Apply temporal smoothing to the imported root and joint trajectories before exporting.",
    )
    parser.add_argument(
        "--smoothing-profile",
        type=str,
        choices=tuple(SMOOTHING_PROFILES),
        default=DEFAULT_SMOOTHING_PROFILE,
        help="Smoothing strength profile to use when --smooth-motion is enabled.",
    )
    parser.add_argument(
        "--target-fps",
        type=_positive_int,
        default=None,
        help="Resample the exported motion clip to this output frequency in Hz. Defaults to the source fps.",
    )
    parser.add_argument(
        "--split-by-anchors",
        action="store_true",
        help="Split long exported clips at anchor frames instead of writing one full-length clip.",
    )
    parser.add_argument(
        "--split-max-duration",
        type=_positive_float,
        default=DEFAULT_SPLIT_MAX_DURATION,
        help="Maximum target duration in seconds for anchor-split clips.",
    )
    parser.add_argument(
        "--split-min-duration",
        type=_positive_float,
        default=DEFAULT_SPLIT_MIN_DURATION,
        help="Minimum allowed duration in seconds for anchor-split clips.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GMR pickle motion file into the Ref2Act .npz format."
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to a GMR .pkl file.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output .npz file. Defaults to the input path with an .npz suffix.",
    )
    add_conversion_arguments(parser)
    return parser

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap NumPy 2.x internal module path to NumPy 1.x path
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _quat_conjugate(quaternions: torch.Tensor) -> torch.Tensor:
    conjugate = quaternions.clone()
    conjugate[..., 1:] *= -1.0
    return conjugate


def _quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def _axis_angle_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    normalized = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)
    normalized = torch.where(normalized[..., :1] < 0.0, -normalized, normalized)

    vector = normalized[..., 1:]
    vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, normalized[..., :1].clamp(min=-1.0, max=1.0))
    axis = vector / vector_norm.clamp_min(1.0e-8)
    axis_angle = axis * angle
    return torch.where(vector_norm > 1.0e-8, axis_angle, 2.0 * vector)


def _linear_derivative(values: torch.Tensor, dt: float) -> torch.Tensor:
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    return torch.gradient(values, spacing=dt, dim=0)[0]


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    num_frames = rotations.shape[0]
    angular_velocity_shape = (*rotations.shape[:-1], 3)
    if num_frames == 0:
        return torch.empty(angular_velocity_shape, device=rotations.device, dtype=rotations.dtype)
    if num_frames == 1:
        return torch.zeros(angular_velocity_shape, device=rotations.device, dtype=rotations.dtype)
    if num_frames == 2:
        q_rel = _quat_mul(rotations[1:], _quat_conjugate(rotations[:-1]))
        omega = _axis_angle_from_quat(q_rel) / dt
        return torch.cat([omega, omega], dim=0)

    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = _quat_mul(q_next, _quat_conjugate(q_prev))
    omega = _axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _build_frame_times(num_frames: int, fps: int, device: torch.device) -> torch.Tensor:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    return torch.arange(num_frames, dtype=torch.float32, device=device) / float(fps)


def _normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    return quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _resample_frames(
    values: torch.Tensor,
    *,
    source_fps: int,
    target_fps: int,
    target_num_frames: int,
    quaternion: bool = False,
) -> torch.Tensor:
    if values.shape[0] == 0:
        return values.clone()

    target_times = _build_frame_times(target_num_frames, target_fps, values.device)
    index_0, index_1, blend = compute_frame_blend_from_fps(target_times, source_fps, values.shape[0])

    if quaternion:
        return _normalize_quaternions(slerp(values, q1=values, blend=blend, start=index_0, end=index_1))
    return interpolate(values, b=values, blend=blend, start=index_0, end=index_1)


def _resample_motion_log(
    log: dict[str, object],
    foot_heights: np.ndarray,
    *,
    source_fps: int,
    target_fps: int,
) -> tuple[dict[str, object], np.ndarray]:
    source_num_frames = int(np.asarray(log["joint_pos"]).shape[0])
    if source_num_frames < 1:
        raise ValueError("Converted motion log must contain at least one frame.")

    target_num_frames = max(int(round(float(source_num_frames) * float(target_fps) / float(source_fps))), 1)
    target_dt = 1.0 / float(target_fps)

    joint_pos = torch.as_tensor(log["joint_pos"], dtype=torch.float32)
    body_pos_w = torch.as_tensor(log["body_pos_w"], dtype=torch.float32)
    body_quat_w = torch.as_tensor(log["body_quat_w"], dtype=torch.float32)
    foot_height_tensor = torch.as_tensor(foot_heights, dtype=torch.float32)

    resampled_joint_pos = _resample_frames(
        joint_pos,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )
    resampled_body_pos_w = _resample_frames(
        body_pos_w,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )
    resampled_body_quat_w = _resample_frames(
        body_quat_w,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
        quaternion=True,
    )
    resampled_foot_heights = _resample_frames(
        foot_height_tensor,
        source_fps=source_fps,
        target_fps=target_fps,
        target_num_frames=target_num_frames,
    )

    resampled_log = {
        "fps": float(target_fps),
        "joint_names": log["joint_names"],
        "body_names": log["body_names"],
        "joint_pos": resampled_joint_pos.cpu().numpy(),
        "joint_vel": _linear_derivative(resampled_joint_pos, target_dt).cpu().numpy(),
        "body_pos_w": resampled_body_pos_w.cpu().numpy(),
        "body_quat_w": resampled_body_quat_w.cpu().numpy(),
        "body_lin_vel_w": _linear_derivative(resampled_body_pos_w, target_dt).cpu().numpy(),
        "body_ang_vel_w": _so3_derivative(resampled_body_quat_w, target_dt).cpu().numpy(),
    }
    return resampled_log, resampled_foot_heights.cpu().numpy()

class GMRMotionData:
    def __init__(
        self,
        file: str,
        device: torch.device,
        joint_order: list[str],
        height_offset: float = 0.0,
        smooth_motion: bool = False,
        smoothing_profile: str = DEFAULT_SMOOTHING_PROFILE,
    ):
        with open(file, "rb") as f:
            motion_data = NumpyCompatUnpickler(f).load()

        self.device = device
        self.joint_order = joint_order
        print(motion_data["fps"])
        self.fps = round(motion_data["fps"])
        self.offset = torch.zeros(3).to(self.device)
        self.offset[-1] += height_offset
        self.root_pos = torch.from_numpy(motion_data["root_pos"]).float().to(self.device) + self.offset
        self.root_rot = torch.from_numpy(motion_data["root_rot"]).float().to(self.device)
        self.root_rot = self.root_rot[:, [3, 0, 1, 2]]
        self.joint_pos = torch.from_numpy(motion_data["dof_pos"]).float().to(self.device)
        self.num_frames = self.joint_pos.size(0)
        if smooth_motion:
            self.root_pos, self.root_rot, self.joint_pos = smooth_motion_trajectory(
                self.root_pos,
                self.root_rot,
                self.joint_pos,
                fps=float(self.fps),
                profile=smoothing_profile,
            )
            print(f"[INFO]: Enabled motion smoothing with profile '{smoothing_profile}'.")
        self.render_interval = 1
        self.physic_dt = 1 / (self.render_interval * self.fps)
        self.current_step = 0

        self.root_lin_vel = _linear_derivative(self.root_pos, self.physic_dt)
        self.root_ang_vel = _so3_derivative(self.root_rot, self.physic_dt)
        self.joint_vel = _linear_derivative(self.joint_pos, self.physic_dt)

    def _linear_derivative(self, values: torch.Tensor, dt: float) -> torch.Tensor:
        return _linear_derivative(values, dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        return _so3_derivative(rotations, dt)
    
    def get_init_state(self):
        motion = (
            self.root_pos[0 : 0 + 1],
            self.root_rot[0 : 0 + 1],
            self.root_lin_vel[0 : 0 + 1],
            self.root_ang_vel[0 : 0 + 1],
            self.joint_pos[0 : 0 + 1],
            self.joint_vel[0 : 0 + 1],
        )

        return motion
    
    def set_root_height(self, height_offset:float):
        offset = torch.zeros(3).to(self.device)
        offset[-1] += height_offset

        self.root_pos += offset
    
    def get_next_state(self):
        motion = (
            self.root_pos[self.current_step : self.current_step + 1],
            self.root_rot[self.current_step : self.current_step + 1],
            self.root_lin_vel[self.current_step : self.current_step + 1],
            self.root_ang_vel[self.current_step : self.current_step + 1],
            self.joint_pos[self.current_step : self.current_step + 1],
            self.joint_vel[self.current_step : self.current_step + 1],
        )
        self.current_step += 1

        reset_flag = False
        if self.current_step >= self.num_frames:
            self.current_step = 0
            reset_flag = True
        return motion, reset_flag


def peek_motion_fps(file: str | Path) -> int:
    with open(file, "rb") as f:
        motion_data = NumpyCompatUnpickler(f).load()
    return round(motion_data["fps"])


def create_conversion_runtime(
    device: str | torch.device,
    fps: int,
    num_agents: int = 1,
) -> ConversionRuntime:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from ref2act.robots.g1 import G1_CFG, JOINT_ORDER

    if num_agents < 1:
        raise ValueError("num_agents must be at least 1.")

    @configclass
    class ConversionSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        dome_light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    render_interval = 1
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / fps, render_interval=render_interval, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = ConversionSceneCfg(num_envs=num_agents, env_spacing=2.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    return ConversionRuntime(
        sim=sim,
        scene=scene,
        joint_order=JOINT_ORDER,
        fps=fps,
        render_interval=render_interval,
        num_agents=num_agents,
    )


def reset_conversion_runtime(runtime: ConversionRuntime) -> None:
    runtime.sim.reset()
    runtime.scene.reset()


def resolve_output_file(input_file: str | Path, output_file: str | Path | None = None) -> Path:
    if output_file is not None:
        return Path(output_file)
    return Path(input_file).with_suffix(".npz")


def extract_feet_height(robot, foot_body_names: list[str], env_id: int = 0) -> dict[str, float]:
    foot_body_indices = [robot.data.body_names.index(body_name) for body_name in foot_body_names]
    foot_heights = robot.data.body_pos_w[env_id, foot_body_indices, 2].detach().cpu().tolist()
    return {body_name: float(height) for body_name, height in zip(foot_body_names, foot_heights, strict=True)}


def extract_foot_heights(robot, foot_body_indices: list[int], env_id: int = 0) -> np.ndarray:
    return robot.data.body_pos_w[env_id, foot_body_indices, 2].detach().cpu().numpy().copy()


def log_in_air_events(has_ground_contact: np.ndarray, dt: float) -> None:
    ground_contact = np.asarray(has_ground_contact, dtype=bool).reshape(-1)
    in_air = ~ground_contact
    run_start: int | None = None

    for frame_index, is_in_air in enumerate(in_air):
        if is_in_air and run_start is None:
            run_start = frame_index
        elif not is_in_air and run_start is not None:
            run_end = frame_index - 1
            start_time = run_start * dt
            end_time = (run_end + 1) * dt
            print(
                f"[IN AIR]: frames {run_start}-{run_end}, "
                f"time {start_time:.3f}s-{end_time:.3f}s"
            )
            run_start = None

    if run_start is not None:
        run_end = len(in_air) - 1
        start_time = run_start * dt
        end_time = (run_end + 1) * dt
        print(
            f"[IN AIR]: frames {run_start}-{run_end}, "
            f"time {start_time:.3f}s-{end_time:.3f}s"
        )


@dataclass
class _MotionConversionState:
    env_id: int
    input_file: Path
    output_file: Path
    motion_data: GMRMotionData
    log: dict[str, object]
    foot_height_frames: list[np.ndarray] = field(default_factory=list)


def _create_motion_log(robot, motion_data: GMRMotionData) -> dict[str, object]:
    return {
        "fps": motion_data.fps,
        "joint_names": robot.data.joint_names,
        "body_names": robot.data.body_names,
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

def _resolve_env_ids(robot, slots: Sequence[_MotionConversionState]) -> torch.Tensor:
    return torch.tensor([slot.env_id for slot in slots], device=robot.device, dtype=torch.long)


def _initialize_motion_slots(robot, scene, slots: Sequence[_MotionConversionState], joint_indices: Sequence[int]) -> None:
    env_ids = _resolve_env_ids(robot, slots)
    root_states = robot.data.default_root_state[env_ids].clone()
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()

    for row, slot in enumerate(slots):
        motion = slot.motion_data.get_init_state()
        (
            reference_root_pos,
            reference_root_rot,
            reference_root_lin_vel,
            reference_root_ang_vel,
            reference_joint_pose,
            reference_joint_vel,
        ) = motion
        root_states[row, :3] = reference_root_pos[0]
        root_states[row, :2] += scene.env_origins[slot.env_id, :2]
        root_states[row, 3:7] = reference_root_rot[0]
        root_states[row, 7:10] = reference_root_lin_vel[0]
        root_states[row, 10:] = reference_root_ang_vel[0]
        joint_pos[row, joint_indices] = reference_joint_pose[0]
        joint_vel[row, joint_indices] = reference_joint_vel[0]

    robot.write_root_state_to_sim(root_states, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _apply_root_height_offsets(robot, slots: Sequence[_MotionConversionState], foot_body_names: list[str]) -> list[int]:
    foot_body_indices = [robot.data.body_names.index(body_name) for body_name in foot_body_names]
    for slot in slots:
        feet_height = extract_feet_height(robot, foot_body_names, env_id=slot.env_id)
        lowest_foot_height = min(feet_height.values()) - 0.03
        slot.motion_data.set_root_height(-lowest_foot_height)
        print(
            f"[INFO]: Applied root height offset {-lowest_foot_height:.6f} "
            f"from lowest foot height {lowest_foot_height:.6f} for {slot.input_file}"
        )
    return foot_body_indices


def _capture_motion_frame(robot, slot: _MotionConversionState, foot_body_indices: list[int]) -> None:
    env_id = slot.env_id
    slot.foot_height_frames.append(extract_foot_heights(robot, foot_body_indices, env_id=env_id))
    slot.log["joint_pos"].append(robot.data.joint_pos[env_id, :].cpu().numpy().copy())
    slot.log["joint_vel"].append(robot.data.joint_vel[env_id, :].cpu().numpy().copy())
    slot.log["body_pos_w"].append(robot.data.body_pos_w[env_id, :].cpu().numpy().copy())
    slot.log["body_quat_w"].append(robot.data.body_quat_w[env_id, :].cpu().numpy().copy())
    slot.log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[env_id, :].cpu().numpy().copy())
    slot.log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[env_id, :].cpu().numpy().copy())


def _validate_split_durations(max_duration: float, min_duration: float) -> None:
    if max_duration <= 0.0:
        raise ValueError("split max duration must be > 0.")
    if min_duration <= 0.0:
        raise ValueError("split min duration must be > 0.")
    if min_duration >= max_duration:
        raise ValueError("split min duration must be less than split max duration.")


def _motion_frame_count_and_fps(log: dict[str, object]) -> tuple[int, float]:
    joint_pos = np.asarray(log["joint_pos"])
    if joint_pos.ndim < 1 or joint_pos.shape[0] < 1:
        raise ValueError("Converted motion log must contain at least one frame.")
    fps = float(np.asarray(log["fps"]).item())
    if fps <= 0.0:
        raise ValueError("Converted motion log must contain a positive fps.")
    return int(joint_pos.shape[0]), fps


def _anchor_split_chunks(
    log: dict[str, object],
    *,
    max_duration: float,
    min_duration: float,
) -> list[tuple[int, int]]:
    _validate_split_durations(max_duration, min_duration)
    num_frames, fps = _motion_frame_count_and_fps(log)
    duration = float(num_frames) / fps
    tolerance = 1.0e-6
    if duration <= max_duration + tolerance:
        return [(0, num_frames)]
    if "anchor_frame_indices" not in log:
        raise ValueError("Anchor-based splitting requires anchor_frame_indices metadata.")

    anchor_indices = np.asarray(log["anchor_frame_indices"], dtype=np.int64).reshape(-1)
    if np.any(anchor_indices < 0) or np.any(anchor_indices >= num_frames):
        raise ValueError("anchor_frame_indices contain values outside the clip frame range.")
    candidate_anchors = np.unique(anchor_indices[(anchor_indices > 0) & (anchor_indices < num_frames)])
    if candidate_anchors.size == 0:
        raise ValueError("Anchor-based splitting requires at least one future anchor frame.")

    boundaries = [0]
    start_frame = 0
    while (num_frames - start_frame) / fps > max_duration + tolerance:
        anchor_durations = (candidate_anchors - start_frame).astype(np.float64) / fps
        future_mask = candidate_anchors > start_frame
        in_window_mask = (
            future_mask
            & (anchor_durations + tolerance >= min_duration)
            & (anchor_durations <= max_duration + tolerance)
        )
        if np.any(in_window_mask):
            end_frame = int(candidate_anchors[in_window_mask][-1])
        else:
            after_min_mask = future_mask & (anchor_durations + tolerance >= min_duration)
            if not np.any(after_min_mask):
                start_time = start_frame / fps
                raise ValueError(
                    "Anchor-based splitting could not find an anchor at least "
                    f"{min_duration:.3f}s after chunk start {start_time:.3f}s."
                )
            end_frame = int(candidate_anchors[after_min_mask][0])

        if end_frame <= start_frame:
            raise ValueError("Anchor-based splitting selected a non-advancing boundary.")
        boundaries.append(end_frame)
        start_frame = end_frame

    boundaries.append(num_frames)
    if len(boundaries) > 2:
        final_duration = (boundaries[-1] - boundaries[-2]) / fps
        if final_duration + tolerance < min_duration:
            del boundaries[-2]

    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def _copy_log_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    return value


def _clip_time_segments(
    log: dict[str, object],
    *,
    start_key: str,
    end_key: str,
    value_key: str,
    start_time: float,
    end_time: float,
) -> dict[str, np.ndarray]:
    present = [key in log for key in (start_key, end_key, value_key)]
    if not any(present):
        return {}
    if not all(present):
        raise ValueError(
            f"Motion log is missing part of the segment metadata: {start_key}, {end_key}, {value_key}."
        )

    starts = np.asarray(log[start_key], dtype=np.float64).reshape(-1)
    ends = np.asarray(log[end_key], dtype=np.float64).reshape(-1)
    values = np.asarray(log[value_key]).reshape(-1)
    if starts.shape != ends.shape or starts.shape != values.shape:
        raise ValueError(f"{start_key}, {end_key}, and {value_key} must have matching shapes.")

    chunk_duration = float(end_time - start_time)
    if chunk_duration <= 0.0:
        raise ValueError("Split chunk must have positive duration.")

    overlap_starts = np.maximum(starts, start_time)
    overlap_ends = np.minimum(ends, end_time)
    mask = overlap_ends > overlap_starts + 1.0e-7
    if not np.any(mask):
        raise ValueError(f"No {start_key}/{end_key} metadata overlaps the split chunk.")

    shifted_starts = np.clip(overlap_starts[mask] - start_time, 0.0, chunk_duration).astype(np.float32)
    shifted_ends = np.clip(overlap_ends[mask] - start_time, 0.0, chunk_duration).astype(np.float32)
    shifted_values = values[mask].copy()
    shifted_starts[0] = np.float32(0.0)
    shifted_ends[-1] = np.float32(chunk_duration)

    positive_duration_mask = shifted_ends > shifted_starts
    if not np.all(positive_duration_mask):
        shifted_starts = shifted_starts[positive_duration_mask]
        shifted_ends = shifted_ends[positive_duration_mask]
        shifted_values = shifted_values[positive_duration_mask]
        if shifted_starts.size == 0:
            raise ValueError(f"No positive-duration {start_key}/{end_key} metadata remains after clipping.")
        shifted_starts[0] = np.float32(0.0)
        shifted_ends[-1] = np.float32(chunk_duration)

    if shifted_starts.size > 1:
        shifted_starts[1:] = shifted_ends[:-1]
    validate_segment_arrays(
        shifted_starts,
        shifted_ends,
        duration=chunk_duration,
        segment_types=shifted_values if value_key == "segment_types" else None,
    )
    return {
        start_key: shifted_starts,
        end_key: shifted_ends,
        value_key: shifted_values,
    }


def _slice_motion_log_by_frames(log: dict[str, object], start_frame: int, end_frame: int) -> dict[str, object]:
    num_frames, fps = _motion_frame_count_and_fps(log)
    if start_frame < 0 or end_frame > num_frames or start_frame >= end_frame:
        raise ValueError("Invalid split frame range.")

    start_time = float(start_frame) / fps
    end_time = float(end_frame) / fps
    segment_keys = {key for group in _SEGMENT_TIME_GROUPS for key in group}
    special_keys = _FRAME_ALIGNED_KEYS | _ANCHOR_ALIGNED_KEYS | segment_keys
    chunk: dict[str, object] = {}

    for key, value in log.items():
        if key in special_keys:
            continue
        chunk[key] = _copy_log_value(value)

    for key in _FRAME_ALIGNED_KEYS:
        if key not in log:
            continue
        values = np.asarray(log[key])
        if values.shape[0] != num_frames:
            raise ValueError(f"{key} must have one entry per motion frame.")
        chunk[key] = values[start_frame:end_frame].copy()

    for start_key, end_key, value_key in _SEGMENT_TIME_GROUPS:
        chunk.update(
            _clip_time_segments(
                log,
                start_key=start_key,
                end_key=end_key,
                value_key=value_key,
                start_time=start_time,
                end_time=end_time,
            )
        )

    if "anchor_frame_indices" in log:
        anchor_indices = np.asarray(log["anchor_frame_indices"], dtype=np.int64).reshape(-1)
        anchor_mask = (anchor_indices >= start_frame) & (anchor_indices < end_frame)
        chunk["anchor_frame_indices"] = (anchor_indices[anchor_mask] - start_frame).astype(np.int64, copy=False)

        if "anchor_times" not in log:
            raise ValueError("Motion log with anchor_frame_indices must also contain anchor_times.")
        anchor_times = np.asarray(log["anchor_times"], dtype=np.float32).reshape(-1)
        if anchor_times.shape != anchor_indices.shape:
            raise ValueError("anchor_frame_indices and anchor_times must have matching shapes.")
        chunk["anchor_times"] = (anchor_times[anchor_mask] - np.float32(start_time)).astype(np.float32, copy=False)

        for key in sorted(_ANCHOR_ALIGNED_KEYS - {"anchor_frame_indices", "anchor_times"}):
            if key not in log:
                continue
            values = np.asarray(log[key])
            if values.shape[0] != anchor_indices.shape[0]:
                raise ValueError(f"{key} must have one entry per selected anchor.")
            chunk[key] = values[anchor_mask].copy()

    return chunk


def _split_part_output_file(output_file: Path, part_index: int) -> Path:
    suffix = output_file.suffix or ".npz"
    return output_file.with_name(f"{output_file.stem}_{part_index:03d}{suffix}")


def split_motion_log_by_anchors(
    log: dict[str, object],
    output_file: str | Path,
    *,
    max_duration: float = DEFAULT_SPLIT_MAX_DURATION,
    min_duration: float = DEFAULT_SPLIT_MIN_DURATION,
) -> list[Path]:
    output_path = Path(output_file)
    chunks = _anchor_split_chunks(log, max_duration=max_duration, min_duration=min_duration)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(chunks) == 1:
        np.savez(output_path, **log)
        return [output_path]

    written_files: list[Path] = []
    for part_index, (start_frame, end_frame) in enumerate(chunks):
        part_log = _slice_motion_log_by_frames(log, start_frame, end_frame)
        part_file = _split_part_output_file(output_path, part_index)
        np.savez(part_file, **part_log)
        written_files.append(part_file)
    return written_files


def _finalize_motion_slot(
    slot: _MotionConversionState,
    *,
    segment_bin_size: float,
    airborne_height_threshold: float,
    segment_method: Literal["time", "anchor"],
    target_fps: int | None,
    split_by_anchors: bool = False,
    split_max_duration: float = DEFAULT_SPLIT_MAX_DURATION,
    split_min_duration: float = DEFAULT_SPLIT_MIN_DURATION,
) -> None:
    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        slot.log[key] = np.stack(slot.log[key], axis=0)

    foot_heights = np.stack(slot.foot_height_frames, axis=0)
    output_fps = slot.motion_data.fps if target_fps is None else target_fps
    output_dt = 1.0 / float(output_fps)

    if output_fps != slot.motion_data.fps:
        slot.log, foot_heights = _resample_motion_log(
            slot.log,
            foot_heights,
            source_fps=slot.motion_data.fps,
            target_fps=output_fps,
        )
    else:
        slot.log["fps"] = float(output_fps)

    output_num_frames = int(np.asarray(slot.log["joint_pos"]).shape[0])
    output_duration = output_dt * output_num_frames

    ground_contact = infer_ground_contact_from_foot_heights(
        foot_heights,
        airborne_height_margin=airborne_height_threshold,
    )
    log_in_air_events(ground_contact, output_dt)

    segment_start_times, segment_end_times, segment_types = build_contact_segments(
        has_ground_contact=ground_contact,
        dt=output_dt,
        duration=output_duration,
        bin_size=segment_bin_size,
    )
    slot.log["segment_start_times"] = segment_start_times
    slot.log["segment_end_times"] = segment_end_times
    slot.log["segment_types"] = segment_types
    if segment_method == SEGMENT_METHOD_ANCHOR or split_by_anchors:
        anchor_diagnostics = build_anchor_selection_diagnostics(
            slot.log,
            airborne_height_margin=airborne_height_threshold,
        )
        slot.log.update(anchor_diagnostics.metadata.as_npz_dict())
        print(
            "[INFO]: Anchor export selected "
            f"{anchor_diagnostics.metadata.frame_indices.shape[0]} anchor(s) "
            f"(lowest_energy_fallback={'yes' if anchor_diagnostics.used_lowest_energy_fallback else 'no'}) "
            f"for {slot.input_file}"
        )

    if split_by_anchors:
        written_files = split_motion_log_by_anchors(
            slot.log,
            slot.output_file,
            max_duration=split_max_duration,
            min_duration=split_min_duration,
        )
        if len(written_files) > 1:
            print(
                f"[INFO]: Anchor split wrote {len(written_files)} motion npz files "
                f"from {slot.input_file}: {', '.join(str(path) for path in written_files)}"
            )
        else:
            print("[INFO]: Motion npz file saved to", written_files[0])
    else:
        slot.output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(slot.output_file, **slot.log)
        print("[INFO]: Motion npz file saved to", slot.output_file)


def _run_motion_chunk(
    sim,
    scene,
    slots: Sequence[_MotionConversionState],
    simulation_app,
    *,
    segment_bin_size: float,
    airborne_height_threshold: float,
    segment_method: Literal["time", "anchor"],
    target_fps: int | None,
    split_by_anchors: bool,
    split_max_duration: float,
    split_min_duration: float,
    camera_follow: bool,
) -> tuple[list[ConversionFailure], bool]:
    robot = scene["robot"]
    joint_indices = robot.find_joints(slots[0].motion_data.joint_order, preserve_order=True)[0]
    foot_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    active_slots = list(slots)
    failures: list[ConversionFailure] = []

    try:
        _initialize_motion_slots(robot, scene, active_slots, joint_indices)
        sim.render()
        scene.update(sim.get_physics_dt())
        foot_body_indices = _apply_root_height_offsets(robot, active_slots, foot_body_names)

        while active_slots and simulation_app.is_running():
            env_ids = _resolve_env_ids(robot, active_slots)
            root_states = robot.data.default_root_state[env_ids].clone()
            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            reset_flags: list[bool] = []

            for row, slot in enumerate(active_slots):
                motion, reset_flag = slot.motion_data.get_next_state()
                (
                    reference_root_pos,
                    reference_root_rot,
                    reference_root_lin_vel,
                    reference_root_ang_vel,
                    reference_joint_pose,
                    reference_joint_vel,
                ) = motion
                root_states[row, :3] = reference_root_pos[0]
                root_states[row, :2] += scene.env_origins[slot.env_id, :2]
                root_states[row, 3:7] = reference_root_rot[0]
                root_states[row, 7:10] = reference_root_lin_vel[0]
                root_states[row, 10:] = reference_root_ang_vel[0]
                joint_pos[row, joint_indices] = reference_joint_pose[0]
                joint_vel[row, joint_indices] = reference_joint_vel[0]
                reset_flags.append(reset_flag)

            robot.write_root_state_to_sim(root_states, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            sim.render()
            scene.update(sim.get_physics_dt())

            if camera_follow:
                pos_lookat = root_states[0, :3].detach().cpu().numpy()
                sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

            next_active_slots: list[_MotionConversionState] = []
            for slot, reset_flag in zip(active_slots, reset_flags, strict=True):
                _capture_motion_frame(robot, slot, foot_body_indices)
                if reset_flag:
                    try:
                        _finalize_motion_slot(
                            slot,
                            segment_bin_size=segment_bin_size,
                            airborne_height_threshold=airborne_height_threshold,
                            segment_method=segment_method,
                            target_fps=target_fps,
                            split_by_anchors=split_by_anchors,
                            split_max_duration=split_max_duration,
                            split_min_duration=split_min_duration,
                        )
                    except Exception as exc:
                        failures.append(
                            ConversionFailure(
                                input_file=slot.input_file,
                                output_file=slot.output_file,
                                error=str(exc),
                            )
                        )
                else:
                    next_active_slots.append(slot)
            active_slots = next_active_slots

        if active_slots:
            raise RuntimeError("Simulation app stopped before conversion completed.")
    except Exception as exc:
        for slot in active_slots:
            failures.append(
                ConversionFailure(
                    input_file=slot.input_file,
                    output_file=slot.output_file,
                    error=str(exc),
                )
            )
        return failures, True

    return failures, False


def convert_motion_files(
    motion_files: Sequence[tuple[str | Path, str | Path]],
    *,
    runtime: ConversionRuntime,
    simulation_app,
    options: ConversionOptions,
    camera_follow: bool = False,
) -> list[ConversionFailure]:
    if len(motion_files) > runtime.num_agents:
        raise ValueError(
            f"Received {len(motion_files)} motions but runtime supports only {runtime.num_agents} agents."
        )

    if not motion_files:
        return []

    reset_conversion_runtime(runtime)
    robot = runtime.scene["robot"]
    slots: list[_MotionConversionState] = []
    failures: list[ConversionFailure] = []

    for env_id, (input_file, output_file) in enumerate(motion_files):
        input_path = Path(input_file)
        output_path = Path(output_file)
        try:
            motion_data = GMRMotionData(
                str(input_path),
                options.device,
                runtime.joint_order,
                options.height_offset,
                smooth_motion=options.smooth_motion,
                smoothing_profile=options.smoothing_profile,
            )
            if motion_data.fps != runtime.fps:
                raise ValueError(
                    f"Motion fps {motion_data.fps} does not match runtime fps {runtime.fps} for {input_path}."
                )
            slots.append(
                _MotionConversionState(
                    env_id=env_id,
                    input_file=input_path,
                    output_file=output_path,
                    motion_data=motion_data,
                    log=_create_motion_log(robot, motion_data),
                )
            )
        except Exception as exc:
            failures.append(
                ConversionFailure(
                    input_file=input_path,
                    output_file=output_path,
                    error=str(exc),
                )
            )

    if not slots:
        return failures

    chunk_failures, runtime_failed = _run_motion_chunk(
        runtime.sim,
        runtime.scene,
        slots,
        simulation_app,
        segment_bin_size=options.segment_bin_size,
        airborne_height_threshold=options.airborne_height_threshold,
        segment_method=options.segment_method,
        target_fps=options.target_fps,
        split_by_anchors=options.split_by_anchors,
        split_max_duration=options.split_max_duration,
        split_min_duration=options.split_min_duration,
        camera_follow=camera_follow,
    )
    failures.extend(chunk_failures)

    if runtime_failed:
        reset_conversion_runtime(runtime)

    return failures


def convert_motion_file(
    input_file: str | Path,
    output_file: str | Path,
    *,
    runtime: ConversionRuntime,
    simulation_app,
    options: ConversionOptions,
) -> Path:
    output_path = Path(output_file)
    failures = convert_motion_files(
        [(input_file, output_path)],
        runtime=runtime,
        simulation_app=simulation_app,
        options=options,
        camera_follow=True,
    )
    if failures:
        raise RuntimeError(failures[0].error)
    return output_path

def main(argv: list[str] | None = None):
    args_cli = build_parser().parse_args(argv)
    output_file = resolve_output_file(args_cli.input_file, args_cli.output_file)
    options = ConversionOptions.from_args(args_cli)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        runtime = create_conversion_runtime(options.device, peek_motion_fps(args_cli.input_file))
        print("[INFO]: Setup complete...")
        convert_motion_file(
            args_cli.input_file,
            output_file,
            runtime=runtime,
            simulation_app=simulation_app,
            options=options,
        )
    finally:
        runtime = locals().get("runtime")
        if runtime is not None:
            runtime.sim.clear_instance()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)

if __name__ == "__main__":
    main()

import torch
import math
from enum import Enum
from dataclasses import dataclass, field

from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_mul, quat_inv, quat_apply, yaw_quat, quat_from_euler_xyz
from .motion_lib import MotionLib
from .utils import IndexLike

@dataclass
class ReferenceMotions:
    # ---- reference motion (world frame) ----
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_positions: torch.Tensor          # [B, N, 3]
    body_quaternions: torch.Tensor        # [B, N, 4]
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor

    # ---- anchor setup ----
    anchor_body_index: int

    # ---- OPTIONAL: robot body poses (world frame), same body order as motion ----
    # You can pass these from robot.data.body_pos_w / body_quat_w.
    robot_body_positions: torch.Tensor | None = None   # [B, N, 3]
    robot_body_quaternions: torch.Tensor | None = None # [B, N, 4]

    # ---- derived fields ----
    body_pos_relative: torch.Tensor = field(init=False)     # [B, N, 3]
    body_quat_relative: torch.Tensor = field(init=False)    # [B, N, 4]

    def __post_init__(self) -> None:
        """
        Compute body_pos_relative_w and body_quat_relative_w.

        If robot_body_* are provided, this matches MotionCommand logic:
          - delta_pos: XY from robot anchor, Z from reference anchor
          - delta_ori: yaw( robot_anchor_quat * inv(ref_anchor_quat) )
          - body_quat_relative_w = delta_ori * ref_body_quat_w
          - body_pos_relative_w  = delta_pos + quat_apply(delta_ori, ref_body_pos_w - ref_anchor_pos_w)
        If robot_body_* are NOT provided, it will produce an identity mapping
        (i.e., relative fields ~= original reference in world), which is usually not what you want
        for tracking, so generally pass robot poses.
        """
        if not (0 <= self.anchor_body_index < self.body_positions.shape[1]):
            raise IndexError(f"anchor_body_index out of range: {self.anchor_body_index}")

        B, N, _ = self.body_positions.shape

        # reference anchor (world)
        ref_anchor_pos_w = self.body_positions[:, self.anchor_body_index]    # [B, 3]
        ref_anchor_quat_w = self.body_quaternions[:, self.anchor_body_index] # [B, 4]

        # robot anchor (world)
        if self.robot_body_positions is None or self.robot_body_quaternions is None:
            # Fallback (not recommended for your use case)
            robot_anchor_pos_w = ref_anchor_pos_w
            robot_anchor_quat_w = ref_anchor_quat_w
        else:
            if self.robot_body_positions.shape[:2] != (B, N):
                raise ValueError(
                    f"robot_body_positions_w must be [B,N,3], got {tuple(self.robot_body_positions.shape)}"
                )
            if self.robot_body_quaternions.shape[:2] != (B, N):
                raise ValueError(
                    f"robot_body_quaternions_w must be [B,N,4], got {tuple(self.robot_body_quaternions.shape)}"
                )
            robot_anchor_pos_w = self.robot_body_positions[:, self.anchor_body_index]    # [B, 3]
            robot_anchor_quat_w = self.robot_body_quaternions[:, self.anchor_body_index] # [B, 4]

        # broadcast to all bodies
        ref_anchor_pos_rep = ref_anchor_pos_w[:, None, :].expand_as(self.body_positions)       # [B,1,3]
        ref_anchor_quat_rep = ref_anchor_quat_w[:, None, :].expand_as(self.body_quaternions)      # [B,1,4]
        robot_anchor_pos_rep = robot_anchor_pos_w[:, None, :].expand_as(self.body_positions)    # [B,1,3]
        robot_anchor_quat_rep = robot_anchor_quat_w[:, None, :].expand_as(self.body_quaternions)  # [B,1,4]

        # delta position: XY from robot, Z from reference
        delta_pos_w = robot_anchor_pos_rep.clone()
        delta_pos_w[..., 2] = ref_anchor_pos_rep[..., 2]

        # delta orientation: yaw( robot_anchor * inv(ref_anchor) )
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_rep, quat_inv(ref_anchor_quat_rep)))  # [B,1,4]

        # derived orientation
        # [B,N,4] = [B,1,4] (*) [B,N,4]
        self.body_quat_relative = quat_mul(delta_ori_w, self.body_quaternions)

        # derived position
        rel = self.body_positions - ref_anchor_pos_rep                 # [B,N,3]
        rel_rot = quat_apply(delta_ori_w, rel)                         # [B,N,3]
        self.body_pos_relative = delta_pos_w + rel_rot               # [B,N,3]


class SamplerMod(Enum):
    Cycle = 0
    Clamp = 1

POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

JOINT_POSITION_RANGE = (-0.1, 0.1)

def pose_noise(size:int, noise_ranges:dict[str, tuple[float, float]], device:torch.device):
    position_noise = []
    for key in ["x", "y", "z"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        position_noise.append(noise)

    euler_noise = []
    for key in ["roll", "pitch", "yaw"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, device=device).uniform_(low_range, up_range)
        euler_noise.append(noise)

    position_noise = torch.cat(position_noise, dim=-1)
    quaterions_noise = quat_from_euler_xyz(*euler_noise)

    return position_noise, quaterions_noise

def velocity_noise(size:int, noise_ranges:dict[str, tuple[float, float]], device:torch.device):
    linear_vel_noise = []
    for key in ["x", "y", "z"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        linear_vel_noise.append(noise)

    ang_vel_noise = []
    for key in ["roll", "pitch", "yaw"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        ang_vel_noise.append(noise)

    linear_vel_noise = torch.cat(linear_vel_noise, dim=-1)
    ang_vel_noise = torch.cat(ang_vel_noise, dim=-1)

    return linear_vel_noise, ang_vel_noise

class InitialSetting:
    pose_range = POSE_RANGE
    velcoity_range = VELOCITY_RANGE
    joint_position_range = JOINT_POSITION_RANGE

    @staticmethod
    def set_robot_initial_state(
        robot: Articulation,
        env_ids: torch.Tensor,
        motion_samples: ReferenceMotions,
        root_index: int,
        add_noise:bool
    ) -> None:
        joint_pos = motion_samples.joint_pos
        joint_vel = motion_samples.joint_vel
        root_pos = motion_samples.body_positions[:, root_index]
        root_quat = motion_samples.body_quaternions[:, root_index]
        root_linear_vel = motion_samples.body_linear_velocities[:, root_index]
        root_angular_vel = motion_samples.body_angular_velocities[:, root_index]

        root_state = robot.data.default_root_state[env_ids].clone()

        if add_noise:
            device = root_pos.device
            root_pos_noise, root_quat_noise = pose_noise(len(env_ids), InitialSetting.pose_range, device)
            root_linear_vel_noise, root_angular_vel_noise = velocity_noise(len(env_ids), InitialSetting.velcoity_range, device)
            joint_pose_noise = torch.empty_like(joint_pos).uniform_(InitialSetting.joint_position_range[0],
                                                                    InitialSetting.joint_position_range[1])

            root_pos += root_pos_noise
            root_quat = quat_mul(root_quat, root_quat_noise)
            root_linear_vel += root_linear_vel_noise
            root_angular_vel += root_angular_vel_noise
            joint_pos += joint_pose_noise

        root_state[:, 0:3] = root_pos
        #root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_linear_vel
        root_state[:, 10:13] = root_angular_vel

        robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

class Sampler:
    def __init__(
        self,
        num_envs: int,
        motion_lib: MotionLib,
        dt: float,
        anchor_body_index: int,
        reset_noise:bool=False,
        bin_size: float | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.motion_lib = motion_lib
        self.anchor_body_index = anchor_body_index
        self.reset_noise = reset_noise

        self.current_times = torch.zeros(num_envs, device=self.device)

        self.bin_size: float | None = None
        self.num_bins = 0
        self.bin_fail_counts: torch.Tensor | None = None
        self.bin_sample_counts: torch.Tensor | None = None

        if bin_size is not None:
            self.init_failure_bins(bin_size)

    # -------------------------
    # Properties
    # -------------------------
    @property
    def duration(self) -> float:
        return self.motion_lib.duration

    # ============================================================
    # ----------- Low-level: time sampling only -----------------
    # ============================================================
    def _sample_rand_times(self, env_ids: IndexLike | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        num_envs = len(env_ids)

        times = torch.rand(num_envs, device=self.device) * self.duration
        self.current_times[env_ids] = times
        self._record_sample_bins(times)

    def _sample_start_times(self, env_ids: IndexLike | None = None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        num_envs = len(env_ids)

        times = torch.zeros(num_envs, device=self.device)
        self.current_times[env_ids] = times
        self._record_sample_bins(times)

    def _sample_failure_weighted_times(
        self,
        env_ids: IndexLike | None = None,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ):

        self._check_failure_bins()

        if env_ids is None:
            env_ids = list(range(self.num_envs))
        num_envs = len(env_ids)

        if min_weight < 0.0:
            raise ValueError("min_weight must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        fail_rate = self.bin_fail_counts / torch.clamp(self.bin_sample_counts, min=1.0)
        weights = (fail_rate + min_weight).pow(1.0 / temperature)

        if torch.sum(weights) <= 0:
            weights = torch.ones_like(weights)

        bin_indices = torch.multinomial(weights, num_envs, replacement=True)
        bin_starts = bin_indices.to(dtype=torch.float32) * self.bin_size
        duration_tensor = torch.tensor(self.duration, dtype=torch.float32)
        bin_ends = torch.minimum(bin_starts + self.bin_size, duration_tensor)

        times = bin_starts + torch.rand(num_envs, device=self.device) * (bin_ends - bin_starts)

        self._record_sample_bins(times)
        self.current_times[env_ids] = times

    def _sample_next_times(self):
        self.current_times += self.dt

        #if on_end == SamplerMod.Cycle:
        #    self.current_times = torch.remainder(self.current_times, self.duration)
        #elif on_end == SamplerMod.Clamp:
        #    self.current_times = torch.clamp(self.current_times, max=self.duration)
        #else:
        #    raise ValueError(f"Unknown on_end mode: {on_end}")

    # ============================================================
    # ----------- High-level: MOTION sampling -------------------
    # ============================================================
    def _build_reference_motions(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        scene: InteractiveScene
    ) -> ReferenceMotions:

        times = self.current_times[env_ids]
        position_offsets = scene.env_origins[env_ids]

        robot_body_positions = robot.data.body_pos_w[env_ids]
        robot_body_quaternions = robot.data.body_quat_w[env_ids]
        
        motions = self.motion_lib.sample_motion(
            times=times,
            position_offsets=position_offsets,
        )

        return ReferenceMotions(
            joint_pos=motions["joint_pos"],
            joint_vel=motions["joint_vel"],
            body_positions=motions["body_positions"],
            body_quaternions=motions["body_quaternions"],
            body_linear_velocities=motions["body_linear_velocities"],
            body_angular_velocities=motions["body_angular_velocities"],
            anchor_body_index=self.anchor_body_index,
            robot_body_positions=robot_body_positions,
            robot_body_quaternions=robot_body_quaternions
        )

    # -------------------------
    # Public API (RENAMED)
    # -------------------------
    def sample_rand_motions(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        scene: InteractiveScene
    ) -> ReferenceMotions:
        self._sample_rand_times(env_ids)

        motion = self._build_reference_motions(env_ids, robot, scene)
        InitialSetting.set_robot_initial_state(robot, env_ids, motion, self.anchor_body_index, self.reset_noise)
        return motion

    def sample_start_motions(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        scene: InteractiveScene
    ) -> ReferenceMotions:
        self._sample_start_times(env_ids)

        motion = self._build_reference_motions(env_ids, robot, scene)
        InitialSetting.set_robot_initial_state(robot, env_ids, motion, self.anchor_body_index, self.reset_noise)
        return motion

    def sample_failure_weighted_motions(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        scene: InteractiveScene,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> ReferenceMotions:
        self._sample_failure_weighted_times(
            env_ids=env_ids,
            min_weight=min_weight,
            temperature=temperature,
        )

        motion = self._build_reference_motions(env_ids, robot, scene)
        InitialSetting.set_robot_initial_state(robot, env_ids, motion, self.anchor_body_index, self.reset_noise)
        return motion

    def sample_next_motions(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        scene: InteractiveScene,
    ) -> ReferenceMotions:
        self._sample_next_times()

        return self._build_reference_motions(env_ids, robot, scene)

    # ============================================================
    # ---------------- failure-bin utils ------------------------
    # ============================================================
    def init_failure_bins(self, bin_size: float) -> None:
        if bin_size <= 0.0:
            raise ValueError("bin_size must be > 0")

        self.bin_size = float(bin_size)
        self.num_bins = max(1, int(math.ceil(self.duration / self.bin_size)))
        self.bin_fail_counts = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        self.bin_sample_counts = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)

    def reset_failure_stats(self) -> None:
        self._check_failure_bins()
        self.bin_fail_counts.zero_()
        self.bin_sample_counts.zero_()

    def record_failures(
        self,
        env_ids: IndexLike | None = None,
        times: torch.Tensor | None = None,
    ) -> None:
        self._check_failure_bins()
        if times is None:
            times = self.current_times if env_ids is None else self.current_times[env_ids]

        bin_indices = self._times_to_bins(times)
        self.bin_fail_counts += torch.bincount(
            bin_indices, minlength=self.num_bins
        ).to(dtype=torch.float32)

    def _check_failure_bins(self) -> None:
        if (
            self.bin_size is None
            or self.bin_fail_counts is None
            or self.bin_sample_counts is None
        ):
            raise RuntimeError("Failure bins not initialized. Call init_failure_bins(...) first.")

    def _times_to_bins(self, times: torch.Tensor) -> torch.Tensor:
        bin_indices = torch.floor(times / self.bin_size).to(dtype=torch.long)
        return torch.clamp(bin_indices, min=0, max=self.num_bins - 1)

    def _record_sample_bins(self, times: torch.Tensor) -> None:
        if self.bin_size is None or self.bin_sample_counts is None:
            return
        bin_indices = self._times_to_bins(times)
        self.bin_sample_counts += torch.bincount(
            bin_indices, minlength=self.num_bins
        ).to(dtype=torch.float32)

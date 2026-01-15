import torch
import math
from enum import Enum
from dataclasses import dataclass, field

from isaaclab.utils.math import quat_mul, quat_inv, quat_apply, yaw_quat
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
    # You can pass these from robot.data.body_link_pos_w / body_link_quat_w.
    robot_body_positions_w: torch.Tensor | None = None   # [B, N, 3]
    robot_body_quaternions_w: torch.Tensor | None = None # [B, N, 4]

    # ---- derived fields ----
    body_pos_relative_w: torch.Tensor = field(init=False)     # [B, N, 3]
    body_quat_relative_w: torch.Tensor = field(init=False)    # [B, N, 4]

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
        if self.robot_body_positions_w is None or self.robot_body_quaternions_w is None:
            # Fallback (not recommended for your use case)
            robot_anchor_pos_w = ref_anchor_pos_w
            robot_anchor_quat_w = ref_anchor_quat_w
        else:
            if self.robot_body_positions_w.shape[:2] != (B, N):
                raise ValueError(
                    f"robot_body_positions_w must be [B,N,3], got {tuple(self.robot_body_positions_w.shape)}"
                )
            if self.robot_body_quaternions_w.shape[:2] != (B, N):
                raise ValueError(
                    f"robot_body_quaternions_w must be [B,N,4], got {tuple(self.robot_body_quaternions_w.shape)}"
                )
            robot_anchor_pos_w = self.robot_body_positions_w[:, self.anchor_body_index]    # [B, 3]
            robot_anchor_quat_w = self.robot_body_quaternions_w[:, self.anchor_body_index] # [B, 4]

        # broadcast to all bodies
        ref_anchor_pos_rep = ref_anchor_pos_w[:, None, :]       # [B,1,3]
        ref_anchor_quat_rep = ref_anchor_quat_w[:, None, :]     # [B,1,4]
        robot_anchor_pos_rep = robot_anchor_pos_w[:, None, :]   # [B,1,3]
        robot_anchor_quat_rep = robot_anchor_quat_w[:, None, :] # [B,1,4]

        # delta position: XY from robot, Z from reference
        delta_pos_w = robot_anchor_pos_rep.clone()
        delta_pos_w[..., 2] = ref_anchor_pos_rep[..., 2]

        # delta orientation: yaw( robot_anchor * inv(ref_anchor) )
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_rep, quat_inv(ref_anchor_quat_rep)))  # [B,1,4]

        # derived orientation
        # [B,N,4] = [B,1,4] (*) [B,N,4]
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quaternions)

        # derived position
        rel = self.body_positions - ref_anchor_pos_rep                 # [B,N,3]
        rel_rot = quat_apply(delta_ori_w, rel)                         # [B,N,3]
        self.body_pos_relative_w = delta_pos_w + rel_rot               # [B,N,3]


class SamplerMod(Enum):
    Cycle = 0
    Clamp = 1


class Sampler:
    def __init__(
        self,
        num_envs: int,
        motion_lib: MotionLib,
        dt: float,
        bin_size: float | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.motion_lib = motion_lib

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
    def _sample_rand_times(self, env_ids: IndexLike | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        num_envs = len(env_ids)

        times = torch.rand(num_envs, device=self.device) * self.duration
        self.current_times[env_ids] = times
        self._record_sample_bins(times)
        return times

    def _sample_start_times(self, env_ids: IndexLike | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        num_envs = len(env_ids)

        times = torch.zeros(num_envs, device=self.device)
        self.current_times[env_ids] = times
        self._record_sample_bins(times)
        return times

    def _sample_failure_weighted_times(
        self,
        env_ids: IndexLike | None = None,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> torch.Tensor:

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
        return times

    def _sample_next_times(self, on_end: SamplerMod = SamplerMod.Clamp) -> torch.Tensor:
        self.current_times += self.dt

        if on_end == SamplerMod.Cycle:
            self.current_times = torch.remainder(self.current_times, self.duration)
        elif on_end == SamplerMod.Clamp:
            self.current_times = torch.clamp(self.current_times, max=self.duration)
        else:
            raise ValueError(f"Unknown on_end mode: {on_end}")

        return self.current_times.clone()

    # ============================================================
    # ----------- High-level: MOTION sampling -------------------
    # ============================================================
    def _build_reference_motions(
        self,
        times: torch.Tensor,
        position_offsets: torch.Tensor | None = None,
    ) -> ReferenceMotions:

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
        )

    # -------------------------
    # Public API (RENAMED)
    # -------------------------
    def sample_rand_motions(
        self,
        env_ids: IndexLike | None = None,
        position_offsets: torch.Tensor | None = None,
    ) -> ReferenceMotions:
        times = self._sample_rand_times(env_ids)
        return self._build_reference_motions(times, position_offsets)

    def sample_start_motions(
        self,
        env_ids: IndexLike | None = None,
        position_offsets: torch.Tensor | None = None,
    ) -> ReferenceMotions:
        times = self._sample_start_times(env_ids)
        return self._build_reference_motions(times, position_offsets)

    def sample_failure_weighted_motions(
        self,
        env_ids: IndexLike | None = None,
        min_weight: float = 0.001,
        temperature: float = 1.0,
        position_offsets: torch.Tensor | None = None,
    ) -> ReferenceMotions:
        times = self._sample_failure_weighted_times(
            env_ids=env_ids,
            min_weight=min_weight,
            temperature=temperature,
        )
        return self._build_reference_motions(times, position_offsets)

    def sample_next_motions(
        self,
        on_end: SamplerMod = SamplerMod.Clamp,
        position_offsets: torch.Tensor | None = None,
    ) -> ReferenceMotions:
        times = self._sample_next_times(on_end)
        return self._build_reference_motions(times, position_offsets)

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

from enum import Enum
import math
import numpy as np
import torch
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from dataclasses import dataclass

from .utils import interpolate, slerp, compute_frame_blend, IndexLike

class SamplerMod(Enum):
    Cycle = 0
    Clamp = 1

@dataclass
class ReferenceMotions:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_positions: torch.Tensor
    body_quaternions: torch.Tensor
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor

class MotionLib:
    def __init__(self, motion_file: str, device: torch.device=torch.device("cpu")) -> None:
        
        motion_data = np.load(motion_file)
        self.device = device

        self.fps = motion_data["fps"]

        self.joint_names = motion_data["joint_names"].tolist()
        self.body_names = motion_data["body_names"].tolist()

        self.joint_pos = torch.as_tensor(motion_data["joint_pos"], dtype=torch.float32, device=self.device)
        self.joint_vel = torch.as_tensor(motion_data["joint_vel"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(motion_data["body_pos_w"], dtype=torch.float32, device=self.device)
        self.body_quaternions = torch.tensor(motion_data["body_quat_w"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            motion_data["body_lin_vel_w"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            motion_data["body_ang_vel_w"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / self.fps
        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.dt * self.num_frames
        print(f"motion data loaded: {self.duration} s")

    def sample_motion(
        self,
        times: torch.Tensor,
        position_offsets: torch.Tensor|None=None
    ) -> ReferenceMotions:
        
        index_0, index_1, blend = compute_frame_blend(
            times, self.duration, self.num_frames, self.dt
        )

        index_0 = index_0.to(device=self.device)
        index_1 = index_1.to(device=self.device)
        blend = blend.to(device=self.device, dtype=torch.float32)

        joint_pos = interpolate(
            self.joint_pos, b=self.joint_pos, blend=blend, start=index_0, end=index_1
        )

        joint_vel = interpolate(
            self.joint_vel, b=self.joint_vel, blend=blend, start=index_0, end=index_1
        )

        body_positions = interpolate(
            self.body_positions, b=self.body_positions, blend=blend, start=index_0, end=index_1
        )

        if position_offsets is not None:
            if position_offsets.ndim == 2:
                position_offsets = position_offsets.unsqueeze(1)
            body_positions += position_offsets

        body_quaternions = slerp(
            self.body_quaternions, q1=self.body_quaternions, blend=blend, start=index_0, end=index_1
        )
        body_linear_velocities = interpolate(
            self.body_linear_velocities, b=self.body_linear_velocities, blend=blend, start=index_0, end=index_1
        )
        body_angular_velocities = interpolate(
            self.body_angular_velocities, b=self.body_angular_velocities, blend=blend, start=index_0, end=index_1
        )

        sampled_motion = ReferenceMotions(
            joint_pos, joint_vel, body_positions, body_quaternions, body_linear_velocities, body_angular_velocities
        )
        
        return sampled_motion
    
class Sampler:
    def __init__(
        self,
        num_envs: int,
        duration: float,
        dt: float,
        num_frames: int,
        bin_size: float | None = None,
        device: torch.device = torch.device("cpu")
    ) -> None:
        
        self.num_envs = num_envs
        self.duration = duration
        self.dt = dt
        self.num_frames = num_frames
        self.device = device

        self.current_times = torch.zeros(num_envs).to(self.device)
        self.bin_size: float | None = None
        self.num_bins = 0
        self.bin_fail_counts: torch.Tensor | None = None
        self.bin_sample_counts: torch.Tensor | None = None
        if bin_size is not None:
            self.init_failure_bins(bin_size)

    def init_failure_bins(self, bin_size: float) -> None:
        if bin_size <= 0.0:
            raise ValueError("bin_size must be > 0")
        self.bin_size = float(bin_size)
        self.num_bins = max(1, int(math.ceil(self.duration / self.bin_size)))
        self.bin_fail_counts = torch.zeros(self.num_bins, dtype=torch.float32).to(self.device)
        self.bin_sample_counts = torch.zeros(self.num_bins, dtype=torch.float32).to(self.device)

    def reset_failure_stats(self) -> None:
        self._check_failure_bins()
        self.bin_fail_counts.zero_()
        self.bin_sample_counts.zero_()

    def record_failures(self, env_ids: IndexLike | None = None, times: torch.Tensor | None = None) -> None:
        self._check_failure_bins()
        if times is None:
            if env_ids is None:
                times = self.current_times
            else:
                times = self.current_times[env_ids]
        bin_indices = self._times_to_bins(times)
        self.bin_fail_counts += torch.bincount(bin_indices, minlength=self.num_bins).to(dtype=torch.float32)

    def sample_rand_times(self, env_ids:IndexLike|None=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = [i for i in range(self.num_envs)]
        num_envs = len(env_ids)
        times = torch.rand(num_envs, device=self.device) * self.duration
        self.current_times[env_ids] = times

        return times

    def sample_start_times(self, env_ids:IndexLike|None=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = [i for i in range(self.num_envs)]
        num_envs = len(env_ids)
        times = torch.zeros(num_envs, device=self.device)
        self.current_times[env_ids] = times

        return times

    def sample_failure_weighted_times(
        self,
        env_ids: IndexLike | None = None,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self._check_failure_bins()
        if env_ids is None:
            env_ids = [i for i in range(self.num_envs)]
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

        self.bin_sample_counts += torch.bincount(bin_indices, minlength=self.num_bins).to(dtype=torch.float32)
        self.current_times[env_ids] = times

        return times

    def get_bin_distributions(
        self,
        min_weight: float = 0.05,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self._check_failure_bins()
        if min_weight < 0.0:
            raise ValueError("min_weight must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        fail_rate = self.bin_fail_counts / torch.clamp(self.bin_sample_counts, min=1.0)
        weights = (fail_rate + min_weight).pow(1.0 / temperature)
        weight_sum = torch.clamp(weights.sum(), min=1.0)
        sample_prob = weights / weight_sum
        return sample_prob
    
    def sample_next(self, on_end: SamplerMod=SamplerMod.Clamp) -> torch.Tensor:
        self.current_times += self.dt
        if on_end == SamplerMod.Cycle:
            self.current_times = torch.remainder(self.current_times, self.duration)
        elif on_end == SamplerMod.Clamp:
            self.current_times = torch.clamp(self.current_times, max=self.duration)
        else:
            raise ValueError(f"Unknown on_end mode: {on_end}")
        times = self.current_times.clone()

        return times

    def _check_failure_bins(self) -> None:
        if self.bin_size is None or self.bin_fail_counts is None or self.bin_sample_counts is None:
            raise RuntimeError("Failure bins not initialized. Call init_failure_bins(...) first.")

    def _times_to_bins(self, times: torch.Tensor) -> torch.Tensor:
        bin_indices = torch.floor(times / self.bin_size).to(dtype=torch.long)
        return torch.clamp(bin_indices, min=0, max=self.num_bins - 1)

class MotionViewer:
    """
    Helper class to visualize motion data from NumPy-file format.
    """

    def __init__(self, motion_file: str, device: torch.device | str = "cpu", render_scene: bool = False) -> None:
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
        self.motion_lib = MotionLib(motion_file=motion_file, device=device)

        self.num_frames = self.motion_lib.num_frames
        self.current_frame = 0
        self.body_positions = self.motion_lib.body_positions.cpu().numpy()
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
            interval=1000 * self.motion_lib.dt,
        )
        plt.show()

import numpy as np
import torch
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from dataclasses import dataclass

from .utils import interpolate, slerp, compute_frame_blend, IndexLike


@dataclass
class ReferenceMotions:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_positions: torch.Tensor
    body_quaternions: torch.Tensor
    body_linear_velocities: torch.Tensor
    body_angular_velocities: torch.Tensor

class MotionLib:
    def __init__(self, motion_file: str, joint_names:list[str]|None=None, device: torch.device=torch.device("cpu")) -> None:
        
        motion_data = np.load(motion_file)
        self.device = device

        self.fps = motion_data["fps"]

        self.joint_names = motion_data["joint_names"].tolist()
        self.body_names = motion_data["body_names"].tolist()

        if joint_names is None:
            joint_names = self.joint_names
        
        self.joint_order = [self.joint_names.index(name) for name in joint_names]

        self.joint_pos = torch.as_tensor(motion_data["joint_positions"], dtype=torch.float32, device=self.device)[:, self.joint_order]
        self.joint_vel = torch.as_tensor(motion_data["joint_velocities"], dtype=torch.float32, device=self.device)[:, self.joint_order]
        self.body_positions = torch.tensor(motion_data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_quaternions = torch.tensor(motion_data["body_quaternions"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            motion_data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            motion_data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / self.fps
        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.dt * self.num_frames

    def get_body_indices(self, body_names: list[str]) -> list[int]:
        return [self.body_names.index(name) for name in body_names]
    
    def get_body_index(self, body_name: str) -> int:
        return self.body_names.index(body_name)

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
    def __init__(self, num_envs: int, duration: float, dt: float, num_frames: int) -> None:
        self.num_envs = num_envs
        self.duration = duration
        self.dt = dt
        self.num_frames = num_frames

        self.current_times = torch.zeros(num_envs)

    def sample_rand_times(self, env_ids:IndexLike|None=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = [i for i in range(self.num_envs)]
        num_envs = len(env_ids)
        times = torch.rand(num_envs) * self.duration
        self.current_times[env_ids] = times

        return times

    def sample_start_times(self, env_ids:IndexLike|None=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = [i for i in range(self.num_envs)]
        num_envs = len(env_ids)
        times = torch.zeros(num_envs)
        self.current_times[env_ids] = times

        return times
    
    def sample_next(self, on_end: str = "cycle") -> torch.Tensor:
        self.current_times += self.dt
        if on_end == "cycle":
            self.current_times = torch.remainder(self.current_times, self.duration)
        elif on_end == "clamp":
            self.current_times = torch.clamp(self.current_times, max=self.duration)
        else:
            raise ValueError(f"Unknown on_end mode: {on_end}")
        times = self.current_times.clone()

        return times

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

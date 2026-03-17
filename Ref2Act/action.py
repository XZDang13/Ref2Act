import torch
from isaaclab.assets import Articulation
from .buffer import DequeBuffer
from .utils import IndexLike


class ActionProcessor:

    def __init__(self, robot: Articulation, action_buffer_length: int = 1, noise_scale: float = 0.0):
        self.scale: float | torch.Tensor = 0.0
        self.offset: float | torch.Tensor = 0.0
        self.noise_scale = noise_scale
        self.device = robot.data.device
        self.num_env = robot.data.joint_pos.size(0)
        self.action_size = robot.data.joint_pos.size(1)

        self.action_buffer = DequeBuffer(
            self.num_env,
            action_buffer_length,
            (self.action_size,),
            device=self.device,
        )

        self.delays = torch.zeros(self.num_env, device=self.device, dtype=torch.long)
        self.applied_action = torch.zeros_like(robot.data.default_joint_pos)
        self.offset_noise = torch.zeros_like(robot.data.default_joint_pos)

    def set_median_scale_offset(
        self,
        robot: Articulation,
    ) -> None:
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.scale = 0.5 * (self.joint_up_limit - self.joint_low_limit)
        self.offset = 0.5 * (self.joint_up_limit + self.joint_low_limit)

    def set_robot_default_scale_offset(
        self,
        robot: Articulation,
    ):
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.offset = robot.data.default_joint_pos
        self.scale = 0.25 * (robot.data.joint_effort_limits[0] / robot.data.default_joint_stiffness[0])
        '''
        print("KP:")
        print(robot.data.joint_stiffness[0])
        print("KD")
        print(robot.data.joint_damping[0])
        print("Action Offset")
        print(self.offset[0])
        print("Action Scale")
        print(self.scale)
        print("------------")
        '''

    def _resolve_env_ids(self, env_ids: IndexLike) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def reset_action_buffer(self, env_ids: IndexLike):
        env_ids = self._resolve_env_ids(env_ids)
        self.action_buffer.reset(env_ids)
        self.applied_action[env_ids, :] = 0.0

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.scale + self.offset

    def set_random_delays(self, env_ids: IndexLike, delay_range: tuple[int, int]):
        env_ids = self._resolve_env_ids(env_ids)
        lower, upper = delay_range
        if lower < 0 or upper < lower:
            raise ValueError(f"Invalid action latency range: {delay_range}")
        if upper >= self.action_buffer.T:
            raise ValueError(
                f"Action latency upper bound {upper} exceeds action buffer capacity {self.action_buffer.T - 1}."
            )
        self.delays[env_ids] = torch.randint(lower, upper + 1, (len(env_ids),), device=self.device)

    def set_random_offset_noise(self, env_ids: IndexLike):
        noise = torch.empty_like(self.offset_noise[env_ids, :]).uniform_(-self.noise_scale, self.noise_scale)
        self.offset_noise[env_ids, :] = noise

    def pre_process_action(self, action: torch.Tensor):
        self.action_buffer.append(action)
        # Delay is defined in control ticks: 0 means latest action, 1 means one step stale, etc.
        self.applied_action = self.action_buffer.get(-(self.delays + 1))

        self.target_joint_position = self.applied_action * self.scale + self.offset + self.offset_noise
        self.target_joint_position.clamp_(self.joint_low_limit, self.joint_up_limit)

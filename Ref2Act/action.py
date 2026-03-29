import torch
from isaaclab.assets import Articulation
from .buffer import DequeBuffer
from .utils import IndexLike


class ActionProcessor:

    def __init__(
        self,
        robot: Articulation,
        action_buffer_length: int = 1,
        noise_scale: float = 0.0,
        action_mod: object | str = "median",
    ):
        self.robot = robot
        self.scale: float | torch.Tensor = 0.0
        self.offset: float | torch.Tensor = 0.0
        self.noise_scale = noise_scale
        self.device = robot.data.device
        self.num_env = robot.data.joint_pos.size(0)
        self.action_size = robot.data.joint_pos.size(1)
        self.action_mode = self._normalize_action_mode(action_mod)

        self.action_buffer = DequeBuffer(
            self.num_env,
            action_buffer_length,
            (self.action_size,),
            device=self.device,
        )

        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.delays = torch.zeros(self.num_env, device=self.device, dtype=torch.long)
        self.applied_action = torch.zeros_like(robot.data.default_joint_pos)
        self.previous_applied_action = torch.zeros_like(robot.data.default_joint_pos)
        self.offset_noise = torch.zeros_like(robot.data.default_joint_pos)
        self.reference_joint_position = robot.data.default_joint_pos.clone()
        self.target_joint_position = robot.data.default_joint_pos.clone()

    def _normalize_action_mode(self, action_mod: object | str) -> str:
        if isinstance(action_mod, str):
            normalized = action_mod
        elif hasattr(action_mod, "name"):
            normalized = str(action_mod.name)
        else:
            normalized = str(action_mod).split(".")[-1]

        normalized = normalized.replace("-", "_").lower()
        if normalized in {"currentresidual", "current_residual"}:
            return "current_residual"
        return normalized

    def _compute_target_joint_position(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_mode == "residual":
            return self.reference_joint_position + action * self.scale + self.offset_noise
        if self.action_mode == "current_residual":
            return self.robot.data.joint_pos + action * self.scale + self.offset_noise
        return action * self.scale + self.offset + self.offset_noise

    def _update_target_joint_position(self, env_ids: IndexLike | None = None) -> None:
        target_joint_position = self._compute_target_joint_position(self.applied_action)
        if env_ids is None:
            self.target_joint_position.copy_(target_joint_position.clamp(self.joint_low_limit, self.joint_up_limit))
            return

        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.target_joint_position[env_ids] = target_joint_position[env_ids].clamp(
            self.joint_low_limit, self.joint_up_limit
        )

    def set_median_scale_offset(
        self,
        robot: Articulation,
    ) -> None:
        self.action_mode = "median"
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.scale = 0.5 * (self.joint_up_limit - self.joint_low_limit)
        self.offset = 0.5 * (self.joint_up_limit + self.joint_low_limit)
        self._update_target_joint_position()

    def set_robot_default_scale_offset(
        self,
        robot: Articulation,
    ) -> None:
        self.action_mode = "offset"
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.offset = robot.data.default_joint_pos
        self.scale = 0.25 * (robot.data.joint_effort_limits[0] / robot.data.default_joint_stiffness[0])
        self._update_target_joint_position()
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

    def set_residual_scale_offset(
        self,
        robot: Articulation,
    ) -> None:
        self.action_mode = "residual"
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.offset = torch.zeros_like(robot.data.default_joint_pos)
        self.scale = 0.25 * (robot.data.joint_effort_limits[0] / robot.data.default_joint_stiffness[0])
        self.reference_joint_position.copy_(robot.data.default_joint_pos)
        self._update_target_joint_position()

    def set_current_residual_scale_offset(
        self,
        robot: Articulation,
    ) -> None:
        self.action_mode = "current_residual"
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.offset = torch.zeros_like(robot.data.default_joint_pos)
        self.scale = 0.25 * (robot.data.joint_effort_limits[0] / robot.data.default_joint_stiffness[0])
        self.reference_joint_position.copy_(robot.data.default_joint_pos)
        self._update_target_joint_position()

    def _resolve_env_ids(self, env_ids: IndexLike) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def set_reference_joint_position(
        self,
        reference_joint_position: torch.Tensor,
        env_ids: IndexLike | None = None,
    ) -> None:
        reference_joint_position = torch.as_tensor(
            reference_joint_position,
            device=self.device,
            dtype=self.reference_joint_position.dtype,
        )
        if env_ids is None:
            self.reference_joint_position.copy_(reference_joint_position)
            self._update_target_joint_position()
            return

        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.reference_joint_position[env_ids] = reference_joint_position
        self._update_target_joint_position(env_ids)

    def reset_action_buffer(self, env_ids: IndexLike) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        self.action_buffer.reset(env_ids)
        self.applied_action[env_ids, :] = 0.0
        self.previous_applied_action[env_ids, :] = 0.0
        self.offset_noise[env_ids, :] = 0.0
        self._update_target_joint_position(env_ids)

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(action, device=self.device, dtype=self.applied_action.dtype)
        return self._compute_target_joint_position(action).clamp(self.joint_low_limit, self.joint_up_limit)

    def set_random_delays(self, env_ids: IndexLike, delay_range: tuple[int, int]) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        lower, upper = delay_range
        if lower < 0 or upper < lower:
            raise ValueError(f"Invalid action latency range: {delay_range}")
        if upper >= self.action_buffer.T:
            raise ValueError(
                f"Action latency upper bound {upper} exceeds action buffer capacity {self.action_buffer.T - 1}."
            )
        self.delays[env_ids] = torch.randint(lower, upper + 1, (len(env_ids),), device=self.device)

    def set_random_offset_noise(self, env_ids: IndexLike) -> None:
        env_ids = self._resolve_env_ids(env_ids)
        noise = torch.empty_like(self.offset_noise[env_ids, :]).uniform_(-self.noise_scale, self.noise_scale)
        self.offset_noise[env_ids, :] = noise
        self._update_target_joint_position(env_ids)

    def pre_process_action(self, action: torch.Tensor) -> None:
        self.action_buffer.append(action)
        # Delay is defined in control ticks: 0 means latest action, 1 means one step stale, etc.
        self.previous_applied_action.copy_(self.applied_action)
        self.applied_action.copy_(self.action_buffer.get(-(self.delays + 1)))
        self._update_target_joint_position()

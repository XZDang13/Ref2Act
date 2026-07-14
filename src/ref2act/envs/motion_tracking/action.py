from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation

from ref2act.common.action_spec import (
    DEFAULT_CLAMP_POLICY,
    ActionModeContext,
    normalize_action_mode,
    resolve_action_mode_strategy,
)
from ref2act.common.buffer import DequeBuffer
from ref2act.common.utils import IndexLike
from ref2act.isaac_compat import to_torch


@dataclass
class ActionSpec:
    mode: str = "median"
    buffer_length: int = 1
    latency_range: tuple[int, int] | None = None
    noise_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.buffer_length < 1:
            raise ValueError("Action buffer_length must be at least 1.")
        if self.latency_range is not None:
            lower, upper = self.latency_range
            if lower < 0 or upper < lower:
                raise ValueError(f"Invalid action latency range: {self.latency_range}")
            if upper >= self.buffer_length:
                raise ValueError(
                    f"Action latency upper bound {upper} exceeds action buffer capacity {self.buffer_length - 1}."
                )


class ActionProcessor:
    def __init__(self, robot: Articulation, spec: ActionSpec):
        self.robot = robot
        self.spec = spec
        self.device = robot.data.device
        joint_pos = to_torch(robot.data.joint_pos)
        self.num_env = joint_pos.size(0)
        self.action_size = joint_pos.size(1)
        self.action_mode = normalize_action_mode(spec.mode)
        self._mode_strategy = resolve_action_mode_strategy(self.action_mode)

        self.action_buffer = DequeBuffer(
            self.num_env,
            spec.buffer_length,
            (self.action_size,),
            device=self.device,
        )

        self.joint_low_limit = to_torch(robot.data.joint_pos_limits)[0, :, 0]
        self.joint_up_limit = to_torch(robot.data.joint_pos_limits)[0, :, 1]
        self.delays = torch.zeros(self.num_env, device=self.device, dtype=torch.long)
        default_joint_pos = to_torch(robot.data.default_joint_pos)
        self.applied_action = torch.zeros_like(default_joint_pos)
        self.previous_applied_action = torch.zeros_like(default_joint_pos)
        self.offset_noise = torch.zeros_like(default_joint_pos)
        self.reference_joint_position = default_joint_pos.clone()
        self.target_joint_position = default_joint_pos.clone()

        self.scale: torch.Tensor
        self.offset: torch.Tensor
        self._initialize_mode_parameters(robot)
        self._update_target_joint_position()

    def _initialize_mode_parameters(self, robot: Articulation) -> None:
        if self.action_mode == "median":
            self.scale = 0.5 * (self.joint_up_limit - self.joint_low_limit)
            self.offset = 0.5 * (self.joint_up_limit + self.joint_low_limit)
            return
        if self.action_mode in {"offset", "residual", "current_residual"}:
            self.scale = 0.25 * (
                to_torch(robot.data.joint_effort_limits)[0] / to_torch(robot.data.default_joint_stiffness)[0]
            )
            if self.action_mode == "offset":
                self.offset = to_torch(robot.data.default_joint_pos).clone()
            else:
                self.offset = torch.zeros_like(to_torch(robot.data.default_joint_pos))
                self.reference_joint_position.copy_(to_torch(robot.data.default_joint_pos))
            return
        raise ValueError(f"Unsupported action mode: {self.action_mode}")

    def _resolve_env_ids(self, env_ids: IndexLike) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

    def _build_mode_context(self, action: torch.Tensor) -> ActionModeContext:
        return ActionModeContext(
            raw_action=torch.as_tensor(action, device=self.device, dtype=self.applied_action.dtype),
            action_scale=self.scale,
            action_offset=self.offset + self.offset_noise,
            joint_pos_limits_lower=self.joint_low_limit,
            joint_pos_limits_upper=self.joint_up_limit,
            current_joint_pos_loader=lambda: to_torch(self.robot.data.joint_pos),
            reference_joint_pos_loader=lambda: self.reference_joint_position,
        )

    def _compute_target_joint_position(self, action: torch.Tensor) -> torch.Tensor:
        context = self._build_mode_context(action)
        target_joint_position = self._mode_strategy.compute_target_joint_pos(context, context.raw_action)
        return DEFAULT_CLAMP_POLICY.clamp(context, target_joint_position)

    def _update_target_joint_position(self, env_ids: IndexLike | None = None) -> None:
        target_joint_position = self._compute_target_joint_position(self.applied_action)
        if env_ids is None:
            self.target_joint_position.copy_(target_joint_position)
            return

        env_ids = self._resolve_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        self.target_joint_position[env_ids] = target_joint_position[env_ids]

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
        return self._compute_target_joint_position(action)

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
        noise = torch.empty_like(self.offset_noise[env_ids, :]).uniform_(-self.spec.noise_scale, self.spec.noise_scale)
        self.offset_noise[env_ids, :] = noise
        self._update_target_joint_position(env_ids)

    def pre_process_action(self, action: torch.Tensor) -> None:
        self.action_buffer.append(torch.as_tensor(action, device=self.device, dtype=self.applied_action.dtype))
        self.previous_applied_action.copy_(self.applied_action)
        self.applied_action.copy_(self.action_buffer.get(-(self.delays + 1)))
        self._update_target_joint_position()

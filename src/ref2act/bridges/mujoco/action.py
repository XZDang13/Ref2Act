from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

import torch

if TYPE_CHECKING:
    from .env import MujocoEnv


@dataclass
class MujocoActionContext:
    raw_action: torch.Tensor
    action_mode: str
    action_scale: torch.Tensor
    action_offset: torch.Tensor
    joint_pos_limits_lower: torch.Tensor
    joint_pos_limits_upper: torch.Tensor
    current_joint_pos_loader: Callable[[], torch.Tensor]
    reference_joint_pos_loader: Callable[[], torch.Tensor]
    _current_joint_pos: torch.Tensor | None = field(default=None, init=False, repr=False)
    _reference_joint_pos: torch.Tensor | None = field(default=None, init=False, repr=False)

    def get_current_joint_pos(self) -> torch.Tensor:
        if self._current_joint_pos is None:
            self._current_joint_pos = self.current_joint_pos_loader()
        return self._current_joint_pos

    def get_reference_joint_pos(self) -> torch.Tensor:
        if self._reference_joint_pos is None:
            self._reference_joint_pos = self.reference_joint_pos_loader()
        return self._reference_joint_pos

    def clamp_target_joint_pos(self, target_joint_pos: torch.Tensor) -> torch.Tensor:
        return torch.clamp(target_joint_pos, self.joint_pos_limits_lower, self.joint_pos_limits_upper)


@dataclass(frozen=True)
class MujocoActionOutput:
    applied_action: torch.Tensor
    target_joint_pos: torch.Tensor


class MujocoActionBuilder(Protocol):
    def process_action(self, env: MujocoEnv, context: MujocoActionContext) -> MujocoActionOutput:
        ...


class IsaacLabMujocoAction:
    """Mirror the current Isaac-style action semantics while allowing overrides."""

    def get_applied_action(self, env: MujocoEnv, context: MujocoActionContext) -> torch.Tensor:
        return context.raw_action

    def get_target_joint_position(
        self,
        env: MujocoEnv,
        context: MujocoActionContext,
        applied_action: torch.Tensor,
    ) -> torch.Tensor:
        if context.action_mode == "residual":
            target_joint_pos = context.get_reference_joint_pos() + applied_action * context.action_scale
        elif context.action_mode == "current_residual":
            target_joint_pos = context.get_current_joint_pos() + applied_action * context.action_scale
        else:
            target_joint_pos = applied_action * context.action_scale + context.action_offset
        return context.clamp_target_joint_pos(target_joint_pos)

    def process_action(self, env: MujocoEnv, context: MujocoActionContext) -> MujocoActionOutput:
        applied_action = self.get_applied_action(env, context)
        target_joint_pos = self.get_target_joint_position(env, context, applied_action)
        return MujocoActionOutput(
            applied_action=applied_action,
            target_joint_pos=target_joint_pos,
        )

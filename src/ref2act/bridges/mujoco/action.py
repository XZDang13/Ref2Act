from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from ref2act.common.action_spec import (
    DEFAULT_CLAMP_POLICY,
    ActionModeContext,
    normalize_action_mode,
    resolve_action_mode_strategy,
)

if TYPE_CHECKING:
    from .env import MujocoEnv


@dataclass
class MujocoActionContext(ActionModeContext):
    action_mode: str = "absolute"


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
        strategy = resolve_action_mode_strategy(normalize_action_mode(context.action_mode))
        target_joint_pos = strategy.compute_target_joint_pos(context, applied_action)
        return DEFAULT_CLAMP_POLICY.clamp(context, target_joint_pos)

    def process_action(self, env: MujocoEnv, context: MujocoActionContext) -> MujocoActionOutput:
        applied_action = self.get_applied_action(env, context)
        target_joint_pos = self.get_target_joint_position(env, context, applied_action)
        return MujocoActionOutput(
            applied_action=applied_action,
            target_joint_pos=target_joint_pos,
        )

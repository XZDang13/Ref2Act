from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from .env import MujocoEnv


@dataclass(frozen=True)
class MujocoObservationContext:
    target_projected_gravity: torch.Tensor
    target_joint_pos: torch.Tensor
    target_joint_vel: torch.Tensor
    projected_gravity: torch.Tensor
    base_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    previous_action: torch.Tensor


class MujocoObservationBuilder(Protocol):
    def get_motion_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...

    def get_robot_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...

    def get_default_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> dict[str, torch.Tensor]:
        ...

    def get_policy_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...


class IsaacLabMujocoObservation:
    """Mirror the Isaac policy observation layout while keeping the bridge extensible."""

    def get_motion_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        return torch.cat(
            [
                context.target_projected_gravity,
                context.target_joint_pos,
                context.target_joint_vel,
            ]
        )

    def get_robot_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        return torch.cat(
            [
                context.projected_gravity,
                context.base_ang_vel,
                context.joint_pos,
                context.joint_vel,
                context.previous_action,
            ]
        )

    def get_default_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> dict[str, torch.Tensor]:
        motion_obs = self.get_motion_observation(env, context)
        robot_obs = self.get_robot_observation(env, context)
        return {
            "motion": motion_obs,
            "robot": robot_obs,
        }

    def get_policy_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        default_observation = self.get_default_observation(env, context)
        return torch.cat([default_observation["motion"], default_observation["robot"]])

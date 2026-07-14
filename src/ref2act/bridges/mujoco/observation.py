from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from ref2act.common.observation_spec import (
    ObservationComposer,
    ObservationContext,
    ObservationGroupSpec,
    ObservationLayout,
    ObservationSpec,
    ObservationTermSpec,
)
from ref2act.envs.motion_tracking.observation import default_training_observation_spec

if TYPE_CHECKING:
    from .env import MujocoEnv


MujocoObservationContext = ObservationContext


class MujocoObservationBuilder(Protocol):
    def get_motion_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...

    def get_robot_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...

    def get_default_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> dict[str, torch.Tensor]:
        ...

    def get_policy_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        ...


def default_mujoco_observation_spec() -> ObservationSpec:
    training_spec = default_training_observation_spec(add_noise=False)
    return ObservationSpec(groups=tuple(group for group in training_spec.groups if group.name in {"motion", "robot"}))


class IsaacLabMujocoObservation:
    """Mirror the Isaac policy observation layout while keeping the bridge extensible."""

    def __init__(self, spec: ObservationSpec | None = None) -> None:
        self.spec = spec or default_mujoco_observation_spec()
        self._composer: ObservationComposer | None = None
        self._layout: ObservationLayout | None = None

    def _get_layout(self, env: MujocoEnv) -> ObservationLayout:
        if self._layout is None:
            action_source = getattr(env, "action_offset", None)
            if action_source is None:
                action_source = getattr(env, "previous_action", None)
            if action_source is None:
                action_source = env.get_joint_pos()
            action_dim = int(torch.as_tensor(action_source).numel())
            self._layout = ObservationLayout(
                joint_dim=action_dim,
                action_dim=action_dim,
                key_body_count=0,
            )
        return self._layout

    def _get_composer(self, env: MujocoEnv) -> ObservationComposer:
        if self._composer is None:
            self._composer = ObservationComposer(
                spec=self.spec,
                layout=self._get_layout(env),
                num_envs=1,
                device="cpu",
            )
        return self._composer

    @staticmethod
    def _squeeze_batch(value: torch.Tensor) -> torch.Tensor:
        return value.squeeze(0) if value.ndim >= 2 and value.shape[0] == 1 else value

    def reset(self, env: MujocoEnv, context: MujocoObservationContext) -> None:
        self._get_composer(env).reset(torch.tensor([0], dtype=torch.long), context)

    def get_motion_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        return self.get_default_observation(env, context)["motion"]

    def get_robot_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        return self.get_default_observation(env, context)["robot"]

    def get_default_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> dict[str, torch.Tensor]:
        outputs = self._get_composer(env).compose(context)
        return {name: self._squeeze_batch(value) for name, value in outputs.items()}

    def get_policy_observation(self, env: MujocoEnv, context: MujocoObservationContext) -> torch.Tensor:
        default_observation = self.get_default_observation(env, context)
        return torch.cat([default_observation[group.name] for group in self.spec.enabled_groups()])

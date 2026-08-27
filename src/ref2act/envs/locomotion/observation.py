from __future__ import annotations

import torch
from isaaclab.assets import Articulation

from ref2act.common.observation_spec import (
    ObservationComposer,
    ObservationGroupSpec,
    ObservationLayout,
    ObservationSpec,
    ObservationTermSpec,
)
from ref2act.common.proprioception import build_proprioceptive_context, default_robot_observation_terms
from ref2act.common.utils import IndexLike
from ref2act.isaac_compat import to_torch


def default_locomotion_observation_spec(add_noise: bool = True) -> ObservationSpec:
    robot_terms = default_robot_observation_terms(add_noise=add_noise)
    privilege_terms = (
        ObservationTermSpec(id="priv_velocity_command", type="velocity_command"),
        *default_robot_observation_terms(add_noise=False),
        ObservationTermSpec(id="priv_anchor_lin_vel_b", type="anchor_lin_vel_b"),
    )
    return ObservationSpec(
        groups=(
            ObservationGroupSpec(
                name="command",
                terms=(ObservationTermSpec(id="velocity_command", type="velocity_command"),),
            ),
            ObservationGroupSpec(name="robot", terms=robot_terms),
            ObservationGroupSpec(name="privilege", terms=privilege_terms),
        )
    )


class LocomotionObservation:
    def __init__(
        self,
        *,
        spec: ObservationSpec,
        layout: ObservationLayout,
        num_envs: int,
        device: torch.device | str,
        anchor_body_index: int,
        policy_order_joint_indices: torch.Tensor,
    ) -> None:
        self.spec = spec
        self.layout = layout
        self.anchor_body_index = int(anchor_body_index)
        self.policy_order_joint_indices = policy_order_joint_indices
        self.composer = ObservationComposer(
            spec=spec,
            layout=layout,
            num_envs=num_envs,
            device=device,
        )

    def _context(
        self,
        robot: Articulation,
        commands: torch.Tensor,
        previous_action: torch.Tensor,
    ):
        return build_proprioceptive_context(
            anchor_quat_w=to_torch(robot.data.body_link_quat_w)[:, self.anchor_body_index],
            anchor_lin_vel_w=to_torch(robot.data.body_link_lin_vel_w)[:, self.anchor_body_index],
            anchor_ang_vel_w=to_torch(robot.data.body_link_ang_vel_w)[:, self.anchor_body_index],
            gravity_vector_w=to_torch(robot.data.GRAVITY_VEC_W),
            joint_pos=to_torch(robot.data.joint_pos)[:, self.policy_order_joint_indices],
            joint_vel=to_torch(robot.data.joint_vel)[:, self.policy_order_joint_indices],
            previous_action=previous_action,
            extras={"velocity_command": commands},
        )

    def reset(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        commands: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> None:
        self.composer.reset(env_ids, self._context(robot, commands, previous_action))

    def get_default_observation(
        self,
        robot: Articulation,
        commands: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.composer.compose(self._context(robot, commands, previous_action))


__all__ = ["LocomotionObservation", "default_locomotion_observation_spec"]

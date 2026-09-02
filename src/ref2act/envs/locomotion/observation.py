from __future__ import annotations

import torch
from isaaclab.assets import Articulation

from ref2act.common.math import quat_apply_inverse, yaw_quat
from ref2act.common.observation_spec import (
    ObservationComposer,
    ObservationGroupSpec,
    ObservationLayout,
    ObservationNoiseSpec,
    ObservationSpec,
    ObservationTermSpec,
)
from ref2act.common.proprioception import build_proprioceptive_context, default_robot_observation_terms
from ref2act.common.utils import IndexLike
from ref2act.isaac_compat import to_torch


def compute_locomotion_velocity_feedback(
    anchor_quat_w: torch.Tensor,
    anchor_lin_vel_w: torch.Tensor,
    anchor_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
    """Return feedback aligned with the commanded axes: yaw-frame vx/vy and body wz."""

    if (
        anchor_quat_w.ndim != 2
        or anchor_quat_w.shape[-1] != 4
        or anchor_lin_vel_w.shape != (anchor_quat_w.shape[0], 3)
        or anchor_ang_vel_w.shape != (anchor_quat_w.shape[0], 3)
    ):
        raise ValueError(
            "Locomotion velocity feedback expects quaternion [N, 4] and "
            "linear/angular velocities [N, 3]."
        )
    linear_velocity_yaw = quat_apply_inverse(
        yaw_quat(anchor_quat_w), anchor_lin_vel_w
    )
    angular_velocity_body = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
    return torch.stack(
        (
            linear_velocity_yaw[:, 0],
            linear_velocity_yaw[:, 1],
            angular_velocity_body[:, 2],
        ),
        dim=-1,
    )


def default_locomotion_observation_spec(
    add_noise: bool = True,
    *,
    include_gait_phase: bool = True,
    include_velocity_feedback: bool = True,
) -> ObservationSpec:
    robot_terms = default_robot_observation_terms(add_noise=add_noise)
    phase_term = (
        (ObservationTermSpec(id="locomotion_phase", type="locomotion_phase"),)
        if include_gait_phase
        else ()
    )
    privilege_phase_term = (
        (ObservationTermSpec(id="priv_locomotion_phase", type="locomotion_phase"),)
        if include_gait_phase
        else ()
    )
    privilege_terms = (
        ObservationTermSpec(id="priv_velocity_command", type="velocity_command"),
        *privilege_phase_term,
        *default_robot_observation_terms(add_noise=False),
        ObservationTermSpec(id="priv_anchor_lin_vel_b", type="anchor_lin_vel_b"),
    )
    feedback_terms = (
        ObservationTermSpec(
            id="locomotion_velocity_feedback",
            type="locomotion_velocity_feedback",
            noise=ObservationNoiseSpec(-0.1, 0.1) if add_noise else None,
        ),
    )
    return ObservationSpec(
        groups=(
            ObservationGroupSpec(
                name="command",
                terms=(
                    ObservationTermSpec(id="velocity_command", type="velocity_command"),
                    *phase_term,
                ),
            ),
            ObservationGroupSpec(
                name="feedback",
                terms=feedback_terms,
                enabled=include_velocity_feedback,
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
        locomotion_phase: torch.Tensor,
        previous_action: torch.Tensor,
    ):
        anchor_quat_w = to_torch(robot.data.body_link_quat_w)[:, self.anchor_body_index]
        anchor_lin_vel_w = to_torch(robot.data.body_link_lin_vel_w)[:, self.anchor_body_index]
        anchor_ang_vel_w = to_torch(robot.data.body_link_ang_vel_w)[
            :, self.anchor_body_index
        ]
        return build_proprioceptive_context(
            anchor_quat_w=anchor_quat_w,
            anchor_lin_vel_w=anchor_lin_vel_w,
            anchor_ang_vel_w=anchor_ang_vel_w,
            gravity_vector_w=to_torch(robot.data.GRAVITY_VEC_W),
            joint_pos=to_torch(robot.data.joint_pos)[:, self.policy_order_joint_indices],
            joint_vel=to_torch(robot.data.joint_vel)[:, self.policy_order_joint_indices],
            previous_action=previous_action,
            extras={
                "velocity_command": commands,
                "locomotion_phase": locomotion_phase,
                "locomotion_velocity_feedback": compute_locomotion_velocity_feedback(
                    anchor_quat_w, anchor_lin_vel_w, anchor_ang_vel_w
                ),
            },
        )

    def reset(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        commands: torch.Tensor,
        locomotion_phase: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> None:
        self.composer.reset(
            env_ids,
            self._context(robot, commands, locomotion_phase, previous_action),
        )

    def get_default_observation(
        self,
        robot: Articulation,
        commands: torch.Tensor,
        locomotion_phase: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.composer.compose(
            self._context(robot, commands, locomotion_phase, previous_action)
        )


__all__ = [
    "LocomotionObservation",
    "compute_locomotion_velocity_feedback",
    "default_locomotion_observation_spec",
]

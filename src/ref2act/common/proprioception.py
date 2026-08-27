from __future__ import annotations

from typing import Mapping

import torch

from .math import quat_apply_inverse, quaternion_to_rotation_6d
from .observation_spec import ObservationContext, ObservationNoiseSpec, ObservationTermSpec


def default_robot_observation_terms(add_noise: bool = True) -> tuple[ObservationTermSpec, ...]:
    """Return the task-shared proprioceptive policy terms.

    For G1 23-DoF this layout is 78 values: orientation 6D, body angular
    velocity, joint position/velocity, and the last physically applied action.
    Base linear velocity is deliberately excluded from the actor observation.
    """

    return (
        ObservationTermSpec(
            id="anchor_ori_6d",
            type="anchor_ori_6d",
            noise=ObservationNoiseSpec(-0.05, 0.05) if add_noise else None,
        ),
        ObservationTermSpec(
            id="anchor_ang_vel_b",
            type="anchor_ang_vel_b",
            noise=ObservationNoiseSpec(-0.2, 0.2) if add_noise else None,
        ),
        ObservationTermSpec(
            id="joint_pos",
            type="joint_pos",
            noise=ObservationNoiseSpec(-0.01, 0.01) if add_noise else None,
        ),
        ObservationTermSpec(
            id="joint_vel",
            type="joint_vel",
            noise=ObservationNoiseSpec(-0.5, 0.5) if add_noise else None,
        ),
        ObservationTermSpec(id="previous_action", type="previous_action"),
    )


def build_proprioceptive_context(
    *,
    anchor_quat_w: torch.Tensor,
    anchor_lin_vel_w: torch.Tensor,
    anchor_ang_vel_w: torch.Tensor,
    gravity_vector_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    previous_action: torch.Tensor,
    extras: Mapping[str, torch.Tensor] | None = None,
) -> ObservationContext:
    """Build the simulator-independent context shared by every downstream task."""

    return ObservationContext(
        projected_gravity=quat_apply_inverse(anchor_quat_w, gravity_vector_w),
        anchor_ori_6d=quaternion_to_rotation_6d(anchor_quat_w),
        anchor_lin_vel_b=quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w),
        anchor_ang_vel_b=quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        previous_action=previous_action.clone(),
        anchor_lin_vel=anchor_lin_vel_w,
        extras={} if extras is None else extras,
    )


__all__ = ["build_proprioceptive_context", "default_robot_observation_terms"]

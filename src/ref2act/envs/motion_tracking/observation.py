from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply_inverse

from ref2act.common.math import relative_transform, quaternion_to_tangent_and_normal
from ref2act.common.observation_spec import (
    ObservationComposer,
    ObservationContext,
    ObservationGroupSpec,
    ObservationLayout,
    ObservationNoiseSpec,
    ObservationSpec,
    ObservationTermSpec,
)
from ref2act.common.utils import IndexLike

from .types import MotionState, ReferenceMotions


def _anchor_ang_vel_b(anchor_quat_w: torch.Tensor, anchor_ang_vel_w: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)


def default_training_observation_spec(add_noise: bool = True) -> ObservationSpec:
    robot_terms = (
        ObservationTermSpec(
            id="projected_gravity",
            type="projected_gravity",
            noise=ObservationNoiseSpec(-0.05, 0.05) if add_noise else None,
        ),
        ObservationTermSpec(
            id="anchor_ang_vel_b",
            type="anchor_ang_vel_b",
            noise=ObservationNoiseSpec(-0.3, 0.3) if add_noise else None,
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
    return ObservationSpec(
        groups=(
            ObservationGroupSpec(
                name="motion",
                terms=(
                    ObservationTermSpec(id="target_projected_gravity", type="target_projected_gravity"),
                    ObservationTermSpec(id="target_joint_pos", type="target_joint_pos"),
                    ObservationTermSpec(id="target_joint_vel", type="target_joint_vel"),
                ),
            ),
            ObservationGroupSpec(name="robot", terms=robot_terms),
            ObservationGroupSpec(
                name="privilege",
                terms=(
                    ObservationTermSpec(id="priv_target_joint_pos", type="target_joint_pos"),
                    ObservationTermSpec(id="priv_target_joint_vel", type="target_joint_vel"),
                    ObservationTermSpec(id="target_anchor_lin_vel", type="target_anchor_lin_vel"),
                    ObservationTermSpec(id="target_priv_anchor_ang_vel_b", type="target_anchor_ang_vel_b"),
                    ObservationTermSpec(id="relative_anchor_pos", type="relative_anchor_pos"),
                    ObservationTermSpec(
                        id="relative_anchor_tangent_and_normal",
                        type="relative_anchor_tangent_and_normal",
                    ),
                    ObservationTermSpec(id="relative_key_pos", type="relative_key_pos"),
                    ObservationTermSpec(
                        id="relative_key_tangent_and_normal",
                        type="relative_key_tangent_and_normal",
                    ),
                    ObservationTermSpec(id="anchor_lin_vel", type="anchor_lin_vel"),
                    ObservationTermSpec(id="priv_anchor_ang_vel_b", type="anchor_ang_vel_b"),
                    ObservationTermSpec(id="priv_joint_pos", type="joint_pos"),
                    ObservationTermSpec(id="priv_joint_vel", type="joint_vel"),
                    ObservationTermSpec(id="priv_previous_action", type="previous_action"),
                ),
            ),
        )
    )


@dataclass(frozen=True)
class ObservationPreset:
    spec: ObservationSpec


class Observation:
    def __init__(
        self,
        *,
        spec: ObservationSpec,
        layout: ObservationLayout,
        num_envs: int,
        device: torch.device,
        anchor_body_index: int,
        key_body_indices: list[int],
    ) -> None:
        self.spec = spec
        self.layout = layout
        self.anchor_body_index = anchor_body_index
        self.key_body_indices = key_body_indices
        self.composer = ObservationComposer(spec=spec, layout=layout, num_envs=num_envs, device=device)

    def describe(self):
        return self.spec.describe(self.layout)

    def get_robot_state(self, robot: Articulation, scene: InteractiveScene) -> MotionState:
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        local_body_positions_w = robot.data.body_pos_w - scene.env_origins.unsqueeze(1)
        body_quaternions_w = robot.data.body_quat_w
        body_linear_velocities_w = robot.data.body_lin_vel_w
        body_angular_velocities_w = robot.data.body_ang_vel_w

        anchor_positions = local_body_positions_w[:, self.anchor_body_index]
        anchor_quaternions = body_quaternions_w[:, self.anchor_body_index]
        anchor_linear_velocities = body_linear_velocities_w[:, self.anchor_body_index]
        anchor_angular_velocities = body_angular_velocities_w[:, self.anchor_body_index]

        key_positions = local_body_positions_w[:, self.key_body_indices]
        key_quaternions = body_quaternions_w[:, self.key_body_indices]
        key_linear_velocities = body_linear_velocities_w[:, self.key_body_indices]
        key_angular_velocities = body_angular_velocities_w[:, self.key_body_indices]

        return MotionState(
            joint_pos,
            joint_vel,
            anchor_positions,
            anchor_quaternions,
            anchor_linear_velocities,
            anchor_angular_velocities,
            key_positions,
            key_quaternions,
            key_linear_velocities,
            key_angular_velocities,
        )

    def get_reference_motion_state(
        self,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene,
    ) -> MotionState:
        joint_pos = reference_motion.joint_pos
        joint_vel = reference_motion.joint_vel

        local_body_positions_w = reference_motion.body_positions - scene.env_origins.unsqueeze(1)
        body_quaternions_w = reference_motion.body_quaternions
        body_linear_velocities_w = reference_motion.body_linear_velocities
        body_angular_velocities_w = reference_motion.body_angular_velocities

        anchor_positions = local_body_positions_w[:, self.anchor_body_index]
        anchor_quaternions = body_quaternions_w[:, self.anchor_body_index]
        anchor_linear_velocities = body_linear_velocities_w[:, self.anchor_body_index]
        anchor_angular_velocities = body_angular_velocities_w[:, self.anchor_body_index]

        key_positions = local_body_positions_w[:, self.key_body_indices]
        key_quaternions = body_quaternions_w[:, self.key_body_indices]
        key_linear_velocities = body_linear_velocities_w[:, self.key_body_indices]
        key_angular_velocities = body_angular_velocities_w[:, self.key_body_indices]

        return MotionState(
            joint_pos,
            joint_vel,
            anchor_positions,
            anchor_quaternions,
            anchor_linear_velocities,
            anchor_angular_velocities,
            key_positions,
            key_quaternions,
            key_linear_velocities,
            key_angular_velocities,
        )

    def build_context(
        self,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene,
        last_applied_actions: torch.Tensor,
    ) -> ObservationContext:
        robot_state = self.get_robot_state(robot, scene)
        reference_state = self.get_reference_motion_state(reference_motion, scene)

        gravity_vector = robot.data.GRAVITY_VEC_W

        target_projected_gravity_b = quat_apply_inverse(reference_state.anchor_quat, gravity_vector)
        robot_projected_gravity_b = quat_apply_inverse(robot_state.anchor_quat, gravity_vector)
        target_anchor_ang_vel_b = _anchor_ang_vel_b(reference_state.anchor_quat, reference_state.anchor_ang_vel)
        robot_anchor_ang_vel_b = _anchor_ang_vel_b(robot_state.anchor_quat, robot_state.anchor_ang_vel)

        relative_anchor_pos, relative_anchor_quat = relative_transform(
            robot_state.anchor_pos,
            robot_state.anchor_quat,
            reference_state.anchor_pos,
            reference_state.anchor_quat,
        )
        relative_key_pos, relative_key_quat = relative_transform(
            robot_state.anchor_pos,
            robot_state.anchor_quat,
            robot_state.key_pos,
            robot_state.key_quat,
        )

        return ObservationContext(
            target_projected_gravity=target_projected_gravity_b,
            target_joint_pos=reference_state.joint_pos,
            target_joint_vel=reference_state.joint_vel,
            target_anchor_lin_vel=reference_state.anchor_lin_vel,
            target_anchor_ang_vel_b=target_anchor_ang_vel_b,
            projected_gravity=robot_projected_gravity_b,
            anchor_ang_vel_b=robot_anchor_ang_vel_b,
            joint_pos=robot_state.joint_pos,
            joint_vel=robot_state.joint_vel,
            previous_action=last_applied_actions.clone(),
            relative_anchor_pos=relative_anchor_pos,
            relative_anchor_tangent_and_normal=quaternion_to_tangent_and_normal(relative_anchor_quat),
            relative_key_pos=relative_key_pos.flatten(1),
            relative_key_tangent_and_normal=quaternion_to_tangent_and_normal(relative_key_quat).flatten(1),
            anchor_lin_vel=robot_state.anchor_lin_vel,
        )

    def reset(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene,
        last_applied_actions: torch.Tensor,
    ) -> None:
        device = getattr(robot.data, "device", robot.data.joint_pos.device)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(device=device, dtype=torch.long)
        else:
            env_ids = torch.tensor(list(env_ids), device=device, dtype=torch.long)
        context = self.build_context(robot, reference_motion, scene, last_applied_actions)
        self.composer.reset(env_ids, context)

    def get_default_observation(
        self,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene,
        last_applied_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self.build_context(robot, reference_motion, scene, last_applied_actions)
        return self.composer.compose(context)

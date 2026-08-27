from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from ref2act.common.math import (
    quat_apply_inverse,
    quaternion_to_rotation_6d,
    quaternion_to_tangent_and_normal,
    relative_transform,
)
from ref2act.common.proprioception import build_proprioceptive_context, default_robot_observation_terms
from ref2act.isaac_compat import to_torch
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


def build_observation_context(
    robot_state: MotionState,
    reference_state: MotionState,
    gravity_vector: torch.Tensor,
    previous_action: torch.Tensor,
) -> ObservationContext:
    """Build the shared, simulator-independent observation context."""
    context = build_proprioceptive_context(
        anchor_quat_w=robot_state.anchor_quat,
        anchor_lin_vel_w=robot_state.anchor_lin_vel,
        anchor_ang_vel_w=robot_state.anchor_ang_vel,
        gravity_vector_w=gravity_vector,
        joint_pos=robot_state.joint_pos,
        joint_vel=robot_state.joint_vel,
        previous_action=previous_action,
    )
    target_projected_gravity_b = quat_apply_inverse(reference_state.anchor_quat, gravity_vector)
    target_anchor_ori_6d = quaternion_to_rotation_6d(reference_state.anchor_quat)
    target_anchor_ang_vel_b = quat_apply_inverse(reference_state.anchor_quat, reference_state.anchor_ang_vel)

    motion_anchor_pos_b, motion_quat_b = relative_transform(
        robot_state.anchor_pos,
        robot_state.anchor_quat,
        reference_state.anchor_pos,
        reference_state.anchor_quat,
    )
    body_pos_b, body_quat_b = relative_transform(
        robot_state.anchor_pos,
        robot_state.anchor_quat,
        robot_state.key_pos,
        robot_state.key_quat,
    )
    motion_ori_b = quaternion_to_rotation_6d(motion_quat_b)

    return replace(
        context,
        target_projected_gravity=target_projected_gravity_b,
        target_anchor_ori_6d=target_anchor_ori_6d,
        target_joint_pos=reference_state.joint_pos,
        target_joint_vel=reference_state.joint_vel,
        target_anchor_lin_vel=reference_state.anchor_lin_vel,
        target_anchor_ang_vel_b=target_anchor_ang_vel_b,
        motion_anchor_pos_b=motion_anchor_pos_b,
        motion_ori_b=motion_ori_b,
        motion_anchor_ori_b=motion_ori_b,
        relative_anchor_pos=motion_anchor_pos_b,
        relative_anchor_tangent_and_normal=quaternion_to_tangent_and_normal(motion_quat_b),
        relative_key_pos=body_pos_b.flatten(1),
        relative_key_tangent_and_normal=quaternion_to_tangent_and_normal(body_quat_b).flatten(1),
        body_pos_b=body_pos_b.flatten(1),
        body_ori_b=quaternion_to_rotation_6d(body_quat_b).flatten(1),
    )


def default_training_observation_spec(add_noise: bool = True) -> ObservationSpec:
    robot_terms = default_robot_observation_terms(add_noise=add_noise)
    return ObservationSpec(
        groups=(
            ObservationGroupSpec(
                name="motion",
                terms=(
                    ObservationTermSpec(id="target_joint_pos", type="target_joint_pos"),
                    #ObservationTermSpec(id="target_joint_vel", type="target_joint_vel"),
                    ObservationTermSpec(
                        id="motion_ori_b",
                        type="motion_ori_b",
                        noise=ObservationNoiseSpec(-0.05, 0.05) if add_noise else None,
                    ),
                ),
            ),
            ObservationGroupSpec(name="robot", terms=robot_terms),
            ObservationGroupSpec(
                name="privilege",
                terms=(
                    ObservationTermSpec(id="priv_target_joint_pos", type="target_joint_pos"),
                    ObservationTermSpec(id="priv_target_joint_vel", type="target_joint_vel"),
                    ObservationTermSpec(id="priv_joint_pos", type="joint_pos"),
                    ObservationTermSpec(id="priv_joint_vel", type="joint_vel"),
                    ObservationTermSpec(id="priv_motion_anchor_pos_b", type="motion_anchor_pos_b"),
                    ObservationTermSpec(id="priv_motion_anchor_ori_b", type="motion_anchor_ori_b"),
                    ObservationTermSpec(id="body_pos", type="body_pos"),
                    ObservationTermSpec(id="body_ori", type="body_ori"),
                    ObservationTermSpec(id="priv_anchor_lin_vel_b", type="anchor_lin_vel_b"),
                    ObservationTermSpec(id="priv_anchor_ang_vel_b", type="anchor_ang_vel_b"),
                    ObservationTermSpec(id="priv_previous_action", type="previous_action"),
                ),
            ),
        )
    )


@dataclass
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
        policy_order_joint_indices: torch.Tensor | None = None,
    ) -> None:
        self.spec = spec
        self.layout = layout
        self.anchor_body_index = anchor_body_index
        self.key_body_indices = key_body_indices
        self.policy_order_joint_indices = policy_order_joint_indices
        self.composer = ObservationComposer(spec=spec, layout=layout, num_envs=num_envs, device=device)

    def describe(self):
        return self.spec.describe(self.layout)

    def get_robot_state(self, robot: Articulation, scene: InteractiveScene) -> MotionState:
        joint_pos = to_torch(robot.data.joint_pos)
        joint_vel = to_torch(robot.data.joint_vel)
        if self.policy_order_joint_indices is not None:
            joint_pos = joint_pos[:, self.policy_order_joint_indices]
            joint_vel = joint_vel[:, self.policy_order_joint_indices]

        local_body_positions_w = to_torch(robot.data.body_link_pos_w) - scene.env_origins.unsqueeze(1)
        body_quaternions_w = to_torch(robot.data.body_link_quat_w)
        body_linear_velocities_w = to_torch(robot.data.body_link_lin_vel_w)
        body_angular_velocities_w = to_torch(robot.data.body_link_ang_vel_w)

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
        if self.policy_order_joint_indices is not None:
            joint_pos = joint_pos[:, self.policy_order_joint_indices]
            joint_vel = joint_vel[:, self.policy_order_joint_indices]

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

        return build_observation_context(
            robot_state,
            reference_state,
            to_torch(robot.data.GRAVITY_VEC_W),
            last_applied_actions,
        )

    def reset(
        self,
        env_ids: IndexLike,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene,
        last_applied_actions: torch.Tensor,
    ) -> None:
        device = getattr(robot.data, "device", to_torch(robot.data.joint_pos).device)
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

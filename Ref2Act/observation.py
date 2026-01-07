import torch
from dataclasses import dataclass

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply_inverse
from .motion_lib import ReferenceMotions
from .math import relative_transform, quaternion_to_tangent_and_normal, quat_diff

@dataclass
class MotionState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    anchor_pos: torch.Tensor
    anchor_quat: torch.Tensor
    anchor_lin_vel: torch.Tensor
    anchor_ang_vel: torch.Tensor
    key_pos: torch.Tensor
    key_quat: torch.Tensor
    key_lin_vel: torch.Tensor
    key_ang_vel: torch.Tensor

    def __post_init__(self):
        rel_pos, rel_quat = relative_transform(
            self.anchor_pos, self.anchor_quat,
            self.key_pos, self.key_quat
        )

        self.rel_key_pos = rel_pos
        self.rel_key_quat = rel_quat


def add_noise_to(data: torch.Tensor, noise_scale: float):
    data += torch.rand_like(data) * noise_scale

    return data

class Observation:
    def __init__(self, robot_anchor_body_indices: list[int], robot_key_body_indices: list[int],
                 motion_anchor_body_indices: list[int], motion_key_body_indices: list[int]) -> None:
        self.robot_anchor_body_indices = robot_anchor_body_indices
        self.robot_key_body_indices = robot_key_body_indices
        self.motion_anchor_body_indices = motion_anchor_body_indices
        self.motion_key_body_indices = motion_key_body_indices
    
    def default_student_observation(self, robot: Articulation, last_action: torch.Tensor,
                                  reference_motion: ReferenceMotions, add_noise:bool=False) -> torch.Tensor:
        reference_joint_pos = reference_motion.joint_pos
        reference_joint_vel = reference_motion.joint_vel
        reference_anchor_quaternions = reference_motion.body_quaternions[:, self.motion_anchor_body_indices]

        reference_anchor_orient = quat_apply_inverse(
            reference_anchor_quaternions, robot.data.GRAVITY_VEC_W
        ).flatten(start_dim=1)
        
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        anchor_angular_vel = robot.data.root_ang_vel_w
        anchor_graity_orient = robot.data.projected_gravity_b

        if add_noise:
            joint_pos = add_noise_to(joint_pos, 0.01)
            joint_vel = add_noise_to(joint_vel, 0.5)
            anchor_angular_vel = add_noise_to(anchor_angular_vel, 0.2)
            anchor_graity_orient = add_noise_to(anchor_graity_orient, 0.05)

        obs = torch.cat(
            [
                reference_joint_pos,
                reference_joint_vel,
                reference_anchor_orient,
                joint_pos,
                joint_vel,
                anchor_angular_vel,
                anchor_graity_orient,
                last_action,
            ],
            dim=-1,
        )

        return obs
    
    def get_robot_state(self, robot: Articulation, scene: InteractiveScene):
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        local_body_positions = robot.data.body_link_pos_w - scene.env_origins.unsqueeze(1)
        local_body_quaternions = robot.data.body_link_quat_w
        body_linear_velocities = robot.data.body_lin_vel_w
        body_angular_velocities = robot.data.body_ang_vel_w

        anchor_positions = local_body_positions[:, self.robot_anchor_body_indices]
        anchor_quaternions = local_body_quaternions[:, self.robot_anchor_body_indices]
        anchor_linear_velocities = body_linear_velocities[:, self.robot_anchor_body_indices]
        anchor_angular_velocities = body_angular_velocities[:, self.robot_anchor_body_indices]

        key_positions = local_body_positions[:, self.robot_key_body_indices]
        key_quaternions = local_body_quaternions[:, self.robot_key_body_indices]
        key_linear_velocities = body_linear_velocities[:, self.robot_key_body_indices]
        key_angular_velocities = body_angular_velocities[:, self.robot_key_body_indices]

        robot_motion_state = MotionState(joint_pos, joint_vel, anchor_positions, anchor_quaternions,
                                         anchor_linear_velocities, anchor_angular_velocities,
                                         key_positions, key_quaternions,
                                         key_linear_velocities, key_angular_velocities)

        return robot_motion_state

    def get_reference_motion_state(
        self,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene
    ):
        joint_pos = reference_motion.joint_pos
        joint_vel = reference_motion.joint_vel

        local_body_positions = reference_motion.body_positions - scene.env_origins.unsqueeze(1)
        local_body_quaternions = reference_motion.body_quaternions
        body_linear_velocities = reference_motion.body_linear_velocities
        body_angular_velocities = reference_motion.body_angular_velocities

        anchor_positions = local_body_positions[:, self.motion_anchor_body_indices]
        anchor_quaternions = local_body_quaternions[:, self.motion_anchor_body_indices]
        anchor_linear_velocities = body_linear_velocities[:, self.motion_anchor_body_indices]
        anchor_angular_velocities = body_angular_velocities[:, self.motion_anchor_body_indices]

        key_positions = local_body_positions[:, self.motion_key_body_indices]
        key_quaternions = local_body_quaternions[:, self.motion_key_body_indices]
        key_linear_velocities = body_linear_velocities[:, self.motion_key_body_indices]
        key_angular_velocities = body_angular_velocities[:, self.motion_key_body_indices]

        reference_motion_state = MotionState(joint_pos, joint_vel, anchor_positions, anchor_quaternions,
                                         anchor_linear_velocities, anchor_angular_velocities,
                                         key_positions, key_quaternions,
                                         key_linear_velocities, key_angular_velocities)
        
        return reference_motion_state
    
    def default_teacher_observation(
        self,
        robot: Articulation,
        scene: InteractiveScene,
        reference_motion: ReferenceMotions
    ) -> torch.Tensor:
        
        robot_motion_state = self.get_robot_state(robot, scene)
        reference_motion_state = self.get_reference_motion_state(reference_motion, scene)

        diff_anchor_pos = robot_motion_state.anchor_pos - reference_motion_state.anchor_pos
        diff_anchor_quat = quat_diff(robot_motion_state.anchor_quat, reference_motion_state.anchor_quat)
        diff_anchor_lin_vel = robot_motion_state.anchor_lin_vel - reference_motion_state.anchor_lin_vel
        diff_anchor_ang_vel = robot_motion_state.anchor_ang_vel - reference_motion_state.anchor_ang_vel

        rel_key_pos, rel_key_quat = relative_transform(robot_motion_state.anchor_pos, robot_motion_state.anchor_quat,
                                                       reference_motion_state.key_pos, reference_motion_state.key_quat)
        


        obs = torch.cat(
            [
                robot_motion_state.anchor_pos.flatten(1),
                quaternion_to_tangent_and_normal(robot_motion_state.anchor_quat).flatten(1),
                robot_motion_state.anchor_lin_vel.flatten(1),
                robot_motion_state.anchor_ang_vel.flatten(1),
                robot_motion_state.rel_key_pos.flatten(1),
                quaternion_to_tangent_and_normal(robot_motion_state.rel_key_quat).flatten(1),
                robot_motion_state.key_lin_vel.flatten(1),
                robot_motion_state.key_ang_vel.flatten(1),
                diff_anchor_pos.flatten(1),
                quaternion_to_tangent_and_normal(diff_anchor_quat).flatten(1),
                diff_anchor_lin_vel.flatten(1),
                diff_anchor_ang_vel.flatten(1),
                rel_key_pos.flatten(1),
                quaternion_to_tangent_and_normal(rel_key_quat).flatten(1)
            ],
            dim=-1
        )
        
        return obs
    
    def default_motion_observation(
        self,
        motion_sample: ReferenceMotions,
        scene: InteractiveScene
    ) -> torch.Tensor:
        
        joint_positions = motion_sample.joint_pos
        joint_velocities = motion_sample.joint_vel

        anchor_positions = motion_sample.body_positions[:, self.motion_anchor_body_indices] - scene.terrain.env_origins
        anchor_quaternions = motion_sample.body_quaternions[:, self.motion_anchor_body_indices]

        anchor_linear_velocities = motion_sample.body_linear_velocities[:, self.motion_anchor_body_indices]
        anchor_angular_velocities = motion_sample.body_angular_velocities[:, self.motion_anchor_body_indices]

        key_positions = motion_sample.body_positions[:, self.motion_key_body_indices]
        key_quaternions = motion_sample.body_quaternions[:, self.motion_key_body_indices]

        rel_key_positions, rel_key_quaternions = relative_transform(
            anchor_positions,
            anchor_quaternions,
            key_positions,
            key_quaternions,
        )

        obs = torch.cat(
            [
                joint_positions,
                joint_velocities,
                anchor_positions.flatten(start_dim=1),
                anchor_quaternions.flatten(start_dim=1),
                anchor_linear_velocities.flatten(start_dim=1),
                anchor_angular_velocities.flatten(start_dim=1),
                rel_key_positions.flatten(start_dim=1),
                rel_key_quaternions.flatten(start_dim=1),
            ],
            dim=-1,
        )

        return obs

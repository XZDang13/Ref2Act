import torch
from dataclasses import dataclass
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_inv
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
    def __init__(self, anchor_body_indices: list[int], key_body_indices: list[int], add_noise:bool=False) -> None:
        self.anchor_body_indices = anchor_body_indices
        self.key_body_indices = key_body_indices
        self.add_noise = add_noise

    
    def get_robot_state(self, robot: Articulation, scene: InteractiveScene):
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel

        local_body_positions = robot.data.body_link_pos_w - scene.env_origins.unsqueeze(1)
        local_body_quaternions = robot.data.body_link_quat_w
        body_linear_velocities = robot.data.body_lin_vel_w
        body_angular_velocities = robot.data.body_ang_vel_w

        anchor_positions = local_body_positions[:, self.anchor_body_indices]
        anchor_quaternions = local_body_quaternions[:, self.anchor_body_indices]
        anchor_linear_velocities = body_linear_velocities[:, self.anchor_body_indices]
        anchor_angular_velocities = body_angular_velocities[:, self.anchor_body_indices]

        key_positions = local_body_positions[:, self.key_body_indices]
        key_quaternions = local_body_quaternions[:, self.key_body_indices]
        key_linear_velocities = body_linear_velocities[:, self.key_body_indices]
        key_angular_velocities = body_angular_velocities[:, self.key_body_indices]

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

        anchor_positions = local_body_positions[:, self.anchor_body_indices]
        anchor_quaternions = local_body_quaternions[:, self.anchor_body_indices]
        anchor_linear_velocities = body_linear_velocities[:, self.anchor_body_indices]
        anchor_angular_velocities = body_angular_velocities[:, self.anchor_body_indices]

        key_positions = local_body_positions[:, self.key_body_indices]
        key_quaternions = local_body_quaternions[:, self.key_body_indices]
        key_linear_velocities = body_linear_velocities[:, self.key_body_indices]
        key_angular_velocities = body_angular_velocities[:, self.key_body_indices]

        reference_motion_state = MotionState(joint_pos, joint_vel, anchor_positions, anchor_quaternions,
                                         anchor_linear_velocities, anchor_angular_velocities,
                                         key_positions, key_quaternions,
                                         key_linear_velocities, key_angular_velocities)
        
        return reference_motion_state
    
    def get_default_observation(self, robot: Articulation, reference_motion: ReferenceMotions,
                                scene: InteractiveScene, last_applied_actions: torch.Tensor) -> dict[str, torch.Tensor]:
        
        obs = {}

        robot_state = self.get_robot_state(robot, scene)
        reference_state = self.get_reference_motion_state(reference_motion, scene)

        obs["policy"] = self.get_policy_observation(robot_state, reference_state, last_applied_actions)
        obs["critic"] = self.get_critic_observation(robot_state, reference_state, last_applied_actions)
        
        return obs
    
    def get_policy_observation(self,
                               robot_state: MotionState,
                               reference_state: MotionState,
                               last_applied_action:torch.Tensor) -> torch.Tensor:
        
        target_joint_pos = reference_state.joint_pos
        target_jiont_vel = robot_state.joint_vel

        target_quat = reference_state.anchor_quat
        robot_quat = robot_state.anchor_quat

        robot_ang_vel = robot_state.anchor_ang_vel.flatten(1)
        robot_joint_pos = robot_state.joint_pos
        robot_joint_vel = robot_state.joint_vel
        last_action = last_applied_action.clone()

        if self.add_noise:
            robot_quat += torch.rand_like(robot_quat) * 0.05
            robot_ang_vel += torch.rand_like(robot_ang_vel) * 0.3
            robot_joint_pos += torch.rand_like(robot_joint_pos) * 0.01
            robot_joint_vel += torch.rand_like(robot_joint_vel) * 0.5

        robot_quat_inv = quat_inv(robot_quat)
        relative_quat = quat_mul(robot_quat_inv, target_quat)

        relative_tangent_and_normal = quaternion_to_tangent_and_normal(relative_quat).flatten(1)

        obs = torch.cat(
            [
                target_joint_pos,
                target_jiont_vel,
                relative_tangent_and_normal,
                robot_ang_vel,
                robot_joint_pos,
                robot_joint_vel,
                last_action
            ], dim=-1
        )

        return obs
    
    def get_critic_observation(self,
                               robot_state: MotionState,
                               reference_state: MotionState,
                               last_applied_action:torch.Tensor) -> torch.Tensor:
        
        target_joint_pos = reference_state.joint_pos
        target_jiont_vel = robot_state.joint_vel

        relative_anchor_pos, relative_anchor_quat = relative_transform(robot_state.anchor_ang_vel,
                                                                       robot_state.anchor_quat,
                                                                       reference_state.anchor_pos,
                                                                       reference_state.anchor_quat)
        
        relative_anchor_pos = relative_anchor_pos.flatten(1)
        relative_anchor_tangent_and_normal = quaternion_to_tangent_and_normal(relative_anchor_quat).flatten(1)

        relative_key_pos, relative_key_quat = relative_transform(robot_state.anchor_pos,
                                                                 robot_state.anchor_quat,
                                                                 robot_state.key_pos,
                                                                 robot_state.key_quat)
        
        relative_key_pos = relative_key_pos.flatten(1)
        relative_key_tangent_and_normal = quaternion_to_tangent_and_normal(relative_key_quat).flatten(1)
        
        anchor_lin_vel = robot_state.anchor_lin_vel.flatten(1)
        anchor_ang_vel = robot_state.anchor_ang_vel.flatten(1)

        joint_pos = robot_state.joint_pos
        joint_vel = robot_state.joint_vel

        last_action = last_applied_action.clone()

        obs = torch.cat(
            [
                target_joint_pos,
                target_jiont_vel,
                relative_anchor_pos,
                relative_anchor_tangent_and_normal,
                relative_key_pos,
                relative_key_tangent_and_normal,
                anchor_lin_vel,
                anchor_ang_vel,
                joint_pos,
                joint_vel,
                last_action
            ], dim=-1
        )

        return obs
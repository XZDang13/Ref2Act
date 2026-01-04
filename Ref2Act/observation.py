import torch

from isaaclab.assets import Articulation
from .motion_lib import ReferenceMotions
from .math import relative_transform

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
    
    def default_robot_observation(self, robot: Articulation, last_action: torch.Tensor, add_noise:bool=False) -> torch.Tensor:
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
                joint_pos,
                joint_vel,
                anchor_angular_vel,
                anchor_graity_orient,
                last_action,
            ],
            dim=-1,
        )

        return obs
    
    def default_robot_privilege_observation(
        self,
        robot: Articulation,
    ) -> torch.Tensor:
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        
        body_positions = robot.data.body_link_pos_w
        body_quaternions = robot.data.body_link_quat_w
        body_linear_velocities = robot.data.body_lin_vel_w
        body_angular_velocities = robot.data.body_ang_vel_w

        anchor_positions = body_positions[:, self.robot_anchor_body_indices]
        anchor_quaternions = body_quaternions[:, self.robot_anchor_body_indices]

        anchor_linear_velocities = body_linear_velocities[:, self.robot_anchor_body_indices]
        anchor_angular_velocities = body_angular_velocities[:, self.robot_anchor_body_indices]

        key_positions = body_positions[:, self.robot_key_body_indices]
        key_quaternions = body_quaternions[:, self.robot_key_body_indices]

        rel_key_positions, rel_key_quaternions = relative_transform(
            anchor_positions,
            anchor_quaternions,
            key_positions,
            key_quaternions,
        )

        obs = torch.cat(
            [
                joint_pos,
                joint_vel,
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
    
    def default_motion_observation(
        self,
        motion_sample: ReferenceMotions,
    ) -> torch.Tensor:
        
        joint_positions = motion_sample.joint_pos
        joint_velocities = motion_sample.joint_vel

        anchor_positions = motion_sample.body_positions[:, self.motion_anchor_body_indices]
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
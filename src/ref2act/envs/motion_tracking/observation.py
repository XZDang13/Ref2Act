import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply_inverse
from ref2act.common.math import relative_transform, quaternion_to_tangent_and_normal

from .types import MotionState, ReferenceMotions


def add_noise_to(data: torch.Tensor, noise_scale: float):
    data += torch.rand_like(data) * noise_scale

    return data


def _anchor_ang_vel_b(anchor_quat_w: torch.Tensor, anchor_ang_vel_w: torch.Tensor) -> torch.Tensor:
    """Project anchor angular velocity from world axes into the anchor/body frame."""
    return quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)


class Observation:
    def __init__(self, anchor_body_index: int, key_body_indices: list[int], add_noise:bool=False) -> None:
        self.anchor_body_index = anchor_body_index
        self.key_body_indices = key_body_indices
        self.add_noise = add_noise

    
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

        robot_motion_state = MotionState(joint_pos, joint_vel, anchor_positions, anchor_quaternions,
                                         anchor_linear_velocities, anchor_angular_velocities,
                                         key_positions, key_quaternions,
                                         key_linear_velocities, key_angular_velocities)

        return robot_motion_state

    def get_reference_motion_state(
        self,
        reference_motion: ReferenceMotions,
        scene: InteractiveScene
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

        motion_obs, robot_obs = self.get_policy_observation(robot_state, reference_state, robot.data.GRAVITY_VEC_W, last_applied_actions)
        privilege_obs = self.get_critic_observation(robot_state, reference_state, last_applied_actions)
        
        obs["motion"] = motion_obs
        obs["robot"] = robot_obs
        obs["privilege"] = privilege_obs

        return obs
    
    def get_policy_observation(self,
                               robot_state: MotionState,
                               reference_state: MotionState,
                               gravity_vector: torch.Tensor,
                               last_applied_action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        target_joint_pos = reference_state.joint_pos
        target_jiont_vel = reference_state.joint_vel

        target_anchor_quat_w = reference_state.anchor_quat
        robot_anchor_quat_w = robot_state.anchor_quat

        target_projected_gravity_b = quat_apply_inverse(target_anchor_quat_w, gravity_vector)
        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat_w, gravity_vector)
        robot_anchor_ang_vel_w = robot_state.anchor_ang_vel

        # Clone tensors before injecting observation noise so we do not mutate
        # the live robot state or contaminate privileged observations.
        robot_anchor_ang_vel_b = _anchor_ang_vel_b(robot_anchor_quat_w, robot_anchor_ang_vel_w).clone()
        robot_joint_pos = robot_state.joint_pos.clone()
        robot_joint_vel = robot_state.joint_vel.clone()
        last_action = last_applied_action.clone()

        if self.add_noise:
            robot_projected_gravity_b += torch.empty_like(robot_projected_gravity_b).uniform_(-0.05, 0.05)
            robot_anchor_ang_vel_b += torch.empty_like(robot_anchor_ang_vel_b).uniform_(-0.3, 0.3)
            robot_joint_pos += torch.empty_like(robot_joint_pos).uniform_(-0.01, 0.01)
            robot_joint_vel += torch.empty_like(robot_joint_vel).uniform_(-0.5, 0.5)

        #print(target_joint_pos[0])
        #print(target_jiont_vel[0])
        #print(target_projected_gravity[0])

        #print(robot_projected_gravity)
        #print(robot_ang_vel)
        #print(robot_joint_pos)
        #print("Joint Vel:")
        #print(robot_joint_vel)
        #print(last_action)
        #print("--------------------")
        motion_obs = torch.cat(
            [
                target_projected_gravity_b,
                target_joint_pos,
                target_jiont_vel
            ], dim=-1
        )

        robot_obs = torch.cat(
            [
                robot_projected_gravity_b,
                robot_anchor_ang_vel_b,
                robot_joint_pos,
                robot_joint_vel,
                last_action
            ], dim=-1
        )


        return motion_obs, robot_obs
    
    def get_critic_observation(self,
                               robot_state: MotionState,
                               reference_state: MotionState,
                               last_applied_action:torch.Tensor) -> torch.Tensor:
        
        target_joint_pos = reference_state.joint_pos
        target_jiont_vel = reference_state.joint_vel

        relative_anchor_pos, relative_anchor_quat = relative_transform(robot_state.anchor_pos,
                                                                       robot_state.anchor_quat,
                                                                       reference_state.anchor_pos,
                                                                       reference_state.anchor_quat)
        
        relative_anchor_pos = relative_anchor_pos
        relative_anchor_tangent_and_normal = quaternion_to_tangent_and_normal(relative_anchor_quat)

        relative_key_pos, relative_key_quat = relative_transform(robot_state.anchor_pos,
                                                                 robot_state.anchor_quat,
                                                                 robot_state.key_pos,
                                                                 robot_state.key_quat)
        
        relative_key_pos = relative_key_pos.flatten(1)
        relative_key_tangent_and_normal = quaternion_to_tangent_and_normal(relative_key_quat).flatten(1)
        
        anchor_lin_vel_w = robot_state.anchor_lin_vel
        anchor_ang_vel_w = robot_state.anchor_ang_vel
        anchor_ang_vel_b = _anchor_ang_vel_b(robot_state.anchor_quat, anchor_ang_vel_w)

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
                anchor_lin_vel_w,
                anchor_ang_vel_b,
                joint_pos,
                joint_vel,
                last_action
            ], dim=-1
        )

        return obs

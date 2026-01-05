import dataclasses
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.sensors import ContactSensor
from .motion_lib import ReferenceMotions
from .math import relative_transform
from .utils import IndexLike

@dataclasses.dataclass
class PenaltyRewardCfg:
    collision_track_body_indices: list[int] = dataclasses.MISSING
    joint_acc_weight:float = -2.5e-7
    joint_torque_wegiht:float = -1e-5
    joint_limit_weight:float = -1e-1
    self_collision_weight:float = -1.0
    self_collision_force_threshold:float = 10.0

@dataclasses.dataclass
class MimicRewardsCfg:
    robot_anchor_body_indices:list[int] = dataclasses.MISSING
    robot_key_body_indices:list[int] = dataclasses.MISSING
    motion_anchor_body_indices:list[int] = dataclasses.MISSING
    motion_key_body_indices: list[int] = dataclasses.MISSING

    position_std:float = 0.3 ** 2
    quaternion_std:float = 0.4 ** 2
    linear_vel_std:float = 1.0 ** 2
    ang_vel_std:float = 3.14 ** 2
    joint_position_std:float = 0.5 ** 2
    joint_vel_std:float = 1.0 ** 2

    mimic_anchor_position_weight:float = 0.5
    mimic_anchor_quaternion_weight:float = 0.5
    mimic_key_position_wegiht:float = 1.0
    mimic_key_quaternion_weight:float = 1.0
    mimic_key_linear_vel_weight:float = 1.0
    mimic_key_ang_vel_weight:float = 1.0
    mimic_joint_position_weight:float = 1.0
    mimic_joint_vel_weight:float = 1.0

    anchor_height_only:bool = True

@dataclasses.dataclass
class AMPRewardsCfg:
    discriminator_reward_scale: float = 2.0
    discriminator_reward_weight: float = 1.0

@dataclasses.dataclass
class RewardsCfg:
    # Mimic reward indices.
    robot_anchor_body_indices: list[int] = dataclasses.MISSING
    robot_key_body_indices: list[int] = dataclasses.MISSING
    motion_anchor_body_indices: list[int] = dataclasses.MISSING
    motion_key_body_indices: list[int] = dataclasses.MISSING
    # Penalty reward indices.
    collision_track_body_indices: list[int] = dataclasses.MISSING
    # Shared options.
    return_vector: bool = False
    # Mimic reward weights.
    position_std:float = 0.3 ** 2
    quaternion_std:float = 0.4 ** 2
    linear_vel_std:float = 1.0 ** 2
    ang_vel_std:float = 3.14 ** 2
    joint_position_std:float = 0.5 ** 2
    joint_vel_std:float = 1.0 ** 2

    mimic_anchor_position_weight:float = 0.5
    mimic_anchor_quaternion_weight:float = 0.5
    mimic_key_position_wegiht:float = 1.0
    mimic_key_quaternion_weight:float = 1.0
    mimic_key_linear_vel_weight:float = 1.0
    mimic_key_ang_vel_weight:float = 1.0
    mimic_joint_position_weight:float = 1.0
    mimic_joint_vel_weight:float = 1.0
    anchor_height_only:bool = True
    # Penalty reward weights.
    joint_acc_weight:float = -2.5e-7
    joint_torque_wegiht:float = -1e-5
    joint_limit_weight:float = -1e-1
    self_collision_weight:float = -0.1
    self_collision_force_threshold:float = 1.0

class Rewards:
    def __init__(self, cfg:RewardsCfg):
        self.cfg = cfg
        penalty_cfg = PenaltyRewardCfg(
            collision_track_body_indices=cfg.collision_track_body_indices,
            joint_acc_weight=cfg.joint_acc_weight,
            joint_torque_wegiht=cfg.joint_torque_wegiht,
            joint_limit_weight=cfg.joint_limit_weight,
            self_collision_weight=cfg.self_collision_weight,
            self_collision_force_threshold=cfg.self_collision_force_threshold,
        )
        mimic_cfg = MimicRewardsCfg(
            robot_anchor_body_indices=cfg.robot_anchor_body_indices,
            robot_key_body_indices=cfg.robot_key_body_indices,
            motion_anchor_body_indices=cfg.motion_anchor_body_indices,
            motion_key_body_indices=cfg.motion_key_body_indices,
            position_std=cfg.position_std,
            quaternion_std=cfg.quaternion_std,
            linear_vel_std=cfg.linear_vel_std,
            ang_vel_std=cfg.ang_vel_std,
            joint_position_std=cfg.joint_position_std,
            joint_vel_std = cfg.joint_vel_std,
            mimic_anchor_position_weight=cfg.mimic_anchor_position_weight,
            mimic_anchor_quaternion_weight=cfg.mimic_anchor_quaternion_weight,
            mimic_key_position_wegiht=cfg.mimic_key_position_wegiht,
            mimic_key_quaternion_weight=cfg.mimic_key_quaternion_weight,
            mimic_key_linear_vel_weight=cfg.mimic_key_linear_vel_weight,
            mimic_key_ang_vel_weight=cfg.mimic_key_ang_vel_weight,
            mimic_joint_position_weight=cfg.mimic_joint_position_weight,
            mimic_joint_vel_weight=cfg.mimic_joint_vel_weight,
            anchor_height_only=cfg.anchor_height_only,
        )
        self.regulation_reward = RegulationReward(penalty_cfg)
        self.mimic_reward = MimicRewards(mimic_cfg)

    def get_task_reward(
        self,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        contact_sensor: ContactSensor,
    ) -> torch.Tensor:
        regulation_reward = self.regulation_reward.get_reward(robot, contact_sensor)
        mimic_reward = self.mimic_reward.get_reward(robot, reference_motion)
        reward_vector = torch.cat([regulation_reward, mimic_reward], dim=-1)
        if self.cfg.return_vector:
            return reward_vector
        return reward_vector.sum(-1)

class RegulationReward:
    def __init__(self, cfg:PenaltyRewardCfg):
        self.cfg = cfg

    def get_reward(self, robot:Articulation, contact_sensor: ContactSensor) -> torch.Tensor:
        joint_acc_penalty = self.joint_acc_l2(robot) * self.cfg.joint_acc_weight
        joint_torque_penalty = self.joint_torque_l2(robot) * self.cfg.joint_torque_wegiht
        joint_limit_penalty = self.joint_limit(robot) * self.cfg.joint_limit_weight
        self_collision_penalty = self.self_collision_penalty(
            contact_sensor,
            self.cfg.collision_track_body_indices,
            self.cfg.self_collision_force_threshold,
        ) * self.cfg.self_collision_weight
        
        reward = torch.stack(
            [joint_acc_penalty, joint_torque_penalty, joint_limit_penalty, self_collision_penalty],
            dim=-1
        )

        return reward
    
    def feet_contact_time(self, sensor: ContactSensor, step_dt:float, physics_dt: float,
                          body_ids: IndexLike, threshold: float) -> torch.Tensor:
        first_air = sensor.compute_first_air(step_dt, physics_dt)[:, body_ids]
        last_contact_time = sensor.data.last_contact_time[:, body_ids]
        reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)

        return reward

    def self_collision_penalty(
        self,
        sensor: ContactSensor,
        body_ids: list[int],
        threshold: float,
    ) -> torch.Tensor:
        net_contact_forces = sensor.data.net_forces_w_history
        if len(body_ids) == 0:
            return torch.zeros(net_contact_forces.shape[0], device=net_contact_forces.device)
        is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
        return torch.sum(is_contact, dim=1)
    
    def joint_acc_l2(self, robot: Articulation):
        return torch.sum(torch.square(robot.data.joint_acc), dim=1)
    
    def joint_torque_l2(self, robot: Articulation):
        return torch.sum(torch.square(robot.data.applied_torque), dim=1)
    
    def joint_limit(self, robot: Articulation):
        out_of_limits = -(
            robot.data.joint_pos[:, :] - robot.data.soft_joint_pos_limits[:, :, 0]
        ).clip(max=0.0)
        out_of_limits += (
            robot.data.joint_pos[:, :] - robot.data.soft_joint_pos_limits[:, :, 1]
        ).clip(min=0.0)

        return torch.sum(out_of_limits, dim=1)

class MimicRewards:
    def __init__(self, cfg:MimicRewardsCfg) -> None:
        self.cfg = cfg

    def get_reward(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        
        anchor_position_error, anchor_quaternion_error = self.anchor_body_pose_error(robot, reference_motion)
        key_position_error, key_quaternion_error = self.key_body_pose_error(robot, reference_motion,)
        key_linear_vel_error, key_ang_vel_error = self.key_body_state_error(robot, reference_motion)
        joint_position_error, joint_vel_error = self.joint_state_error(robot, reference_motion)

        anchor_position_reward = torch.exp(-anchor_position_error / self.cfg.position_std) * self.cfg.mimic_anchor_position_weight
        anchor_quaternion_reward = torch.exp(-anchor_quaternion_error / self.cfg.quaternion_std) * self.cfg.mimic_anchor_quaternion_weight
        key_position_reward = torch.exp(-key_position_error / self.cfg.position_std) * self.cfg.mimic_key_position_wegiht
        key_quaternion_reward = torch.exp(-key_quaternion_error / self.cfg.quaternion_std) * self.cfg.mimic_key_quaternion_weight
        key_linear_vel_reward = torch.exp(-key_linear_vel_error / self.cfg.linear_vel_std) * self.cfg.mimic_key_linear_vel_weight
        key_ang_vel_reward = torch.exp(-key_ang_vel_error / self.cfg.ang_vel_std) * self.cfg.mimic_key_ang_vel_weight
        joint_position_reward = torch.exp(-joint_position_error / 1.0)
        joint_vel_reward = torch.exp(-joint_vel_error / 1)


        reward = torch.stack([anchor_position_reward, anchor_quaternion_reward, key_position_reward,
                                key_quaternion_reward, key_linear_vel_reward, key_ang_vel_reward],
                                dim=-1
                            )
        
        return reward

    def anchor_body_pose_error(self, robot: Articulation, reference_motion: ReferenceMotions):
        position_slice = slice(None)
        if self.cfg.anchor_height_only:
            position_slice = slice(2, 3)

        robot_anchor_body_positions = robot.data.body_pos_w[:, self.cfg.robot_anchor_body_indices, position_slice]
        robot_anchor_body_quaternions = robot.data.body_quat_w[:, self.cfg.robot_anchor_body_indices]

        reference_anchor_body_positions = reference_motion.body_positions[:, self.cfg.motion_anchor_body_indices, position_slice]
        reference_anchor_body_quaternions = reference_motion.body_quaternions[:, self.cfg.motion_anchor_body_indices]

        position_error = (robot_anchor_body_positions-reference_anchor_body_positions).square().sum(-1).mean(-1)
        quaternion_error = quat_error_magnitude(robot_anchor_body_quaternions, reference_anchor_body_quaternions).square().mean(-1)

        return position_error, quaternion_error

    
    def key_body_pose_error(self, robot: Articulation, reference_motion: ReferenceMotions):

        robot_anchor_body_positions = robot.data.body_pos_w[:, self.cfg.robot_anchor_body_indices]
        robot_anchor_body_quaternions = robot.data.body_quat_w[:, self.cfg.robot_anchor_body_indices]

        robot_key_body_positions = robot.data.body_pos_w[:, self.cfg.robot_key_body_indices]
        robot_key_body_quaternions = robot.data.body_quat_w[:, self.cfg.robot_key_body_indices]
        
        robot_key_body_relative_positions, robot_key_body_relative_quaternions = relative_transform(
            robot_anchor_body_positions, robot_anchor_body_quaternions,
            robot_key_body_positions, robot_key_body_quaternions
        ) 

        reference_anchor_body_positions = reference_motion.body_positions[:, self.cfg.motion_anchor_body_indices]
        reference_anchor_body_quaternions = reference_motion.body_quaternions[:, self.cfg.motion_anchor_body_indices]
        reference_key_body_positions = reference_motion.body_positions[:, self.cfg.motion_key_body_indices]
        reference_key_body_quaternions = reference_motion.body_quaternions[:, self.cfg.motion_key_body_indices]

        reference_key_body_relative_positions, reference_key_body_relative_quaternions = relative_transform(
            reference_anchor_body_positions, reference_anchor_body_quaternions,
            reference_key_body_positions, reference_key_body_quaternions
        )

        position_error = (robot_key_body_relative_positions-reference_key_body_relative_positions).square().sum(-1).mean(-1)
        quaternion_error = quat_error_magnitude(robot_key_body_relative_quaternions, reference_key_body_relative_quaternions).square().mean(-1)
        

        return position_error, quaternion_error
    
    def key_body_state_error(self, robot: Articulation, reference_motion: ReferenceMotions):
        robot_key_body_lin_vel = robot.data.body_lin_vel_w[:, self.cfg.robot_key_body_indices]
        robot_key_body_ang_vel = robot.data.body_ang_vel_w[:, self.cfg.robot_key_body_indices]

        reference_key_body_lin_vel = reference_motion.body_linear_velocities[:, self.cfg.motion_key_body_indices]
        reference_key_body_ang_vel = reference_motion.body_angular_velocities[:, self.cfg.motion_key_body_indices]

        lin_vel_error = (robot_key_body_lin_vel-reference_key_body_lin_vel).square().sum(-1).mean(-1)
        ang_vel_error = (robot_key_body_ang_vel-reference_key_body_ang_vel).square().sum(-1).mean(-1)

        return lin_vel_error, ang_vel_error 
    
    def joint_state_error(self, robot: Articulation, reference_motion: ReferenceMotions):
        robot_joint_pos = robot.data.joint_pos
        robot_joint_vel = robot.data.joint_vel

        reference_joint_pos = reference_motion.joint_pos
        reference_joint_vel = reference_motion.joint_vel

        pos_error = (robot_joint_pos-reference_joint_pos).square().sum(-1)
        vel_error = (robot_joint_vel-reference_joint_vel).square().sum(-1)

        return pos_error, vel_error

class AMPReward:
    def __init__(self, cfg: AMPRewardsCfg):
        self.cfg = cfg

    def get_rewards(self, logits: torch.Tensor) -> torch.Tensor:
        rewards = torch.nn.functional.softplus(logits)

        return rewards

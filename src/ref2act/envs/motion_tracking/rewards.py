import dataclasses
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat
from isaaclab.sensors import ContactSensor
from ref2act.common.utils import IndexLike

from .action import ActionProcessor
from .types import ReferenceMotions

@dataclasses.dataclass
class PenaltyRewardCfg:
    collision_track_body_indices: list[int] = dataclasses.MISSING
    foot_body_indices: list[int] = dataclasses.MISSING
    foot_contact_body_indices: list[int] = dataclasses.MISSING
    joint_acc_weight:float = -2.5e-7
    joint_torque_wegiht:float = -1e-5
    joint_limit_weight:float = -10.0
    self_collision_weight:float = -1.0
    self_collision_force_threshold:float = 10.0
    foot_slip_weight: float = -0.1
    foot_slip_force_threshold: float = 1.0
    action_rate_weight:float = -1e-2

@dataclasses.dataclass
class MimicRewardsCfg:
    anchor_body_index:int = dataclasses.MISSING
    key_body_indices:list[int] = dataclasses.MISSING

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
    mimic_joint_position_weight:float = 0.0
    mimic_joint_vel_weight:float = 0.0

    anchor_height_only:bool = False

@dataclasses.dataclass
class AMPRewardsCfg:
    discriminator_reward_scale: float = 2.0
    discriminator_reward_weight: float = 1.0

@dataclasses.dataclass
class RewardsCfg:
    # Mimic reward indices.
    anchor_body_index: int = dataclasses.MISSING
    key_body_indices: list[int] = dataclasses.MISSING
    # Penalty reward indices.
    collision_track_body_indices: list[int] = dataclasses.MISSING
    foot_body_indices: list[int] = dataclasses.MISSING
    foot_contact_body_indices: list[int] = dataclasses.MISSING
    # Shared options.
    dt: float = dataclasses.MISSING

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
    anchor_height_only:bool = False
    # Penalty reward weights.
    joint_acc_weight:float = -2.5e-7
    joint_torque_wegiht:float = -1e-5
    joint_limit_weight:float = -10.0
    self_collision_weight:float = -0.1
    self_collision_force_threshold:float = 1.0
    foot_slip_weight: float = -0.1
    foot_slip_force_threshold: float = 1.0
    action_rate_weight:float = -1e-3

class Rewards:
    def __init__(self, cfg:RewardsCfg):
        self.cfg = cfg
        penalty_cfg = PenaltyRewardCfg(
            collision_track_body_indices=cfg.collision_track_body_indices,
            foot_body_indices=cfg.foot_body_indices,
            foot_contact_body_indices=cfg.foot_contact_body_indices,
            joint_acc_weight=cfg.joint_acc_weight,
            joint_torque_wegiht=cfg.joint_torque_wegiht,
            joint_limit_weight=cfg.joint_limit_weight,
            self_collision_weight=cfg.self_collision_weight,
            self_collision_force_threshold=cfg.self_collision_force_threshold,
            foot_slip_weight=cfg.foot_slip_weight,
            foot_slip_force_threshold=cfg.foot_slip_force_threshold,
            action_rate_weight=cfg.action_rate_weight,
        )
        mimic_cfg = MimicRewardsCfg(
            anchor_body_index=cfg.anchor_body_index,
            key_body_indices=cfg.key_body_indices,
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
        action_model: ActionProcessor
    ) -> torch.Tensor:
        regulation_reward = self.regulation_reward.get_reward(robot, contact_sensor, action_model)
        mimic_reward = self.mimic_reward.get_reward(robot, reference_motion)
        reward_vector = torch.cat([regulation_reward, mimic_reward], dim=-1) * self.cfg.dt
        if self.cfg.return_vector:
            return reward_vector
        return reward_vector.sum(-1)

class RegulationReward:
    def __init__(self, cfg:PenaltyRewardCfg):
        self.cfg = cfg

    def get_reward(self, robot:Articulation, contact_sensor: ContactSensor, action_model: ActionProcessor) -> torch.Tensor:
        joint_acc_penalty = self.joint_acc_l2(robot) * self.cfg.joint_acc_weight
        joint_torque_penalty = self.joint_torque_l2(robot) * self.cfg.joint_torque_wegiht
        joint_limit_penalty = self.joint_limit(robot) * self.cfg.joint_limit_weight
        self_collision_penalty = self.self_collision_penalty(
            contact_sensor,
            self.cfg.collision_track_body_indices,
            self.cfg.self_collision_force_threshold,
        ) * self.cfg.self_collision_weight
        foot_slip_penalty = self.foot_slip_penalty(
            robot,
            contact_sensor,
            self.cfg.foot_body_indices,
            self.cfg.foot_contact_body_indices,
            self.cfg.foot_slip_force_threshold,
        ) * self.cfg.foot_slip_weight
        action_rate_penalty = self.action_rate_l2(action_model) * self.cfg.action_rate_weight
        
        reward = torch.stack(
            [
                joint_acc_penalty,
                joint_torque_penalty,
                joint_limit_penalty,
                self_collision_penalty,
                foot_slip_penalty,
                action_rate_penalty,
            ],
            dim=-1
        )

        return reward
    
    def feet_contact_time(self, sensor: ContactSensor, step_dt:float, physics_dt: float,
                          body_ids: IndexLike, threshold: float) -> torch.Tensor:
        first_air = sensor.compute_first_air(step_dt, physics_dt)[:, body_ids]
        last_contact_time = sensor.data.last_contact_time[:, body_ids]
        reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)

        return reward
    
    def action_rate_l2(self, action_model: ActionProcessor) -> torch.Tensor:
        action_delta = action_model.applied_action - action_model.previous_applied_action
        return torch.sum(torch.square(action_delta), dim=1)

    def foot_slip_penalty(
        self,
        robot: Articulation,
        sensor: ContactSensor,
        foot_body_ids: list[int],
        foot_contact_body_ids: list[int],
        threshold: float,
    ) -> torch.Tensor:
        num_envs = robot.data.body_lin_vel_w.shape[0]
        device = robot.data.body_lin_vel_w.device
        if len(foot_body_ids) == 0 or len(foot_contact_body_ids) == 0:
            return torch.zeros(num_envs, device=device)

        contact_history = sensor.data.net_forces_w_history
        if contact_history is None:
            net_contact_forces = sensor.data.net_forces_w
            if net_contact_forces is None:
                return torch.zeros(num_envs, device=device)
            contact_history = net_contact_forces.unsqueeze(1)

        is_contact = (
            torch.norm(contact_history[:, :, foot_contact_body_ids], dim=-1).amax(dim=1) > threshold
        ).to(robot.data.body_lin_vel_w.dtype)
        foot_planar_vel = torch.linalg.norm(robot.data.body_lin_vel_w[:, foot_body_ids, :2], dim=-1)
        return torch.sum(foot_planar_vel * is_contact, dim=1)

    def self_collision_penalty(
        self,
        sensor: ContactSensor,
        body_ids: list[int],
        threshold: float,
    ) -> torch.Tensor:
        net_contact_forces = sensor.data.net_forces_w_history
        if len(body_ids) == 0:
            if net_contact_forces is not None:
                return torch.zeros(net_contact_forces.shape[0], device=net_contact_forces.device)
            net_contact_forces = sensor.data.net_forces_w
            num_envs = 0 if net_contact_forces is None else net_contact_forces.shape[0]
            device = sensor.device if net_contact_forces is None else net_contact_forces.device
            return torch.zeros(num_envs, device=device)

        if net_contact_forces is None:
            net_contact_forces = sensor.data.net_forces_w
            num_envs = 0 if net_contact_forces is None else net_contact_forces.shape[0]
            device = sensor.device if net_contact_forces is None else net_contact_forces.device
            return torch.zeros(num_envs, device=device)

        filtered_contact_forces = sensor.data.force_matrix_w_history
        if filtered_contact_forces is None:
            # Unfiltered net forces cannot distinguish self-contact from expected
            # contacts such as feet on the ground, so do not use them here.
            return torch.zeros(net_contact_forces.shape[0], device=net_contact_forces.device)

        contact_magnitudes = torch.norm(filtered_contact_forces[:, :, body_ids], dim=-1)
        is_contact = contact_magnitudes.amax(dim=1).amax(dim=-1) > threshold
        return torch.sum(is_contact, dim=1)
    
    def joint_acc_l2(self, robot: Articulation) -> torch.Tensor:
        return torch.sum(torch.square(robot.data.joint_acc), dim=1)
    
    def joint_torque_l2(self, robot: Articulation) -> torch.Tensor:
        return torch.sum(torch.square(robot.data.applied_torque), dim=1)
    
    def joint_limit(self, robot: Articulation) -> torch.Tensor:
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
        self.anchor_position_error = 0
        self.anchor_quaternion_error = 0
        self.key_position_error = 0
        self.key_ang_vel_error = 0

    #def get_logs(self) -> dict[str, torch.Tensor]:
    #    return {
    #        "anchor_position_error": self.anchor_position_error,
    #        "anchor_quaternion_error": self.anchor_quaternion_error,
    #        "anchor_linear_vel_error": self.anchor_linear_vel_error,
    #        "anchor_ang_vel_error": self.anchor_ang_vel_error,
    #        "key_position_error": self.key_position_error,
    #        "key_quaternion_error": self.key_quaternion_error,
    #        "key_linear_vel_error": self.key_linear_vel_error,
    #        "key_ang_vel_error": self.key_ang_vel_error,
    #        "anchor_position_reward": self.anchor_position_reward,
    #        "anchor_quaternion_reward": self.anchor_quaternion_reward,
    #        "anchor_linear_vel_reward": self.anchor_linear_vel_reward,
    #        "anchor_ang_vel_reward": self.anchor_ang_vel_reward,
    #        "key_position_reward": self.key_position_reward,
    #        "key_quaternion_reward": self.key_quaternion_reward,
    #        "key_linear_vel_reward": self.key_linear_vel_reward,
    #        "key_ang_vel_reward": self.key_ang_vel_reward,
    #    }

    def get_reward(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        
        anchor_position_error, anchor_quaternion_error = self.anchor_body_pose_error(robot, reference_motion)
        relative_key_body_position_error, relative_key_body_quaternion_error = self.key_body_pose_error(robot, reference_motion)
        key_body_lin_vel_error, key_body_ang_vel_error = self.key_body_state_error(robot, reference_motion)
        joint_position_error, joint_vel_error = self.joint_state_error(robot, reference_motion)

        self.anchor_position_error = anchor_position_error.mean().item()
        self.anchor_quaternion_error = anchor_quaternion_error.mean().item()
        self.relative_key_body_position_error = relative_key_body_position_error.mean().item()
        self.relative_key_body_quaternion_error = relative_key_body_quaternion_error.mean().item()
        self.key_body_lin_vel_error = key_body_lin_vel_error.mean().item()
        self.key_body_ang_vel_error = key_body_ang_vel_error.mean().item()
        self.joint_position_error = joint_position_error.mean().item()
        self.joint_vel_error = joint_vel_error.mean().item()

        anchor_position_reward = torch.exp(-anchor_position_error / self.cfg.position_std) * self.cfg.mimic_anchor_position_weight
        anchor_quaternion_reward = torch.exp(-anchor_quaternion_error / self.cfg.quaternion_std) * self.cfg.mimic_anchor_quaternion_weight
        
        relative_key_body_position_reward = torch.exp(-relative_key_body_position_error / self.cfg.position_std) * self.cfg.mimic_key_position_wegiht
        relative_key_body_quaternion_reward = torch.exp(-relative_key_body_quaternion_error / self.cfg.quaternion_std) * self.cfg.mimic_key_quaternion_weight
        
        key_body_lin_vel_reward = torch.exp(-key_body_lin_vel_error / self.cfg.linear_vel_std) * self.cfg.mimic_key_linear_vel_weight
        key_body_ang_vel_reward = torch.exp(-key_body_ang_vel_error / self.cfg.ang_vel_std) * self.cfg.mimic_key_ang_vel_weight

        self.anchor_position_reward = anchor_position_reward.mean().item()
        self.anchor_quaternion_reward = anchor_quaternion_reward.mean().item()
        self.relative_key_body_position_reward = relative_key_body_position_reward.mean().item()
        self.relative_key_body_quaternion_reward = relative_key_body_quaternion_reward.mean().item()
        self.key_body_lin_vel_reward = key_body_lin_vel_reward.mean().item()
        self.key_body_ang_vel_reward = key_body_ang_vel_reward.mean().item()
        self.joint_position_reward = 0.0
        self.joint_vel_reward = 0.0

        reward = torch.stack(
            [
                anchor_position_reward, anchor_quaternion_reward,
                relative_key_body_position_reward, relative_key_body_quaternion_reward,
                key_body_lin_vel_reward, key_body_ang_vel_reward,
            ],dim=-1
        )
        
        return reward

    def anchor_body_pose_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> tuple[torch.Tensor, torch.Tensor]:
        position_slice = slice(None)
        if self.cfg.anchor_height_only:
            position_slice = slice(2, 3)

        robot_anchor_body_positions = robot.data.body_pos_w[:, self.cfg.anchor_body_index, position_slice]
        robot_anchor_body_quaternions = robot.data.body_quat_w[:, self.cfg.anchor_body_index]

        reference_anchor_body_positions = reference_motion.body_positions[:, self.cfg.anchor_body_index, position_slice]
        reference_anchor_body_quaternions = reference_motion.body_quaternions[:, self.cfg.anchor_body_index]

        position_error = (robot_anchor_body_positions-reference_anchor_body_positions).square().sum(-1)
        quaternion_error = quat_error_magnitude(robot_anchor_body_quaternions, reference_anchor_body_quaternions).square()

        return position_error, quaternion_error
    
    def key_body_pose_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> tuple[torch.Tensor, torch.Tensor]:
        robot_key_body_positions = robot.data.body_pos_w[:, self.cfg.key_body_indices]
        robot_key_body_quaternions = robot.data.body_quat_w[:, self.cfg.key_body_indices]

        reference_relative_key_body_positions = reference_motion.body_pos_relative[:, self.cfg.key_body_indices]
        reference_relative_key_body_quaternions = reference_motion.body_quat_relative[:, self.cfg.key_body_indices]

        position_error = (robot_key_body_positions-reference_relative_key_body_positions).square().sum(-1).mean(-1)
        quaternion_error = quat_error_magnitude(robot_key_body_quaternions, reference_relative_key_body_quaternions).square().mean(-1)
        
        return position_error, quaternion_error
    
    def key_body_state_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> tuple[torch.Tensor, torch.Tensor]:
        robot_key_body_lin_vel = robot.data.body_lin_vel_w[:, self.cfg.key_body_indices]
        robot_key_body_ang_vel = robot.data.body_ang_vel_w[:, self.cfg.key_body_indices]

        alignment_quaternion = self.reference_alignment_quaternion(robot, reference_motion)
        alignment_quaternion = alignment_quaternion[:, None, :].expand(-1, len(self.cfg.key_body_indices), -1)

        reference_key_body_lin_vel = quat_apply(
            alignment_quaternion,
            reference_motion.body_linear_velocities[:, self.cfg.key_body_indices],
        )
        reference_key_body_ang_vel = quat_apply(
            alignment_quaternion,
            reference_motion.body_angular_velocities[:, self.cfg.key_body_indices],
        )

        lin_vel_error = (robot_key_body_lin_vel-reference_key_body_lin_vel).square().sum(-1).mean(-1)
        ang_vel_error = (robot_key_body_ang_vel-reference_key_body_ang_vel).square().sum(-1).mean(-1)

        return lin_vel_error, ang_vel_error

    def reference_alignment_quaternion(
        self,
        robot: Articulation,
        reference_motion: ReferenceMotions,
    ) -> torch.Tensor:
        robot_anchor_quaternion = robot.data.body_quat_w[:, self.cfg.anchor_body_index]
        reference_anchor_quaternion = reference_motion.body_quaternions[:, self.cfg.anchor_body_index]
        return yaw_quat(quat_mul(robot_anchor_quaternion, quat_inv(reference_anchor_quaternion)))
        
    def joint_state_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> tuple[torch.Tensor, torch.Tensor]:
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

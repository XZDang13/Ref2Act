from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from .motion_lib import Sampler, ReferenceMotions

class Termination:
    def __init__(
        self,
        *,
        anchor_body_indices: list[int],
        end_effector_body_indices: list[int],
        anchor_pos_error_threshold:float,
        anchor_ori_error_threshold:float,
        end_effector_pos_error_threshold:float,
        height_only: bool = True
    ) -> None:
        self.anchor_body_indices = anchor_body_indices
        self.end_effector_body_indices = end_effector_body_indices
        self.anchor_pos_error_threshold = anchor_pos_error_threshold
        self.anchor_ori_error_threshold = anchor_ori_error_threshold
        self.end_effector_pos_error_threshold = end_effector_pos_error_threshold
        self.height_only = height_only

    def time_out(self, episode_length_buf: torch.Tensor, max_episode_length: torch.Tensor) -> torch.Tensor:
        return episode_length_buf >= (max_episode_length - 1)

    def height_terminate(self, base_height: torch.Tensor) -> torch.Tensor:
        return base_height < self.termination_height

    def anchor_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        position_slice = slice(None)
        if self.height_only:
            position_slice = slice(2, 3)

        robot_anchor_positions = robot.data.body_link_pos_w[:, self.anchor_body_indices, position_slice]
        reference_anchor_positions = reference_motion.body_positions[:, self.anchor_body_indices, position_slice]

        if self.height_only:
            return torch.any(torch.abs(robot_anchor_positions - reference_anchor_positions) > self.anchor_pos_error_threshold, dim=-1)

        return torch.any(torch.norm(robot_anchor_positions - reference_anchor_positions, dim=-1) > self.anchor_pos_error_threshold, dim=-1) 
        
    def anchor_ori_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        robot_anchor_quat = robot.data.body_link_quat_w[:, self.anchor_body_indices]
        reference_anchor_quat = reference_motion.body_quaternions[:, self.anchor_body_indices]

        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat, robot.data.GRAVITY_VEC_W)
        reference_projected_gravity_b = quat_apply_inverse(reference_anchor_quat, robot.data.GRAVITY_VEC_W)

        return torch.abs(robot_projected_gravity_b[:, 2] - reference_projected_gravity_b[:, 2]) > self.anchor_ori_error_threshold
    
    def end_effector_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        position_slice = slice(None)
        if self.height_only:
            position_slice = slice(2, 3)

        robot_end_effector_positions = robot.data.body_link_pos_w[:, self.end_effector_body_indices, position_slice]
        reference_end_effector_positions = reference_motion.body_positions[:, self.end_effector_body_indices, position_slice]

        if self.height_only:
            return torch.any(torch.abs(robot_end_effector_positions - reference_end_effector_positions) > self.end_effector_pos_error_threshold, dim=-1)

        return torch.any(torch.norm(robot_end_effector_positions - reference_end_effector_positions, dim=-1) > self.end_effector_pos_error_threshold, dim=-1)

    def end_of_motion(self, sampler: Sampler) -> torch.Tensor:
        return sampler.current_times >= sampler.duration

    def get_dones(
        self,
        episode_length_buf: torch.Tensor,
        max_episode_length: torch.Tensor,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        sampler: Sampler
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.time_out(episode_length_buf, max_episode_length)
        end_motiont_erminate = self.end_of_motion(sampler).to(time_out.device)
        anchor_pose_terminate = self.anchor_pos_error_terminate(robot, reference_motion)
        anchor_ori_terminate = self.anchor_ori_error_terminate(robot, reference_motion)
        end_effector_pos_terminate = self.end_effector_pos_error_terminate(robot, reference_motion)

        terminate = end_motiont_erminate | anchor_pose_terminate | anchor_ori_terminate | end_effector_pos_terminate
        
        return terminate, time_out

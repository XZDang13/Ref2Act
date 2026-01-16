from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from .sampler import Sampler, ReferenceMotions

class Termination:
    def __init__(
        self,
        *,
        anchor_body_index: int,
        end_effector_body_indices: list[int],
        anchor_pos_error_threshold:float,
        anchor_ori_error_threshold:float,
        end_effector_pos_error_threshold:float,
        height_only: bool = True
    ) -> None:
        self.anchor_body_index = anchor_body_index
        self.end_effector_body_indices = end_effector_body_indices
        self.anchor_pos_error_threshold = anchor_pos_error_threshold
        self.anchor_ori_error_threshold = anchor_ori_error_threshold
        self.end_effector_pos_error_threshold = end_effector_pos_error_threshold
        self.height_only = height_only
        self.terminated_env_ids = torch.empty(0, dtype=torch.long)

    def time_out(self, episode_length_buf: torch.Tensor, max_episode_length: torch.Tensor) -> torch.Tensor:
        return episode_length_buf >= (max_episode_length - 1)

    def anchor_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        # positions: [B, 3]
        robot_pos = robot.data.body_pos_w[:, self.anchor_body_index]
        ref_pos   = reference_motion.body_positions[:, self.anchor_body_index]

        diff = robot_pos - ref_pos  # [B, 3]

        if self.height_only:
            # use z only -> [B]
            dist = diff[..., 2].abs()
        else:
            # full distance -> [B]
            dist = torch.norm(diff, dim=-1)

        # terminate if anchor exceeds threshold -> [B]
        return dist > self.anchor_pos_error_threshold
       
    def anchor_ori_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        robot_anchor_quat = robot.data.body_link_quat_w[:, self.anchor_body_index]
        reference_anchor_quat = reference_motion.body_quaternions[:, self.anchor_body_index]

        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat, robot.data.GRAVITY_VEC_W)
        reference_projected_gravity_b = quat_apply_inverse(reference_anchor_quat, robot.data.GRAVITY_VEC_W)

        return torch.abs(robot_projected_gravity_b[:, 2] - reference_projected_gravity_b[:, 2]) > self.anchor_ori_error_threshold
    
    def end_effector_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions):
        robot_pos = robot.data.body_pos_w[:, self.end_effector_body_indices]
        ref_pos   = reference_motion.body_pos_relative[:, self.end_effector_body_indices]

        diff = robot_pos - ref_pos  # [B, E, 3]

        if self.height_only:
            dist = diff[..., 2].abs()          # [B, E]
        else:
            dist = torch.norm(diff, dim=-1)    # [B, E]

        return (dist > self.end_effector_pos_error_threshold).any(dim=1)  # [B]
    
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
        max_epsidoe_time_out = self.time_out(episode_length_buf, max_episode_length)
        end_motion_time_out= self.end_of_motion(sampler).to(max_epsidoe_time_out.device)
        anchor_pose_terminate = self.anchor_pos_error_terminate(robot, reference_motion)
        anchor_ori_terminate = self.anchor_ori_error_terminate(robot, reference_motion)
        end_effector_pos_terminate = self.end_effector_pos_error_terminate(robot, reference_motion)

        time_out = max_epsidoe_time_out | end_motion_time_out
        terminate =  anchor_pose_terminate | anchor_ori_terminate | end_effector_pos_terminate
        self.track_terminated_env_ids(terminate)

        #terminate = torch.zeros_like(terminate)       

        return terminate, time_out

    def track_terminated_env_ids(self, failed: torch.Tensor) -> torch.Tensor:
        terminated_env_ids = torch.nonzero(failed, as_tuple=False).squeeze(-1)
        self.terminated_env_ids = terminated_env_ids
        return terminated_env_ids

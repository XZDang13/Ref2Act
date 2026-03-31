from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse
from ref2act.motion.sampling import MotionSampler

from .types import ReferenceMotions


class Termination:
    def __init__(
        self,
        *,
        anchor_body_index: int,
        end_effector_body_indices: list[int],
        anchor_pos_error_threshold: float,
        anchor_ori_error_threshold: float,
        end_effector_pos_error_threshold: float,
        height_only: bool = True,
        end_effector_height_only: bool = False,
        probabilistic_error_termination: bool = False,
        error_termination_ramp_multiplier: float = 2.0,
        error_termination_sigmoid_steepness: float = 8.0,
    ) -> None:
        if error_termination_ramp_multiplier <= 1.0:
            raise ValueError("error_termination_ramp_multiplier must be greater than 1.0.")
        if error_termination_sigmoid_steepness <= 0.0:
            raise ValueError("error_termination_sigmoid_steepness must be positive.")

        self.anchor_body_index = anchor_body_index
        self.end_effector_body_indices = end_effector_body_indices
        self.anchor_pos_error_threshold = anchor_pos_error_threshold
        self.anchor_ori_error_threshold = anchor_ori_error_threshold
        self.end_effector_pos_error_threshold = end_effector_pos_error_threshold
        self.height_only = height_only
        self.end_effector_height_only = end_effector_height_only
        self.probabilistic_error_termination = probabilistic_error_termination
        self.error_termination_ramp_multiplier = error_termination_ramp_multiplier
        self.error_termination_sigmoid_steepness = error_termination_sigmoid_steepness
        self.terminated_env_ids = torch.empty(0, dtype=torch.long)

    def time_out(self, episode_length_buf: torch.Tensor, max_episode_length: torch.Tensor) -> torch.Tensor:
        return episode_length_buf >= (max_episode_length - 1)

    def anchor_pos_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        robot_pos = robot.data.body_pos_w[:, self.anchor_body_index]
        ref_pos = reference_motion.body_positions[:, self.anchor_body_index]

        diff = robot_pos - ref_pos

        if self.height_only:
            return diff[..., 2].abs()
        return torch.norm(diff, dim=-1)

    def anchor_ori_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        robot_anchor_quat = robot.data.body_quat_w[:, self.anchor_body_index]
        reference_anchor_quat = reference_motion.body_quaternions[:, self.anchor_body_index]

        robot_projected_gravity_b = quat_apply_inverse(robot_anchor_quat, robot.data.GRAVITY_VEC_W)
        reference_projected_gravity_b = quat_apply_inverse(reference_anchor_quat, robot.data.GRAVITY_VEC_W)

        return torch.abs(robot_projected_gravity_b[:, 2] - reference_projected_gravity_b[:, 2])

    def end_effector_pos_error(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        robot_pos = robot.data.body_pos_w[:, self.end_effector_body_indices]
        ref_pos = reference_motion.body_pos_relative[:, self.end_effector_body_indices]

        diff = robot_pos - ref_pos
        if self.end_effector_height_only:
            return diff[..., 2].abs()
        return torch.norm(diff, dim=-1)

    def end_of_motion(self, sampler: MotionSampler) -> torch.Tensor:
        return sampler.current_times >= sampler.get_current_durations()

    def _error_to_termination_probability(self, error: torch.Tensor, threshold: float) -> torch.Tensor:
        threshold_tensor = error.new_tensor(threshold)
        over_threshold = error > threshold_tensor
        if not torch.any(over_threshold):
            return torch.zeros_like(error)

        ramp_span = threshold_tensor * (self.error_termination_ramp_multiplier - 1.0)
        normalized_error = torch.clamp((error - threshold_tensor) / ramp_span, min=0.0, max=1.0)

        steepness = error.new_tensor(self.error_termination_sigmoid_steepness)
        sigmoid_values = torch.sigmoid(steepness * (normalized_error - 0.5))
        sigmoid_start = torch.sigmoid(-0.5 * steepness)
        sigmoid_end = torch.sigmoid(0.5 * steepness)
        normalized_sigmoid = (sigmoid_values - sigmoid_start) / (sigmoid_end - sigmoid_start)

        probabilities = torch.where(over_threshold, normalized_sigmoid, torch.zeros_like(error))
        return probabilities.clamp_(0.0, 1.0)

    def _sample_termination(self, probabilities: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(probabilities) < probabilities

    def _error_terminate(self, error: torch.Tensor, threshold: float) -> torch.Tensor:
        if not self.probabilistic_error_termination:
            return error > threshold

        probabilities = self._error_to_termination_probability(error, threshold)
        return self._sample_termination(probabilities)

    def anchor_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        return self._error_terminate(self.anchor_pos_error(robot, reference_motion), self.anchor_pos_error_threshold)

    def anchor_ori_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        return self._error_terminate(self.anchor_ori_error(robot, reference_motion), self.anchor_ori_error_threshold)

    def end_effector_pos_error_terminate(self, robot: Articulation, reference_motion: ReferenceMotions) -> torch.Tensor:
        end_effector_hits = self._error_terminate(
            self.end_effector_pos_error(robot, reference_motion),
            self.end_effector_pos_error_threshold,
        )
        return end_effector_hits.any(dim=1)

    def get_dones(
        self,
        episode_length_buf: torch.Tensor,
        max_episode_length: torch.Tensor,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        sampler: MotionSampler,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_episode_time_out = self.time_out(episode_length_buf, max_episode_length)
        end_motion_time_out = self.end_of_motion(sampler).to(max_episode_time_out.device)
        anchor_pose_terminate = self.anchor_pos_error_terminate(robot, reference_motion)
        anchor_ori_terminate = self.anchor_ori_error_terminate(robot, reference_motion)
        end_effector_pos_terminate = self.end_effector_pos_error_terminate(robot, reference_motion)

        time_out = max_episode_time_out | end_motion_time_out
        terminate = anchor_pose_terminate | anchor_ori_terminate | end_effector_pos_terminate
        self.track_terminated_env_ids(terminate)

        return terminate, time_out

    def track_terminated_env_ids(self, failed: torch.Tensor) -> torch.Tensor:
        terminated_env_ids = torch.nonzero(failed, as_tuple=False).squeeze(-1)
        self.terminated_env_ids = terminated_env_ids
        return terminated_env_ids

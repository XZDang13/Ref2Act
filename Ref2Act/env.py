import dataclasses
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .config.env_cfg import G1MotionTrackingEnvCfg, PiPlusMotionTrackingEnvCfg, ActionMod
from .action import ActionProcessor
from .curriculum import TerminationThresholdCurriculum
from .motion_lib import MotionLib
from .observation import Observation
from .rewards import Rewards, RewardsCfg
from .termination import Termination
from .visualization import ReferenceMotionViewer
from .sampler import Sampler, SamplingStrategy

class G1MotionTrackingEnv(DirectRLEnv):
    cfg: G1MotionTrackingEnvCfg | PiPlusMotionTrackingEnvCfg
    _REFERENCE_MOTION_FIELDS = (
        "joint_pos",
        "joint_vel",
        "body_positions",
        "body_quaternions",
        "body_linear_velocities",
        "body_angular_velocities",
        "robot_body_positions",
        "robot_body_quaternions",
        "body_pos_relative",
        "body_quat_relative",
    )

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_history_length = self.cfg.action_buffer_length
        if hasattr(self.cfg, "action_latency_range"):
            action_history_length = max(action_history_length, self.cfg.action_latency_range[1] + 1)
        self.action_processer = ActionProcessor(
            self.robot,
            action_history_length,
            self.cfg.action_noise,
            action_mod=self.cfg.action_mod,
        )
        if self.cfg.action_mod == ActionMod.Median:
            self.action_processer.set_median_scale_offset(self.robot)
        elif self.cfg.action_mod == ActionMod.Offset:
            self.action_processer.set_robot_default_scale_offset(self.robot)
        elif self.cfg.action_mod == ActionMod.Residual:
            self.action_processer.set_residual_scale_offset(self.robot)
        elif self.cfg.action_mod == ActionMod.CurrentResidual:
            self.action_processer.set_current_residual_scale_offset(self.robot)
        else:
            raise ValueError(f"Unsupported action mode: {self.cfg.action_mod}")

        self.motion_lib = MotionLib(self.cfg.expert_motion_file, self.device)

        anchor_body_index = self._resolve_shared_body_index(self.cfg.anchor_body_name)
        key_body_indices = [self._resolve_shared_body_index(name) for name in self.cfg.key_body_names]
        enf_effector_indices = [self._resolve_shared_body_index(name) for name in self.cfg.end_effector_body_names]
        foot_body_indices = [self.robot.data.body_names.index(name) for name in self.cfg.foot_body_names]
        self.root_link_index = self._resolve_shared_body_index(self.cfg.root_link_name)

        collision_track_body_indices, _ = self.contact_sensor.find_bodies(self.cfg.collision_track_body_names)
        foot_contact_body_indices, _ = self.contact_sensor.find_bodies(self.cfg.foot_body_names, preserve_order=True)
        self.collision_track_body_indices = collision_track_body_indices

        self.sampler = Sampler(
            num_envs=self.cfg.scene.num_envs,
            motion_lib=self.motion_lib,
            dt=self.step_dt,
            anchor_body_index=anchor_body_index,
            root_body_index=self.root_link_index,
            reset_noise=self.cfg.add_reset_noise,
            bin_size=self.cfg.bin_size,
            failure_decay=self.cfg.failure_decay,
            device=self.device,
        )
        if self._get_sampling_strategy() == SamplingStrategy.FailureWeighted and self.cfg.bin_size is None:
            raise ValueError("Failure-weighted sampling requires cfg.bin_size to be set.")
        if (
            self._get_sampling_strategy() == SamplingStrategy.FailureWeighted
            and not self.sampler.supports_failure_weighted_sampling
        ):
            raise ValueError(
                "Failure-weighted sampling requires motion clips with segment metadata. "
                "Reconvert the motion .npz files with `convert --segment-bin-size ...`."
            )

        self.observation_model = Observation(anchor_body_index, key_body_indices, self.cfg.add_obs_noise)
        
        
        reward_cfg_kwargs = {
            "anchor_height_only": getattr(self.cfg, "anchor_height_only", self.cfg.height_only),
        }
        for field in dataclasses.fields(RewardsCfg):
            if field.name in {
                "anchor_body_index",
                "key_body_indices",
                "collision_track_body_indices",
                "foot_body_indices",
                "foot_contact_body_indices",
                "dt",
                "anchor_height_only",
            }:
                continue
            if hasattr(self.cfg, field.name):
                reward_cfg_kwargs[field.name] = getattr(self.cfg, field.name)
        reward_cfg = RewardsCfg(
            anchor_body_index=anchor_body_index,
            key_body_indices=key_body_indices,
            collision_track_body_indices=collision_track_body_indices,
            foot_body_indices=foot_body_indices,
            foot_contact_body_indices=foot_contact_body_indices,
            dt=self.step_dt,
            **reward_cfg_kwargs,
        )
        self.reward_model = Rewards(reward_cfg)

        viewer_enabled = getattr(self.cfg, "reference_motion_viewer_enabled", True)
        self.reference_motion_viewer = ReferenceMotionViewer(key_body_indices) if viewer_enabled else None
        self.termination_model = Termination(
            anchor_body_index=anchor_body_index,
            end_effector_body_indices=enf_effector_indices,
            anchor_pos_error_threshold=self.cfg.anchor_pos_error_threshold,
            anchor_ori_error_threshold=self.cfg.anchor_ori_error_threshold,
            end_effector_pos_error_threshold=self.cfg.end_effector_pos_error_threshold,
            height_only=self.cfg.height_only,
            end_effector_height_only=getattr(self.cfg, "end_effector_height_only", False),
            probabilistic_error_termination=getattr(self.cfg, "probabilistic_error_termination", False),
            error_termination_ramp_multiplier=getattr(self.cfg, "error_termination_ramp_multiplier", 2.0),
            error_termination_sigmoid_steepness=getattr(
                self.cfg, "error_termination_sigmoid_steepness", 8.0
            ),
        )
        self.termination_curriculum = TerminationThresholdCurriculum(
            self.termination_model,
            getattr(self.cfg, "termination_curriculum", None),
        )
        self._apply_termination_curriculum(step=0)

    def _resolve_shared_body_index(self, body_name: str) -> int:
        try:
            robot_index = self.robot.data.body_names.index(body_name)
        except ValueError as exc:
            raise ValueError(f"Body '{body_name}' was not found in robot.body_names.") from exc

        try:
            motion_index = self.motion_lib.body_names.index(body_name)
        except ValueError as exc:
            raise ValueError(f"Body '{body_name}' was not found in motion_lib.body_names.") from exc

        if robot_index != motion_index:
            raise ValueError(
                f"Body '{body_name}' has mismatched indices between robot ({robot_index}) "
                f"and motion clip ({motion_index}). Ref2Act requires matching body order "
                "for shared robot/reference bodies."
            )

        return robot_index

    def _store_reference_motion(self, env_ids: torch.Tensor, reference_motion) -> None:
        if not hasattr(self, "reference_motion") or len(env_ids) == self.num_envs:
            self.reference_motion = reference_motion
            return

        for field_name in self._REFERENCE_MOTION_FIELDS:
            current_value = getattr(self.reference_motion, field_name)
            updated_value = getattr(reference_motion, field_name)
            if current_value is None or updated_value is None:
                setattr(self.reference_motion, field_name, updated_value)
                continue
            current_value[env_ids] = updated_value

    def _advance_reference_motion(self) -> None:
        reference_motion = self.sampler.sample_next_motions(self.robot._ALL_INDICES, self.robot, self.scene)
        self._store_reference_motion(self.robot._ALL_INDICES, reference_motion)
        self.action_processer.set_reference_joint_position(reference_motion.joint_pos)

    def _apply_termination_curriculum(self, step: int) -> None:
        current_thresholds = self.termination_curriculum.apply(step=step)
        if not self.termination_curriculum.has_schedules:
            return

        for threshold_name, threshold_value in current_thresholds.items():
            self.extras[f"curriculum/{threshold_name}"] = threshold_value

    def _get_sampling_strategy(self) -> SamplingStrategy:
        strategy = getattr(self.cfg, "sampling_strategy", None)
        if strategy is not None:
            return strategy
        if getattr(self.cfg, "random_start", False):
            return SamplingStrategy.Random
        return SamplingStrategy.Start

    def get_joint_params(self):
        joint_names = self.robot.data.joint_names
        joint_effort_limits = self.robot.data.joint_effort_limits[0]
        joint_pos_limits = self.robot.data.default_joint_limits[0]
        joint_stiffness = self.robot.data.default_joint_stiffness[0]
        joint_damping = self.robot.data.default_joint_damping[0]
        action_offset = self.action_processer.offset[0]
        action_scale = self.action_processer.scale

        return {
            "joint_names": joint_names,
            "joint_effort_limits": joint_effort_limits,
            "joint_pos_limits": joint_pos_limits,
            "joint_stiffness": joint_stiffness,
            "joint_damping": joint_damping,
            "action_offset": action_offset,
            "action_scale": action_scale,
            "action_mode": self.action_processer.action_mode,
        }
        
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Direct scene construction bypasses InteractiveScene's terrain registration,
        # so wire it back to expose terrain-based environment origins.
        self.scene._terrain = self.terrain

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Keep rewards, dones, and the returned observations aligned to the same
        # reference frame for the current environment step.
        self._advance_reference_motion()
        self.action_processer.pre_process_action(actions)
        #print(self.action_processer.applied_action[0])
        #print("-------------")

    def _apply_action(self):
        self.robot.set_joint_position_target(self.action_processer.target_joint_position)
        #print(self.robot.data.applied_torque)
        
    def _get_observations(self):
        if self.reference_motion_viewer is not None:
            self.reference_motion_viewer.visualize(self.reference_motion)

        self.previous_actions = self.action_processer.applied_action.clone()
        

        obs = self.observation_model.get_default_observation(self.robot,
                                                             self.reference_motion,
                                                             self.scene,
                                                             self.action_processer.applied_action)

        return obs
    
    def _get_rewards(self) -> torch.Tensor:
        reward = self.reward_model.get_task_reward(self.robot, self.reference_motion, self.contact_sensor,
                                                   self.action_processer)
        
        #mimic_logs = self.reward_model.mimic_reward.get_logs()

        #self.extras["anchor_position_reward"] = mimic_logs["anchor_position_reward"]
        #self.extras["anchor_quaternion_reward"] = mimic_logs["anchor_quaternion_reward"]
        #self.extras["anchor_linear_vel_reward"] = mimic_logs["anchor_linear_vel_reward"]
        #self.extras["anchor_ang_vel_reward"] = mimic_logs["anchor_ang_vel_reward"]
        #self.extras["key_position_reward"] = mimic_logs["key_position_reward"]
        #self.extras["key_quaternion_reward"] = mimic_logs["key_quaternion_reward"]
        #self.extras["key_linear_vel_reward"] = mimic_logs["key_linear_vel_reward"]
        #self.extras["key_ang_vel_reward"] = mimic_logs["key_ang_vel_reward"]

        return reward
     
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._apply_termination_curriculum(step=self.common_step_counter)
        terminate, time_out = self.termination_model.get_dones(self.episode_length_buf, self.max_episode_length, self.robot,
                                                self.reference_motion, self.sampler)
        
        self.sampler.record_failures(self.termination_model.terminated_env_ids)
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        #if len(env_ids) == self.num_envs and self.cfg.training:
        #    self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.action_processer.reset_action_buffer(env_ids)
        if self.cfg.add_action_noise:
            self.action_processer.set_random_offset_noise(env_ids)

        reference_motion = self.sampler.sample_reset_motions(
            env_ids,
            self.robot,
            self.scene,
            strategy=self._get_sampling_strategy(),
            min_weight=self.cfg.failure_weight_min,
            temperature=self.cfg.failure_temperature,
        )

        self._store_reference_motion(env_ids, reference_motion)
        self.action_processer.set_reference_joint_position(reference_motion.joint_pos, env_ids)
        self.target_pos = self.reference_motion.joint_pos

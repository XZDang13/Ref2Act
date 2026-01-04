import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp import undesired_contacts

from .config.env_cfg import G1MotionTrackingEnvCfg, ActionMod
from .action import ActionProcessor
from .motion_lib import MotionLib, Sampler
from .observation import Observation
from .scence_setter import InitialSetting
from .rewards import Rewards, RewardsCfg

class G1MotionTrackingEnv(DirectRLEnv):
    cfg:G1MotionTrackingEnvCfg

    def __init__(self, cfg, render_mode = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        self.action_processer = ActionProcessor(self.robot, self.cfg.action_buffer_length)
        if self.cfg.action_mod == ActionMod.Median:
            self.action_processer.set_median_scale_offset(self.robot)
        elif self.cfg.action_mod == ActionMod.Offset:
            self.action_processer.set_robot_default_scale_offset(self.robot, self.cfg.action_scale)


        self.motion_lib = MotionLib(self.cfg.expert_motion_file, self.robot.data.joint_names, self.device)

        robot_anchor_body_indices = [self.robot.data.body_names.index(name) for name in self.cfg.anchor_body_names]
        robot_key_body_indices = [self.robot.data.body_names.index(name) for name in self.cfg.key_body_names]
        motion_anchor_body_indices = self.motion_lib.get_body_indices(self.cfg.anchor_body_names)
        motion_key_body_indices = self.motion_lib.get_body_indices(self.cfg.key_body_names)
        self.root_link_index = self.motion_lib.get_body_index(self.cfg.root_link_name)

        collision_track_body_indices, _ = self.contact_sensor.find_bodies(self.cfg.collision_track_body_names)
        self.collision_track_body_indices = collision_track_body_indices
        print(self.collision_track_body_indices)

        self.sampler = Sampler(self.cfg.scene.num_envs, self.motion_lib.duration,
                               self.step_dt, self.motion_lib.num_frames)

        self.observation_model = Observation(robot_anchor_body_indices, robot_key_body_indices,
                                             motion_anchor_body_indices, motion_key_body_indices)
        
        reward_cfg = RewardsCfg(robot_anchor_body_indices=robot_anchor_body_indices,
                                robot_key_body_indices=robot_key_body_indices,
                                motion_anchor_body_indices=motion_anchor_body_indices,
                                motion_key_body_indices=motion_key_body_indices,
                                collision_track_body_indices=collision_track_body_indices,
                                self_collision_force_threshold=self.cfg.contact_sensor.force_threshold)
        
        self.reward_model = Rewards(reward_cfg)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.action_processer.pre_process_action(actions)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.action_processer.target_joint_position)
        
    def _get_observations(self):
        self.previous_actions = self.action_processer.applied_action.clone()

        default_obs = self.observation_model.default_robot_observation(self.robot, self.previous_actions,
                                                                       self.cfg.add_obs_noise)
        previlege_obs = self.observation_model.default_robot_privilege_observation(self.robot)
        times = self.sampler.sample_next("clamp")
        self.reference_motion = self.motion_lib.sample_motion(times, self.scene.env_origins)

        return {"default": default_obs, "previlege_obs": previlege_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        reward = self.reward_model.get_task_reward(self.robot, self.reference_motion, self.contact_sensor)

        return reward
     
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height = self.robot.data.root_state_w[:, 2]
        terminate = base_height < self.cfg.termination_height
        #terminate = base_height < 0
        return terminate, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        self.action_processer.reset_action_buffer(env_ids)

        if self.cfg.training:
            times = self.sampler.sample_rand_times(env_ids)
        else:
            times = self.sampler.sample_start_times(env_ids)

        self.reference_motion = self.motion_lib.sample_motion(times, self.scene.env_origins[env_ids])

        InitialSetting.set_robot_initial_state(
            self.robot,
            env_ids,
            self.reference_motion,
            self.root_link_index,
            self.cfg.add_reset_noise
        )

        self.target_pos = self.reference_motion.joint_pos

from __future__ import annotations

from importlib import import_module

import isaaclab.sim as sim_utils
import torch
from isaaclab.envs import DirectRLEnv

from ref2act.envs.motion_tracking.action import ActionProcessor
from ref2act.isaac_compat import to_torch
from ref2act.robots.spec import resolve_robot_spec


def resolve_cfg_factory(cfg_factory: str):
    module_name, _, attr_name = cfg_factory.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid cfg_factory: {cfg_factory!r}")
    module = import_module(module_name)
    cfg_cls = getattr(module, attr_name)
    return cfg_cls()


class LeggedRobotEnv(DirectRLEnv):
    """Shared Isaac Lab runtime for task-specific legged-robot environments."""

    def __init__(self, cfg=None, render_mode=None, cfg_factory: str | None = None, **kwargs):
        if cfg is None:
            if cfg_factory is None:
                raise ValueError("Either cfg or cfg_factory must be provided.")
            cfg = resolve_cfg_factory(cfg_factory)

        robot_spec_name = getattr(cfg, "robot_spec_name", None)
        if not isinstance(robot_spec_name, str):
            raise TypeError("LeggedRobotEnv cfg.robot_spec_name must be a string.")
        robot_spec = resolve_robot_spec(robot_spec_name)
        if int(cfg.action_space) != robot_spec.action_dim:
            raise ValueError(
                f"action_space={cfg.action_space} does not match "
                f"robot_spec.action_dim={robot_spec.action_dim}."
            )
        self.robot_spec = robot_spec

        super().__init__(cfg, render_mode, **kwargs)

        # Isaac Lab 3 may expose DirectRLEnv buffers through Warp proxies.  The
        # task implementations use indexed Torch writes, so retain Torch views.
        self.episode_length_buf = to_torch(self.episode_length_buf)
        self.reset_terminated = to_torch(self.reset_terminated)
        self.reset_time_outs = to_torch(self.reset_time_outs)
        self.reset_buf = to_torch(self.reset_buf)

        self.robot_spec.validate_joint_names(self.robot.data.joint_names)
        self._policy_order_joint_indices = torch.tensor(
            self.robot_spec.policy_order_indices(self.robot.data.joint_names),
            device=self.device,
            dtype=torch.long,
        )
        self._simulator_order_joint_indices = torch.tensor(
            self.robot_spec.simulator_order_indices(self.robot.data.joint_names),
            device=self.device,
            dtype=torch.long,
        )
        self.anchor_body_index = self.robot_spec.body_index(
            self.robot.data.body_names,
            self.robot_spec.anchor_body,
        )
        self.root_body_index = self.robot_spec.body_index(
            self.robot.data.body_names,
            self.robot_spec.root_body,
        )
        self.foot_body_indices = self.robot_spec.body_indices(
            self.robot.data.body_names,
            self.robot_spec.foot_bodies,
        )

        self.action_processor = ActionProcessor(self.robot, self.cfg.action)
        # Preserve the historical typo for randomization callbacks and existing
        # external scripts until the old motion environment is migrated.
        self.action_processer = self.action_processor

    def _setup_scene(self) -> None:
        articulation_type = self.cfg.robot.class_type
        if isinstance(articulation_type, str):
            module_name, _, attr_name = articulation_type.partition(":")
            articulation_type = getattr(import_module(module_name), attr_name)
        self.robot = articulation_type(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        contact_sensor_type = self.cfg.contact_sensor.class_type
        if isinstance(contact_sensor_type, str):
            module_name, _, attr_name = contact_sensor_type.partition(":")
            contact_sensor_type = getattr(import_module(module_name), attr_name)
        self.contact_sensor = contact_sensor_type(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene._terrain = self.terrain

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.action_processor.target_joint_position)

    def _sim_to_policy_order(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, self._policy_order_joint_indices]

    def _policy_to_sim_order(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, self._simulator_order_joint_indices]

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None or len(env_ids) == self.num_envs:
            return to_torch(self.robot._ALL_INDICES)
        return to_torch(env_ids)

    def _reset_robot_to_default(
        self,
        env_ids: torch.Tensor,
        *,
        joint_position_noise: float = 0.0,
    ) -> None:
        root_state = to_torch(self.robot.data.default_root_state)[env_ids].clone()
        root_state[:, :3] += to_torch(self.scene.env_origins)[env_ids]
        joint_pos = to_torch(self.robot.data.default_joint_pos)[env_ids].clone()
        joint_vel = to_torch(self.robot.data.default_joint_vel)[env_ids].clone()

        if joint_position_noise > 0.0:
            joint_pos += torch.empty_like(joint_pos).uniform_(
                -joint_position_noise,
                joint_position_noise,
            )
            joint_pos = torch.clamp(
                joint_pos,
                min=to_torch(self.robot.data.joint_pos_limits)[env_ids, :, 0],
                max=to_torch(self.robot.data.joint_pos_limits)[env_ids, :, 1],
            )

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    def _reset_common(self, env_ids: torch.Tensor | None, *, joint_position_noise: float = 0.0) -> torch.Tensor:
        env_ids = self._normalize_env_ids(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.action_processor.reset_action_buffer(env_ids)
        if self.cfg.action.noise_scale > 0.0:
            self.action_processor.set_random_offset_noise(env_ids)
        self._reset_robot_to_default(env_ids, joint_position_noise=joint_position_noise)
        return env_ids


__all__ = ["LeggedRobotEnv", "resolve_cfg_factory"]

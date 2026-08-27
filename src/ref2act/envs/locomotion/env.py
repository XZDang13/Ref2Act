from __future__ import annotations

import torch

from ref2act.common.math import quat_apply_inverse
from ref2act.common.observation_spec import ObservationLayout
from ref2act.envs.base import LeggedRobotEnv
from ref2act.isaac_compat import to_torch

from .commands import UniformVelocityCommandGenerator
from .observation import LocomotionObservation
from .rewards import LocomotionRewardInputs, compute_locomotion_reward_terms


class LocomotionEnv(LeggedRobotEnv):
    """Blind velocity-command locomotion with no reference-motion dependency."""

    def __init__(self, cfg=None, render_mode=None, cfg_factory: str | None = None, **kwargs):
        if cfg is None and cfg_factory is not None:
            from ref2act.envs.base import resolve_cfg_factory

            cfg = resolve_cfg_factory(cfg_factory)
            cfg_factory = None
        if cfg is None:
            raise ValueError("Either cfg or cfg_factory must be provided.")

        self._derive_observation_spaces(cfg)
        super().__init__(cfg, render_mode, cfg_factory=cfg_factory, **kwargs)

        layout = ObservationLayout(
            joint_dim=self.robot_spec.action_dim,
            action_dim=self.robot_spec.action_dim,
            key_body_count=0,
            command_dim=3,
        )
        self.command_generator = UniformVelocityCommandGenerator(
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            step_dt=self.step_dt,
            device=self.device,
        )
        self.observation_model = LocomotionObservation(
            spec=self.cfg.observation,
            layout=layout,
            num_envs=self.num_envs,
            device=self.device,
            anchor_body_index=self.anchor_body_index,
            policy_order_joint_indices=self._policy_order_joint_indices,
        )
        self._terrain_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    @staticmethod
    def _derive_observation_spaces(cfg) -> None:
        description = cfg.observation.describe(
            ObservationLayout(
                joint_dim=int(cfg.action_space),
                action_dim=int(cfg.action_space),
                key_body_count=0,
                command_dim=3,
            )
        )
        cfg.command_observation_space = description.group_dims.get("command", 0)
        cfg.robot_observation_space = description.group_dims.get("robot", 0)
        cfg.critic_observation_space = description.group_dims.get("privilege", 0)
        cfg.policy_observation_space = sum(
            dim for group_name, dim in description.group_dims.items() if group_name != "privilege"
        )
        cfg.observation_space = cfg.policy_observation_space

    def _anchor_state_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[:, self.anchor_body_index]
        anchor_lin_vel_b = quat_apply_inverse(
            anchor_quat_w,
            to_torch(self.robot.data.body_link_lin_vel_w)[:, self.anchor_body_index],
        )
        anchor_ang_vel_b = quat_apply_inverse(
            anchor_quat_w,
            to_torch(self.robot.data.body_link_ang_vel_w)[:, self.anchor_body_index],
        )
        projected_gravity_b = quat_apply_inverse(anchor_quat_w, to_torch(self.robot.data.GRAVITY_VEC_W))
        return anchor_lin_vel_b, anchor_ang_vel_b, projected_gravity_b

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.command_generator.step()
        self.action_processor.pre_process_action(self._policy_to_sim_order(actions))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return self.observation_model.get_default_observation(
            self.robot,
            self.command_generator.commands,
            self._sim_to_policy_order(self.action_processor.applied_action),
        )

    def _get_rewards(self) -> torch.Tensor:
        base_lin_vel_b, base_ang_vel_b, projected_gravity_b = self._anchor_state_b()
        terms = compute_locomotion_reward_terms(
            LocomotionRewardInputs(
                commands=self.command_generator.commands,
                base_linear_velocity_b=base_lin_vel_b,
                base_angular_velocity_b=base_ang_vel_b,
                projected_gravity_b=projected_gravity_b,
                joint_pos=to_torch(self.robot.data.joint_pos),
                default_joint_pos=to_torch(self.robot.data.default_joint_pos),
                joint_acc=to_torch(self.robot.data.joint_acc),
                applied_torque=to_torch(self.robot.data.applied_torque),
                applied_action=self.action_processor.applied_action,
                previous_applied_action=self.action_processor.previous_applied_action,
            ),
            self.cfg.rewards,
        )
        log = self.extras.setdefault("log", {})
        for name, value in terms.items():
            log[f"Reward/{name}"] = value.mean().detach()
        return torch.stack(tuple(terms.values()), dim=-1).sum(dim=-1) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, projected_gravity_b = self._anchor_state_b()
        base_height = to_torch(self.robot.data.body_link_pos_w)[:, self.root_body_index, 2]
        fallen = base_height < float(self.cfg.minimum_base_height)
        tilted = -projected_gravity_b[:, 2] < float(self.cfg.minimum_upright_projection)
        terminated = fallen | tilted
        self._terrain_success.zero_()
        boundary = getattr(self.cfg, "terrain_out_of_bounds_distance", None)
        if boundary is not None:
            base_xy = to_torch(self.robot.data.root_link_pos_w)[:, :2]
            origin_xy = to_torch(self.scene.env_origins)[:, :2]
            self._terrain_success = torch.linalg.vector_norm(base_xy - origin_xy, dim=-1) > float(boundary)
        time_out = (self.episode_length_buf >= (self.max_episode_length - 1)) | self._terrain_success
        return terminated, time_out

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        if not bool(getattr(self.cfg, "terrain_curriculum", False)):
            return
        if not hasattr(self.terrain, "update_env_origins"):
            return
        valid_episode = self.episode_length_buf[env_ids] > 0
        move_up = self._terrain_success[env_ids] & valid_episode
        move_down = self.reset_terminated[env_ids] & valid_episode
        self.terrain.update_env_origins(env_ids, move_up=move_up, move_down=move_down)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        normalized_env_ids = self._normalize_env_ids(env_ids)
        self._update_terrain_curriculum(normalized_env_ids)
        env_ids = self._reset_common(
            normalized_env_ids,
            joint_position_noise=float(self.cfg.joint_position_reset_noise),
        )
        self.command_generator.reset(env_ids)
        self.observation_model.reset(
            env_ids,
            self.robot,
            self.command_generator.commands,
            self._sim_to_policy_order(self.action_processor.applied_action),
        )


__all__ = ["LocomotionEnv"]

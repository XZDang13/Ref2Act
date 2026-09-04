from __future__ import annotations

import math

import torch

from ref2act.common.math import quat_apply, quat_apply_inverse, yaw_quat
from ref2act.common.observation_spec import ObservationLayout
from ref2act.envs.base import LeggedRobotEnv
from ref2act.isaac_compat import to_torch

from .commands import make_velocity_command_generator
from .observation import LocomotionObservation
from .rewards import (
    LocomotionRewardInputs,
    compute_feet_air_time_reward,
    compute_feet_air_time_positive_biped_reward,
    compute_feet_phase_reward,
    compute_feet_gait_reward,
    compute_foot_clearance_reward,
    compute_locomotion_gait_phase,
    compute_locomotion_phase_features,
    compute_locomotion_reward_terms,
)
from .task_rewards import (
    FlatLocomotionRewardCfg,
    FlatLocomotionRewardInputs,
    compute_flat_locomotion_reward_terms,
    phase_gait_signals,
    leg_lateral_separation,
)


def compute_locomotion_termination(
    *,
    terrain_relative_base_height: torch.Tensor,
    upright_projection: torch.Tensor,
    illegal_contact: torch.Tensor,
    minimum_base_height: float,
    minimum_upright_projection: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return termination masks using terrain-relative height."""

    fallen = terrain_relative_base_height < float(minimum_base_height)
    tilted = upright_projection < float(minimum_upright_projection)
    terminated = fallen | tilted | illegal_contact
    return fallen, tilted, terminated


class LocomotionEnv(LeggedRobotEnv):
    """Blind velocity-command locomotion with no reference-motion dependency."""

    def _setup_scene(self) -> None:
        """Build the locomotion scene with isolated robots and traversable terrain.

        IsaacLab's default curriculum-origin assignment reuses a small terrain
        grid across thousands of environments.  The nested G1 collision model
        then puts many robots at exactly the same world pose.  Explicit USD
        collision groups prevent cross-environment contacts, but the overlapping
        broad phase remains prohibitively expensive.  Generated locomotion
        terrains therefore allocate one physical patch per environment, plus a
        guard ring that lets edge environments traverse beyond their spawn tile.
        """

        terrain_generator = getattr(self.cfg.terrain, "terrain_generator", None)
        unique_origins = bool(
            terrain_generator is not None
            and getattr(self.cfg, "unique_terrain_origins", False)
        )
        logical_num_cols = 0
        logical_num_rows = 0
        guard_tiles = 0
        if unique_origins:
            num_envs = int(self.scene.cfg.num_envs)
            logical_num_cols = min(64, num_envs)
            logical_num_rows = math.ceil(num_envs / logical_num_cols)
            guard_tiles = max(0, int(getattr(self.cfg, "terrain_guard_tiles", 1)))
            terrain_generator.num_cols = logical_num_cols + 2 * guard_tiles
            terrain_generator.num_rows = logical_num_rows + 2 * guard_tiles

        super()._setup_scene()

        if unique_origins:
            num_envs = int(self.scene.cfg.num_envs)
            env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
            terrain_levels = (
                torch.div(env_ids, logical_num_cols, rounding_mode="floor")
                + guard_tiles
            )
            terrain_types = torch.remainder(env_ids, logical_num_cols) + guard_tiles
            self.terrain.terrain_levels = terrain_levels
            self.terrain.terrain_types = terrain_types
            self.terrain.max_terrain_level = int(terrain_generator.num_rows)
            self.terrain.env_origins = self.terrain.terrain_origins[
                terrain_levels, terrain_types
            ].clone()

        # Unique physical origins avoid broad-phase overlap at reset, but robots
        # are free to cross tiles during locomotion.  They must therefore still
        # be isolated from robots belonging to other vector environments.
        if self.scene.cfg.filter_collisions and "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(
                global_prim_paths=[self.cfg.terrain.prim_path]
            )

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
        self.command_generator = make_velocity_command_generator(
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            step_dt=self.step_dt,
            device=self.device,
        )
        self._command_curriculum_linear_sum = torch.zeros((), device=self.device)
        self._command_curriculum_yaw_sum = torch.zeros((), device=self.device)
        self._command_curriculum_steps = 0
        self._gait_phase_offset = torch.zeros(self.num_envs, device=self.device)
        self._policy_action = torch.zeros(
            (self.num_envs, self.robot_spec.action_dim), device=self.device
        )
        self._previous_policy_action = torch.zeros_like(self._policy_action)
        self.observation_model = LocomotionObservation(
            spec=self.cfg.observation,
            layout=layout,
            num_envs=self.num_envs,
            device=self.device,
            anchor_body_index=self.anchor_body_index,
            policy_order_joint_indices=self._policy_order_joint_indices,
        )
        self._terrain_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        foot_sensor_ids, foot_sensor_names = self.contact_sensor.find_bodies(
            list(self.robot_spec.foot_bodies), preserve_order=True
        )
        if len(foot_sensor_ids) != len(self.robot_spec.foot_bodies):
            raise RuntimeError(
                "Locomotion contact sensor must contain every configured foot; got "
                f"{foot_sensor_names}."
            )
        self._foot_contact_sensor_indices = torch.tensor(
            foot_sensor_ids, dtype=torch.long, device=self.device
        )
        foot_body_ids, _ = self.robot.find_bodies(
            list(self.robot_spec.foot_bodies), preserve_order=True
        )
        self._foot_body_indices = torch.tensor(
            foot_body_ids, dtype=torch.long, device=self.device
        )
        if isinstance(self.cfg.rewards, FlatLocomotionRewardCfg):
            knee_names = ["left_knee_link", "right_knee_link"]
            knee_body_ids, found_names = self.robot.find_bodies(knee_names, preserve_order=True)
            if list(found_names) != knee_names:
                raise RuntimeError(f"Leg clearance requires ordered left/right knees; got {found_names}.")
            self._knee_body_indices = torch.tensor(knee_body_ids, dtype=torch.long, device=self.device)
        all_contact_ids, _ = self.contact_sensor.find_bodies(".*", preserve_order=True)
        foot_sensor_id_set = set(foot_sensor_ids)
        non_foot_ids = [index for index in all_contact_ids if index not in foot_sensor_id_set]
        if not non_foot_ids:
            raise RuntimeError("Locomotion contact sensor must expose non-foot bodies.")
        self._undesired_contact_sensor_indices = torch.tensor(
            non_foot_ids, dtype=torch.long, device=self.device
        )

        def joint_indices(config_name: str) -> torch.Tensor:
            patterns = getattr(self.cfg, config_name)
            indices, names = self.robot.find_joints(patterns, preserve_order=True)
            if not indices:
                raise RuntimeError(f"{config_name} matched no robot joints: {patterns}.")
            return torch.tensor(indices, dtype=torch.long, device=self.device)

        self._reward_hip_joint_indices = joint_indices("reward_hip_joint_names")
        self._reward_arm_joint_indices = joint_indices("reward_arm_joint_names")
        self._reward_torso_joint_indices = joint_indices("reward_torso_joint_names")
        self._reward_leg_joint_indices = joint_indices("reward_leg_joint_names")
        self._reward_ankle_joint_indices = joint_indices("reward_ankle_joint_names")

        termination_ids, termination_names = self.contact_sensor.find_bodies(
            self.cfg.termination_body_names, preserve_order=True
        )
        if not termination_ids:
            raise RuntimeError(
                "termination_body_names matched no contact-sensor bodies: "
                f"{self.cfg.termination_body_names}."
            )
        self._termination_contact_sensor_indices = torch.tensor(
            termination_ids, dtype=torch.long, device=self.device
        )
        knee_ids, _ = self.contact_sensor.find_bodies(".*_knee_link")
        hand_ids, _ = self.contact_sensor.find_bodies(".*_rubber_hand_link")
        self._diagnostic_knee_contact_indices = torch.tensor(
            knee_ids, dtype=torch.long, device=self.device
        )
        self._diagnostic_hand_contact_indices = torch.tensor(
            hand_ids, dtype=torch.long, device=self.device
        )

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
        self._previous_policy_action.copy_(self._policy_action)
        self._policy_action.copy_(actions)
        self.action_processor.pre_process_action(self._policy_to_sim_order(actions))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return self.observation_model.get_default_observation(
            self.robot,
            self.command_generator.commands,
            self._gait_phase_features(),
            self._sim_to_policy_order(self.action_processor.applied_action),
        )

    def _gait_phase(self) -> torch.Tensor:
        return compute_locomotion_gait_phase(
            self.episode_length_buf,
            self._gait_phase_offset,
            self.command_generator.commands,
            step_dt=float(self.step_dt),
            period=float(self.cfg.rewards.gait_period),
            offsets=tuple(self.cfg.rewards.gait_offsets),
            stand_phase=float(self.cfg.rewards.gait_stand_phase),
        )

    def _gait_phase_features(self) -> torch.Tensor:
        return compute_locomotion_phase_features(self._gait_phase())

    def _feet_air_time_reward(self, *, command_gate: bool) -> torch.Tensor:
        """Legacy completed-flight signal retained only for task-free exploration."""

        first_contact = to_torch(self.contact_sensor.compute_first_contact(self.step_dt))[
            :, self._foot_contact_sensor_indices
        ]
        last_air_time = to_torch(self.contact_sensor.data.last_air_time)[
            :, self._foot_contact_sensor_indices
        ]
        reward = compute_feet_air_time_reward(
            last_air_time,
            first_contact,
            threshold=float(self.cfg.rewards.feet_air_time_threshold),
            maximum=float(self.cfg.rewards.feet_air_time_maximum),
        )
        if command_gate:
            reward *= (
                torch.linalg.vector_norm(self.command_generator.commands[:, :2], dim=-1)
                > 0.1
            )
        return reward

    def _feet_air_time_positive_biped_reward(self) -> torch.Tensor:
        current_air_time = to_torch(self.contact_sensor.data.current_air_time)[
            :, self._foot_contact_sensor_indices
        ]
        current_contact_time = to_torch(self.contact_sensor.data.current_contact_time)[
            :, self._foot_contact_sensor_indices
        ]
        contact_count = torch.sum(current_contact_time > 0.0, dim=-1)
        log = self.extras.setdefault("log", {})
        log["Gait/single_stance_fraction"] = (contact_count == 1).float().mean().detach()
        log["Gait/double_support_fraction"] = (contact_count == 2).float().mean().detach()
        log["Gait/flight_fraction"] = (contact_count == 0).float().mean().detach()
        log["Gait/current_air_time_mean"] = current_air_time.mean().detach()
        contact_history = to_torch(self.contact_sensor.data.net_forces_w_history)

        def support_fraction(indices: torch.Tensor) -> torch.Tensor:
            if indices.numel() == 0:
                return torch.zeros((), device=self.device)
            forces = torch.linalg.vector_norm(
                contact_history[:, :, indices], dim=-1
            ).amax(dim=(1, 2))
            return (forces > 1.0).float().mean()

        log["Gait/knee_support_fraction"] = support_fraction(
            self._diagnostic_knee_contact_indices
        ).detach()
        log["Gait/hand_support_fraction"] = support_fraction(
            self._diagnostic_hand_contact_indices
        ).detach()
        # Flat v4 keeps support diagnostics but has no air-time reward.
        if isinstance(self.cfg.rewards, FlatLocomotionRewardCfg):
            return torch.zeros(self.num_envs, device=self.device)
        return compute_feet_air_time_positive_biped_reward(
            current_air_time,
            current_contact_time,
            self.command_generator.commands,
            threshold=float(self.cfg.rewards.feet_air_time_threshold),
        )

    def _feet_slide_penalty(self) -> torch.Tensor:
        contact_history = to_torch(self.contact_sensor.data.net_forces_w_history)
        contacts = (
            torch.linalg.vector_norm(
                contact_history[:, :, self._foot_contact_sensor_indices], dim=-1
            ).amax(dim=1)
            > 1.0
        )
        foot_velocity_xy = to_torch(self.robot.data.body_link_lin_vel_w)[
            :, self._foot_body_indices, :2
        ]
        return torch.sum(
            torch.linalg.vector_norm(foot_velocity_xy, dim=-1) * contacts, dim=-1
        )

    def _feet_gait_reward(self) -> torch.Tensor:
        current_contact_time = to_torch(self.contact_sensor.data.current_contact_time)[
            :, self._foot_contact_sensor_indices
        ]
        return compute_feet_gait_reward(
            current_contact_time,
            self.command_generator.commands,
            self.episode_length_buf,
            step_dt=float(self.step_dt),
            period=float(self.cfg.rewards.gait_period),
            offsets=tuple(self.cfg.rewards.gait_offsets),
            stance_threshold=float(self.cfg.rewards.gait_stance_threshold),
        )

    def _terrain_ground_height_at(self, query_xy: torch.Tensor) -> torch.Tensor:
        """Return nearest sampled terrain height for each world-frame XY query."""

        if query_xy.ndim not in (2, 3) or query_xy.shape[0] != self.num_envs:
            raise ValueError("query_xy must have shape [env, 2] or [env, query, 2].")
        squeeze_query = query_xy.ndim == 2
        queries = query_xy.unsqueeze(1) if squeeze_query else query_xy
        origin_height = to_torch(self.scene.env_origins)[:, 2].unsqueeze(-1)
        height_sensor = self.scene.sensors.get("base_height_sensor")
        if height_sensor is None:
            result = origin_height.expand(-1, queries.shape[1])
            return result[:, 0] if squeeze_query else result

        ray_hits = to_torch(height_sensor.data.ray_hits_w)
        valid = torch.isfinite(ray_hits[..., 2])
        distance_sq = torch.sum(
            (queries.unsqueeze(2) - ray_hits[:, None, :, :2]).square(), dim=-1
        )
        distance_sq = torch.where(valid[:, None, :], distance_sq, torch.inf)
        nearest = torch.argmin(distance_sq, dim=-1)
        hit_height = torch.gather(
            ray_hits[..., 2], dim=1, index=nearest
        )
        has_hit = valid.any(dim=-1, keepdim=True)
        result = torch.where(has_hit, hit_height, origin_height.expand_as(hit_height))
        return result[:, 0] if squeeze_query else result

    def _feet_clearance_reward(self) -> torch.Tensor:
        foot_position_w = self._foot_contact_point_position_w()
        ground_height = self._terrain_ground_height_at(foot_position_w[..., :2])
        foot_clearance = foot_position_w[..., 2] - ground_height
        foot_velocity_xy = to_torch(self.robot.data.body_link_lin_vel_w)[
            :, self._foot_body_indices, :2
        ]
        return compute_foot_clearance_reward(
            foot_clearance,
            foot_velocity_xy,
            target_height=float(self.cfg.rewards.feet_clearance_target),
            std=float(self.cfg.rewards.feet_clearance_std),
            tanh_mult=float(self.cfg.rewards.feet_clearance_tanh_mult),
        )

    def _feet_phase_reward(self) -> torch.Tensor:
        foot_position_w = self._foot_contact_point_position_w()
        ground_height = self._terrain_ground_height_at(foot_position_w[..., :2])
        foot_height = foot_position_w[..., 2] - ground_height
        return compute_feet_phase_reward(
            foot_height,
            self._gait_phase(),
            stance_height=float(self.cfg.rewards.feet_stance_height),
            swing_height=float(self.cfg.rewards.feet_phase_swing_height),
            tracking_sigma=float(self.cfg.rewards.feet_phase_tracking_sigma),
        )

    def _foot_contact_point_position_w(self) -> torch.Tensor:
        """Return the virtual sole-center points used by HoloSoma's G1 reward."""

        foot_position_w = to_torch(self.robot.data.body_link_pos_w)[
            :, self._foot_body_indices
        ]
        foot_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self._foot_body_indices
        ]
        local_offset = torch.as_tensor(
            self.cfg.rewards.feet_contact_point_offset,
            dtype=foot_position_w.dtype,
            device=foot_position_w.device,
        ).view(1, 1, 3).expand_as(foot_position_w)
        return foot_position_w + quat_apply(
            foot_quat_w.reshape(-1, 4), local_offset.reshape(-1, 3)
        ).reshape_as(foot_position_w)

    def _pose_penalty(self) -> torch.Tensor:
        joint_pos = self._sim_to_policy_order(to_torch(self.robot.data.joint_pos))
        default_joint_pos = self._sim_to_policy_order(
            to_torch(self.robot.data.default_joint_pos)
        )
        weights = torch.as_tensor(
            self.cfg.rewards.pose_weights,
            device=self.device,
            dtype=joint_pos.dtype,
        )
        if weights.shape != (self.robot_spec.action_dim,):
            raise ValueError(
                "Locomotion pose_weights must match the policy joint order: "
                f"expected {self.robot_spec.action_dim}, got {weights.numel()}."
            )
        # Legacy/task-free reward configs have no deadband; preserve their
        # original squared pose error. Flat v6 uses per-joint tolerances.
        tolerances = torch.as_tensor(
            getattr(self.cfg.rewards, "pose_tolerances", (0.0,) * self.robot_spec.action_dim),
            device=self.device,
            dtype=joint_pos.dtype,
        )
        if tolerances.shape != weights.shape:
            raise ValueError("Locomotion pose_tolerances must match the policy joint order.")
        excess = ((joint_pos - default_joint_pos).abs() - tolerances).clamp(min=0.0)
        return torch.sum(excess.square() * weights, dim=-1)

    def _close_feet_xy_penalty(self) -> torch.Tensor:
        foot_position_w = to_torch(self.robot.data.body_link_pos_w)[
            :, self._foot_body_indices
        ]
        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        foot_delta_yaw = quat_apply_inverse(
            yaw_quat(anchor_quat_w),
            foot_position_w[:, 0] - foot_position_w[:, 1],
        )
        return (
            torch.abs(foot_delta_yaw[:, 1])
            < float(self.cfg.rewards.close_feet_threshold)
        ).to(foot_position_w.dtype)

    def _feet_orientation_penalty(self) -> torch.Tensor:
        foot_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self._foot_body_indices
        ]
        gravity_w = to_torch(self.robot.data.GRAVITY_VEC_W)
        gravity_w = gravity_w[:, None, :].expand(-1, foot_quat_w.shape[1], -1)
        projected = quat_apply_inverse(
            foot_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)
        ).reshape(self.num_envs, foot_quat_w.shape[1], 3)
        return torch.linalg.vector_norm(projected[..., :2], dim=-1).sum(dim=-1)

    def _undesired_contact_penalty(self) -> torch.Tensor:
        contact_history = to_torch(self.contact_sensor.data.net_forces_w_history)
        magnitudes = torch.linalg.vector_norm(
            contact_history[:, :, self._undesired_contact_sensor_indices], dim=-1
        )
        contacts = magnitudes.amax(dim=1) > float(
            self.cfg.rewards.undesired_contact_force_threshold
        )
        return contacts.to(dtype=contact_history.dtype).sum(dim=-1)

    def _joint_deviation_l1(self, indices: torch.Tensor) -> torch.Tensor:
        joint_pos = to_torch(self.robot.data.joint_pos)[:, indices]
        default_joint_pos = to_torch(self.robot.data.default_joint_pos)[:, indices]
        return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=-1)

    def _terrain_relative_base_height(self) -> torch.Tensor:
        """Return pelvis height above the local ground directly below it."""

        base_position_w = to_torch(self.robot.data.body_link_pos_w)[
            :, self.root_body_index
        ]
        ground_height_w = self._terrain_ground_height_at(base_position_w[:, :2])
        return base_position_w[:, 2] - ground_height_w

    def _joint_position_limit_penalty(self) -> torch.Tensor:
        joint_pos = to_torch(self.robot.data.joint_pos)
        limits = to_torch(self.robot.data.soft_joint_pos_limits)
        below = -(joint_pos - limits[:, :, 0]).clamp(max=0.0)
        above = (joint_pos - limits[:, :, 1]).clamp(min=0.0)
        return torch.sum(below + above, dim=-1)

    def _locomotion_reward_terms(self) -> dict[str, torch.Tensor]:
        """Return the configured native locomotion objective."""

        if isinstance(self.cfg.rewards, FlatLocomotionRewardCfg):
            return self._flat_locomotion_reward_terms()

        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        anchor_lin_vel_w = to_torch(self.robot.data.body_link_lin_vel_w)[
            :, self.anchor_body_index
        ]
        base_lin_vel_b, base_ang_vel_b, projected_gravity_b = self._anchor_state_b()
        base_lin_vel_yaw = quat_apply_inverse(yaw_quat(anchor_quat_w), anchor_lin_vel_w)
        terminated = getattr(self, "reset_terminated", None)
        if terminated is None:
            terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        zero = torch.zeros(self.num_envs, device=self.device)
        rewards_cfg = self.cfg.rewards
        return compute_locomotion_reward_terms(
            LocomotionRewardInputs(
                commands=self.command_generator.commands,
                base_linear_velocity_b=base_lin_vel_b,
                base_angular_velocity_b=base_ang_vel_b,
                base_linear_velocity_yaw_frame=base_lin_vel_yaw,
                projected_gravity_b=projected_gravity_b,
                base_height=self._terrain_relative_base_height(),
                joint_velocity=to_torch(self.robot.data.joint_vel),
                joint_acc=to_torch(self.robot.data.joint_acc),
                applied_torque=to_torch(self.robot.data.applied_torque),
                applied_action=self.action_processor.applied_action,
                previous_applied_action=self.action_processor.previous_applied_action,
                terminated=terminated,
                feet_air_time=torch.zeros(self.num_envs, device=self.device),
                feet_phase=(
                    self._feet_phase_reward() if rewards_cfg.feet_phase != 0.0 else zero
                ),
                pose=self._pose_penalty() if rewards_cfg.pose != 0.0 else zero,
                close_feet_xy=(
                    self._close_feet_xy_penalty()
                    if rewards_cfg.close_feet_xy != 0.0
                    else zero
                ),
                feet_orientation=(
                    self._feet_orientation_penalty()
                    if rewards_cfg.feet_orientation != 0.0
                    else zero
                ),
                gait=self._feet_gait_reward() if rewards_cfg.gait != 0.0 else zero,
                feet_clearance=(
                    self._feet_clearance_reward()
                    if rewards_cfg.feet_clearance != 0.0
                    else zero
                ),
                feet_slide=(
                    self._feet_slide_penalty() if rewards_cfg.feet_slide != 0.0 else zero
                ),
                undesired_contacts=(
                    self._undesired_contact_penalty()
                    if rewards_cfg.undesired_contacts != 0.0
                    else zero
                ),
                dof_pos_limits=(
                    self._joint_position_limit_penalty()
                    if rewards_cfg.dof_pos_limits != 0.0
                    else zero
                ),
                joint_deviation_hip=(
                    self._joint_deviation_l1(self._reward_hip_joint_indices)
                    if rewards_cfg.joint_deviation_hip != 0.0
                    else zero
                ),
                joint_deviation_arms=(
                    self._joint_deviation_l1(self._reward_arm_joint_indices)
                    if rewards_cfg.joint_deviation_arms != 0.0
                    else zero
                ),
                joint_deviation_torso=(
                    self._joint_deviation_l1(self._reward_torso_joint_indices)
                    if rewards_cfg.joint_deviation_torso != 0.0
                    else zero
                ),
            ),
            rewards_cfg,
        )

    def _flat_locomotion_reward_terms(self) -> dict[str, torch.Tensor]:
        """Evaluate the clean phase-conditioned flat-locomotion task."""

        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        anchor_lin_vel_w = to_torch(self.robot.data.body_link_lin_vel_w)[
            :, self.anchor_body_index
        ]
        base_lin_vel_b, base_ang_vel_b, projected_gravity_b = self._anchor_state_b()
        joint_pos = to_torch(self.robot.data.joint_pos)[:, self._reward_ankle_joint_indices]
        limits = to_torch(self.robot.data.soft_joint_pos_limits)[
            :, self._reward_ankle_joint_indices
        ]
        below = (limits[:, :, 0] - joint_pos).clamp(min=0.0)
        above = (joint_pos - limits[:, :, 1]).clamp(min=0.0)
        foot_position_w = self._foot_contact_point_position_w()
        ground_height = self._terrain_ground_height_at(foot_position_w[..., :2])
        feet_height = foot_position_w[..., 2] - ground_height
        current_contact_time = to_torch(self.contact_sensor.data.current_contact_time)[
            :, self._foot_contact_sensor_indices
        ]
        terminated = getattr(self, "reset_terminated", None)
        if terminated is None:
            terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        inputs = FlatLocomotionRewardInputs(
            commands=self.command_generator.commands,
            base_linear_velocity_b=base_lin_vel_b,
            base_angular_velocity_b=base_ang_vel_b,
            base_linear_velocity_yaw_frame=quat_apply_inverse(
                yaw_quat(anchor_quat_w), anchor_lin_vel_w
            ),
            projected_gravity_b=projected_gravity_b,
            base_height=self._terrain_relative_base_height(),
            gait_phase=self._gait_phase(),
            feet_height=feet_height,
            feet_contact=current_contact_time > 0.0,
            leg_lateral_separation=leg_lateral_separation(
                foot_position_w,
                to_torch(self.robot.data.body_link_pos_w)[:, self._knee_body_indices],
                anchor_quat_w,
            ),
            joint_acc=to_torch(self.robot.data.joint_acc)[
                :, self._reward_leg_joint_indices
            ],
            applied_torque=to_torch(self.robot.data.applied_torque)[
                :, self._reward_leg_joint_indices
            ],
            action=self._policy_action,
            previous_action=self._previous_policy_action,
            terminated=terminated,
            pose=self._pose_penalty(),
            feet_slide=self._feet_slide_penalty(),
            dof_pos_limits=(below + above).sum(-1),
        )
        self._feet_air_time_positive_biped_reward()  # Support diagnostics only.
        gait = phase_gait_signals(inputs, self.cfg.rewards)
        moving = (
            inputs.commands.abs()
            > float(self.cfg.rewards.command_activity_threshold)
        ).any(dim=-1)
        moving_count = moving.sum().clamp(min=1).to(inputs.commands.dtype)

        def moving_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * moving).sum() / moving_count

        log = self.extras.setdefault("log", {})
        for index, name in enumerate(("feet", "knees")):
            width = inputs.leg_lateral_separation[:, index]
            minimum = self.cfg.rewards.leg_min_lateral_separation[index]
            log[f"Gait/{name}_signed_width_mean_m"] = width.mean().detach()
            log[f"Gait/{name}_clearance_violation_fraction"] = (width < minimum).float().mean().detach()
            log[f"Gait/{name}_crossing_fraction"] = (width < 0.0).float().mean().detach()
        log["Gait/phase_contact_match_fraction"] = moving_mean(
            gait["phase_contact_match"]
        ).detach()
        log["Gait/swing_contact_violation_fraction"] = moving_mean(
            gait["swing_contact_penalty"]
        ).detach()
        log["Gait/stance_missing_contact_fraction"] = moving_mean(
            gait["stance_missing_contact_penalty"]
        ).detach()
        log["Gait/unexpected_double_support_fraction"] = moving_mean(
            gait["unexpected_double_support_penalty"]
        ).detach()
        log["Gait/swing_height_error_mean"] = moving_mean(
            gait["swing_foot_height_l2"]
        ).detach()
        log["Gait/target_single_stance_fraction"] = (
            gait["target_single_stance"].sum() / moving_count
        ).detach()
        log["Gait/target_double_support_fraction"] = (
            gait["target_double_support"].sum() / moving_count
        ).detach()
        return compute_flat_locomotion_reward_terms(inputs, self.cfg.rewards)

    def _update_command_curriculum(self, terms: dict[str, torch.Tensor]) -> None:
        cfg = self.cfg.command
        if not bool(cfg.curriculum_enabled):
            return
        linear_weight = float(self.cfg.rewards.track_lin_vel_xy_exp)
        yaw_weight = float(self.cfg.rewards.track_ang_vel_z_exp)
        if linear_weight <= 0.0 and yaw_weight <= 0.0:
            return
        if linear_weight > 0.0:
            self._command_curriculum_linear_sum += (
                terms["track_lin_vel_xy_exp"].mean().detach() / linear_weight
            )
        if yaw_weight > 0.0:
            self._command_curriculum_yaw_sum += (
                terms["track_ang_vel_z_exp"].mean().detach() / yaw_weight
            )
        self._command_curriculum_steps += 1
        if self._command_curriculum_steps >= int(self.max_episode_length):
            divisor = float(self._command_curriculum_steps)
            linear_score = float((self._command_curriculum_linear_sum / divisor).item())
            yaw_score = float((self._command_curriculum_yaw_sum / divisor).item())
            self.command_generator.update_curriculum(
                linear_score=linear_score,
                yaw_score=yaw_score,
            )
            self._command_curriculum_linear_sum.zero_()
            self._command_curriculum_yaw_sum.zero_()
            self._command_curriculum_steps = 0
            log = self.extras.setdefault("log", {})
            log["Curriculum/linear_tracking_score"] = linear_score
            log["Curriculum/yaw_tracking_score"] = yaw_score

        log = self.extras.setdefault("log", {})
        log["Curriculum/lin_vel_x_max"] = self.command_generator.current_linear_x_range[1]
        log["Curriculum/lin_vel_y_max"] = self.command_generator.current_linear_y_range[1]
        log["Curriculum/yaw_rate_max"] = self.command_generator.current_yaw_rate_range[1]

    def _get_rewards(self) -> torch.Tensor:
        terms = self._locomotion_reward_terms()
        self._update_command_curriculum(terms)
        log = self.extras.setdefault("log", {})
        for name, value in terms.items():
            log[f"Reward/{name}"] = value.mean().detach()

        # Keep the three commanded axes visible independently.  The reward has
        # intentionally different longitudinal/lateral bandwidths, so the
        # combined XY term alone cannot reveal a policy that only tracks x.
        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        anchor_lin_vel_w = to_torch(self.robot.data.body_link_lin_vel_w)[
            :, self.anchor_body_index
        ]
        anchor_lin_vel_yaw = quat_apply_inverse(
            yaw_quat(anchor_quat_w), anchor_lin_vel_w
        )
        anchor_ang_vel_b = quat_apply_inverse(
            anchor_quat_w,
            to_torch(self.robot.data.body_link_ang_vel_w)[:, self.anchor_body_index],
        )
        commands = self.command_generator.commands
        log["Tracking/lin_vel_x_abs_error"] = (
            commands[:, 0] - anchor_lin_vel_yaw[:, 0]
        ).abs().mean().detach()
        log["Tracking/lin_vel_y_abs_error"] = (
            commands[:, 1] - anchor_lin_vel_yaw[:, 1]
        ).abs().mean().detach()
        log["Tracking/yaw_rate_abs_error"] = (
            commands[:, 2] - anchor_ang_vel_b[:, 2]
        ).abs().mean().detach()
        category_ids = getattr(self.command_generator, "category_ids", None)
        category_names = getattr(self.command_generator, "category_names", ())
        if category_ids is not None:
            scales = torch.tensor(
                (*self.cfg.rewards.linear_velocity_scales, self.cfg.rewards.yaw_rate_scale),
                device=self.device,
            )
            normalized_error = torch.stack(
                (
                    (commands[:, 0] - anchor_lin_vel_yaw[:, 0]).abs(),
                    (commands[:, 1] - anchor_lin_vel_yaw[:, 1]).abs(),
                    (commands[:, 2] - anchor_ang_vel_b[:, 2]).abs(),
                ),
                dim=-1,
            ) / scales
            for category, name in enumerate(category_names):
                selected = category_ids == category
                log[f"Command/{name}_fraction"] = selected.float().mean().detach()
                if torch.any(selected):
                    log[f"Tracking/{name}_normalized_mae"] = (
                        normalized_error[selected].mean().detach()
                    )
        return torch.stack(tuple(terms.values()), dim=-1).sum(dim=-1) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, projected_gravity_b = self._anchor_state_b()
        contact_force_history = to_torch(self.contact_sensor.data.net_forces_w_history)
        base_contact_force = torch.linalg.vector_norm(
            contact_force_history[:, :, self._termination_contact_sensor_indices], dim=-1
        ).amax(dim=(1, 2))
        base_contact = base_contact_force > float(
            self.cfg.illegal_contact_force_threshold
        )
        fallen, tilted, terminated = compute_locomotion_termination(
            terrain_relative_base_height=self._terrain_relative_base_height(),
            upright_projection=-projected_gravity_b[:, 2],
            illegal_contact=base_contact,
            minimum_base_height=float(self.cfg.minimum_base_height),
            minimum_upright_projection=float(self.cfg.minimum_upright_projection),
        )
        log = self.extras.setdefault("log", {})
        log["Termination/fallen"] = fallen.float().mean().detach()
        log["Termination/tilted"] = tilted.float().mean().detach()
        log["Termination/base_contact"] = base_contact.float().mean().detach()
        log["State/pelvis_height_mean"] = (
            self._terrain_relative_base_height().mean().detach()
        )
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
        if hasattr(self, "_policy_action"):
            self._policy_action[env_ids] = 0.0
            self._previous_policy_action[env_ids] = 0.0
        if bool(getattr(self.cfg.rewards, "gait_randomize_phase", False)):
            self._gait_phase_offset[env_ids] = torch.empty(
                env_ids.numel(), device=self.device
            ).uniform_(-torch.pi, torch.pi)
        else:
            self._gait_phase_offset[env_ids] = 0.0
        self.observation_model.reset(
            env_ids,
            self.robot,
            self.command_generator.commands,
            self._gait_phase_features(),
            self._sim_to_policy_order(self.action_processor.applied_action),
        )


__all__ = ["LocomotionEnv", "compute_locomotion_termination"]

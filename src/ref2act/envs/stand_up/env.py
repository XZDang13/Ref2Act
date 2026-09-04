from __future__ import annotations

import torch

from ref2act.common.math import (
    quat_apply_inverse,
    quat_from_euler_xyz,
    quaternion_to_rotation_6d,
)
from ref2act.envs.base import LeggedRobotEnv
from ref2act.isaac_compat import to_torch

from .rewards import (
    StandUpRewardInputs,
    compute_stand_up_reward_terms,
    stand_up_progress_scores,
    support_score,
)


PAIR_PROPRIOCEPTION_DIM = 55
PAIR_TARGET_DIM = 23
STAND_UP_POLICY_DIM = PAIR_PROPRIOCEPTION_DIM + PAIR_TARGET_DIM
STAND_UP_PRIVILEGED_DIM = 9
STAND_UP_CRITIC_DIM = STAND_UP_POLICY_DIM + STAND_UP_PRIVILEGED_DIM


def stand_up_observation_contract() -> dict[str, tuple[str, ...] | int]:
    """Describe the exact actor/critic layout used by downstream adapters."""

    return {
        "policy_dim": STAND_UP_POLICY_DIM,
        "critic_dim": STAND_UP_CRITIC_DIM,
        "policy_terms": (
            "anchor_orientation_6d[6]",
            "anchor_angular_velocity_body[3]",
            "joint_position_policy_order[23]",
            "joint_velocity_policy_order[23]",
            "clamped_q_target_policy_order[23]",
        ),
        "critic_privileged_terms": (
            "anchor_linear_velocity_body[3]",
            "root_height[1]",
            "mean_shoulder_height[1]",
            "left_foot_support[1]",
            "right_foot_support[1]",
            "non_foot_support[1]",
            "episode_progress[1]",
        ),
    }


class StandUpEnv(LeggedRobotEnv):
    """Mostly-supine G1 stand-up task with a PAIR-compatible actor contract."""

    def __init__(self, cfg=None, render_mode=None, cfg_factory: str | None = None, **kwargs):
        if cfg is None and cfg_factory is not None:
            from ref2act.envs.base import resolve_cfg_factory

            cfg = resolve_cfg_factory(cfg_factory)
            cfg_factory = None
        if cfg is None:
            raise ValueError("Either cfg or cfg_factory must be provided.")
        if cfg.action.mode != "offset":
            raise ValueError("Stand-up requires Ref2Act offset actions.")
        if (
            int(cfg.action.buffer_length) != 1
            or cfg.action.latency_range is not None
            or float(cfg.action.noise_scale) != 0.0
        ):
            raise ValueError("Stand-up requires zero action delay and noise.")
        if cfg.events is not None:
            raise ValueError("Stand-up does not permit domain randomization events.")
        if not 0.0 <= float(cfg.standing_reset_fraction) <= 1.0:
            raise ValueError("standing_reset_fraction must be in [0, 1].")
        if float(cfg.rise_reference_shoulder_height) >= float(
            cfg.target_shoulder_height
        ):
            raise ValueError("rise shoulder reference must be below its target.")
        if float(cfg.rise_reference_root_height) >= float(cfg.target_root_height):
            raise ValueError("rise root reference must be below its target.")
        if float(cfg.rise_reference_upright_projection) >= float(
            cfg.rewards.rise_target_upright
        ):
            raise ValueError("rise upright reference must be below its target.")
        if not 0.0 <= float(cfg.assistance_initial_max_gravity_ratio) <= 1.0:
            raise ValueError("assistance_initial_max_gravity_ratio must be in [0, 1].")
        if not 0.0 <= float(cfg.assistance_zero_probability) <= 1.0:
            raise ValueError("assistance_zero_probability must be in [0, 1].")
        if not 0.0 <= float(cfg.assistance_minimum_force_fraction) <= 1.0:
            raise ValueError("assistance_minimum_force_fraction must be in [0, 1].")
        if float(cfg.assistance_full_duration_s) < 0.0:
            raise ValueError("assistance_full_duration_s must be non-negative.")
        if float(cfg.assistance_fade_duration_s) <= 0.0:
            raise ValueError("assistance_fade_duration_s must be positive.")
        assistance_end = float(cfg.assistance_full_duration_s) + float(
            cfg.assistance_fade_duration_s
        )
        if assistance_end >= float(cfg.episode_length_s):
            raise ValueError("assistance must end before the episode does.")
        cfg.observation_space = {
            "policy": STAND_UP_POLICY_DIM,
            "critic": STAND_UP_CRITIC_DIM,
        }
        super().__init__(cfg, render_mode, cfg_factory=cfg_factory, **kwargs)

        shoulder_names = ["left_shoulder_roll_link", "right_shoulder_roll_link"]
        shoulder_ids, found = self.robot.find_bodies(shoulder_names, preserve_order=True)
        if list(found) != shoulder_names:
            raise RuntimeError(f"Stand-up requires ordered shoulder bodies; got {found}.")
        self._shoulder_body_indices = torch.tensor(
            shoulder_ids, device=self.device, dtype=torch.long
        )

        foot_sensor_ids, found = self.contact_sensor.find_bodies(
            list(self.robot_spec.foot_bodies), preserve_order=True
        )
        if len(foot_sensor_ids) != 2:
            raise RuntimeError(f"Stand-up requires both foot contact bodies; got {found}.")
        self._foot_contact_sensor_indices = torch.tensor(
            foot_sensor_ids, device=self.device, dtype=torch.long
        )
        non_foot_ids, found = self.contact_sensor.find_bodies(
            self.cfg.non_foot_support_body_names, preserve_order=True
        )
        if not non_foot_ids:
            raise RuntimeError(
                "non_foot_support_body_names matched no contact-sensor bodies: "
                f"{self.cfg.non_foot_support_body_names}; got {found}."
            )
        self._non_foot_contact_sensor_indices = torch.tensor(
            non_foot_ids, device=self.device, dtype=torch.long
        )

        assistance_body_name = str(self.cfg.assistance_body_name)
        assistance_body_ids, found = self.robot.find_bodies(
            [assistance_body_name], preserve_order=True
        )
        if list(found) != [assistance_body_name]:
            raise RuntimeError(
                f"Stand-up assistance requires {assistance_body_name!r}; got {found}."
            )
        self._assistance_body_indices = torch.tensor(
            assistance_body_ids, device=self.device, dtype=torch.long
        )
        body_mass = to_torch(self.robot.data.body_mass)
        gravity_magnitude = torch.linalg.vector_norm(
            torch.tensor(self.cfg.sim.gravity, device=self.device, dtype=body_mass.dtype)
        )
        if gravity_magnitude <= 0.0:
            raise ValueError("stand-up assistance requires non-zero simulator gravity")
        self._robot_weight = body_mass.sum(dim=-1) * gravity_magnitude
        self._assistance_max_gravity_ratio = float(
            self.cfg.assistance_initial_max_gravity_ratio
        )
        self._fixed_assistance_force_newtons: float | None = None
        self._episode_assistance_force = torch.zeros_like(self._robot_weight)
        self._assistance_force_w = torch.zeros(
            self.num_envs, 1, 3, device=self.device, dtype=body_mass.dtype
        )
        self._assistance_torque_w = torch.zeros_like(self._assistance_force_w)
        self._last_assistance_force = torch.zeros_like(
            self._robot_weight
        )

        self._stable_hold_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._completed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._completion_event = torch.zeros_like(self._completed)
        self._stable_stand = torch.zeros_like(self._completed)
        self._unsafe_termination = torch.zeros_like(self._completed)
        self._episode_outcome = torch.zeros_like(self._completed)
        self._standing_reset = torch.zeros_like(self._completed)

    def set_assistance_ratio(self, gravity_ratio: float) -> None:
        """Set a fixed replay force as a fraction of mean robot weight."""

        ratio = float(gravity_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("assistance gravity ratio must be in [0, 1]")
        self.set_assistance_force_newtons(
            float(self._robot_weight.mean().item()) * ratio
        )

    def set_assistance_force_newtons(self, force_newtons: float) -> None:
        """Set an exact replay force while retaining the time-based fade-out."""

        force = float(force_newtons)
        if force < 0.0:
            raise ValueError("assistance force must be non-negative")
        self._fixed_assistance_force_newtons = force
        self._episode_assistance_force.fill_(force)
        self._episode_assistance_force[self._standing_reset] = 0.0

    def set_assistance_max_ratio(self, gravity_ratio: float) -> None:
        """Set the training envelope used by subsequent episode resets."""

        ratio = float(gravity_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("assistance maximum gravity ratio must be in [0, 1]")
        self._fixed_assistance_force_newtons = None
        self._assistance_max_gravity_ratio = ratio

    def _episode_progress(self) -> torch.Tensor:
        return (
            self.episode_length_buf.to(dtype=self._robot_weight.dtype)
            / max(float(self.max_episode_length - 1), 1.0)
        ).clamp(0.0, 1.0)

    def _apply_assistance(self) -> None:
        elapsed_s = self.episode_length_buf.to(self._robot_weight.dtype) * float(
            self.step_dt
        )
        fade_start = float(self.cfg.assistance_full_duration_s)
        fade_duration = float(self.cfg.assistance_fade_duration_s)
        fade_progress = ((elapsed_s - fade_start) / fade_duration).clamp(0.0, 1.0)
        # Cosine reaches zero with no force discontinuity at either boundary.
        time_gate = 0.5 * (1.0 + torch.cos(torch.pi * fade_progress))
        time_gate = torch.where(
            elapsed_s >= fade_start + fade_duration,
            torch.zeros_like(time_gate),
            time_gate,
        )
        force = self._episode_assistance_force * time_gate
        self._assistance_force_w.zero_()
        self._assistance_force_w[:, 0, 2] = force
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._assistance_force_w,
            torques=self._assistance_torque_w,
            body_ids=self._assistance_body_indices,
            is_global=True,
        )
        self._last_assistance_force.copy_(force)

    def _anchor_state_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        linear_velocity_b = quat_apply_inverse(
            anchor_quat_w,
            to_torch(self.robot.data.body_link_lin_vel_w)[:, self.anchor_body_index],
        )
        angular_velocity_b = quat_apply_inverse(
            anchor_quat_w,
            to_torch(self.robot.data.body_link_ang_vel_w)[:, self.anchor_body_index],
        )
        projected_gravity_b = quat_apply_inverse(
            anchor_quat_w, to_torch(self.robot.data.GRAVITY_VEC_W)
        )
        return linear_velocity_b, angular_velocity_b, projected_gravity_b

    def _terrain_relative_heights(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins_z = to_torch(self.scene.env_origins)[:, 2]
        root_height = (
            to_torch(self.robot.data.body_link_pos_w)[:, self.root_body_index, 2]
            - origins_z
        )
        shoulder_height = (
            to_torch(self.robot.data.body_link_pos_w)[
                :, self._shoulder_body_indices, 2
            ].mean(dim=-1)
            - origins_z
        )
        return root_height, shoulder_height

    def _support_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        force_history = to_torch(self.contact_sensor.data.net_forces_w_history)
        foot_force_magnitude = torch.linalg.vector_norm(
            force_history[:, :, self._foot_contact_sensor_indices], dim=-1
        ).amax(dim=1)
        foot_vertical_force = force_history[
            :, :, self._foot_contact_sensor_indices, 2
        ].clamp_min(0.0).amax(dim=1)
        non_foot_force_magnitude = torch.linalg.vector_norm(
            force_history[:, :, self._non_foot_contact_sensor_indices], dim=-1
        ).amax(dim=(1, 2))
        foot_contact = foot_force_magnitude > float(self.cfg.support_force_threshold)
        non_foot_contact = non_foot_force_magnitude > float(
            self.cfg.support_force_threshold
        )
        support_scale = float(self.cfg.support_force_observation_scale)
        foot_support = (foot_force_magnitude / support_scale).clamp(0.0, 1.0)
        non_foot_support = (non_foot_force_magnitude / support_scale).clamp(0.0, 1.0)
        observation = torch.cat(
            (foot_support, non_foot_support.unsqueeze(-1)), dim=-1
        )
        return foot_contact, non_foot_contact, observation, foot_vertical_force

    def _settling(self) -> torch.Tensor:
        return self.episode_length_buf < int(self.cfg.settling_steps)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._apply_assistance()
        safe_actions = torch.where(
            self._settling().unsqueeze(-1), torch.zeros_like(actions), actions
        )
        self.action_processor.pre_process_action(
            self._policy_to_sim_order(safe_actions)
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        anchor_quat_w = to_torch(self.robot.data.body_link_quat_w)[
            :, self.anchor_body_index
        ]
        linear_velocity_b, angular_velocity_b, _ = self._anchor_state_b()
        policy = torch.cat(
            (
                quaternion_to_rotation_6d(anchor_quat_w),
                angular_velocity_b,
                self._sim_to_policy_order(to_torch(self.robot.data.joint_pos)),
                self._sim_to_policy_order(to_torch(self.robot.data.joint_vel)),
                self._sim_to_policy_order(
                    self.action_processor.target_joint_position
                ),
            ),
            dim=-1,
        )
        root_height, shoulder_height = self._terrain_relative_heights()
        _, _, support, _ = self._support_state()
        progress = self._episode_progress().to(policy.dtype)
        privilege = torch.cat(
            (
                linear_velocity_b,
                root_height.unsqueeze(-1),
                shoulder_height.unsqueeze(-1),
                support,
                progress.unsqueeze(-1),
            ),
            dim=-1,
        )
        critic = torch.cat((policy, privilege), dim=-1)
        if policy.shape[-1] != STAND_UP_POLICY_DIM:
            raise RuntimeError(f"Expected {STAND_UP_POLICY_DIM} policy values, got {policy.shape[-1]}.")
        if critic.shape[-1] != STAND_UP_CRITIC_DIM:
            raise RuntimeError(f"Expected {STAND_UP_CRITIC_DIM} critic values, got {critic.shape[-1]}.")
        return {"policy": policy, "critic": critic}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_height, shoulder_height = self._terrain_relative_heights()
        linear_velocity_b, angular_velocity_b, projected_gravity_b = self._anchor_state_b()
        _, non_foot_contact, _, foot_vertical_force = self._support_state()
        joint_velocity = to_torch(self.robot.data.joint_vel)
        foot_load_ratio = foot_vertical_force / self._robot_weight.unsqueeze(-1).clamp_min(
            1.0e-6
        )
        physical_foot_support = (
            (foot_load_ratio.sum(dim=-1) >= float(self.cfg.success_total_foot_load_ratio))
            & (
                foot_load_ratio.amin(dim=-1)
                >= float(self.cfg.success_each_foot_load_ratio)
            )
        )
        unassisted = self._last_assistance_force <= 1.0e-6
        height_ratio = torch.minimum(
            root_height / float(self.cfg.target_root_height),
            shoulder_height / float(self.cfg.target_shoulder_height),
        )
        upright = -projected_gravity_b[:, 2]
        self._stable_stand = (
            (height_ratio >= float(self.cfg.success_height_ratio))
            & (upright >= float(self.cfg.success_upright_projection))
            & (torch.linalg.vector_norm(linear_velocity_b, dim=-1) <= float(self.cfg.success_linear_velocity))
            & (torch.linalg.vector_norm(angular_velocity_b, dim=-1) <= float(self.cfg.success_angular_velocity))
            & (joint_velocity.abs().amax(dim=-1) <= float(self.cfg.success_joint_velocity))
            & physical_foot_support
            & (~non_foot_contact)
            & unassisted
            & (~self._settling())
        )
        self._stable_hold_steps = torch.where(
            self._stable_stand,
            self._stable_hold_steps + 1,
            torch.zeros_like(self._stable_hold_steps),
        )
        newly_complete = self._stable_hold_steps >= int(self.cfg.success_hold_steps)
        self._completion_event = newly_complete & (~self._completed)
        self._completed |= newly_complete

        state_tensors = (
            root_height,
            shoulder_height,
            linear_velocity_b,
            angular_velocity_b,
            joint_velocity,
        )
        finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        for value in state_tensors:
            value_finite = torch.isfinite(value)
            if value.ndim > 1:
                value_finite = value_finite.flatten(1).all(dim=-1)
            finite &= value_finite
        self._unsafe_termination = (
            (~finite)
            | (torch.linalg.vector_norm(linear_velocity_b, dim=-1) > float(self.cfg.maximum_linear_velocity))
            | (torch.linalg.vector_norm(angular_velocity_b, dim=-1) > float(self.cfg.maximum_angular_velocity))
            | (joint_velocity.abs().amax(dim=-1) > float(self.cfg.maximum_joint_velocity))
        )
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        log = self.extras.setdefault("log", {})
        log["Success/stable_fraction"] = self._stable_stand.float().mean().detach()
        log["Success/completion_event_fraction"] = self._completion_event.float().mean().detach()
        log["Success/ever_completed_fraction"] = self._completed.float().mean().detach()
        episode_outcome = self._unsafe_termination | time_out
        self._episode_outcome.copy_(episode_outcome)
        log["Success/episode_outcome_events"] = episode_outcome.float().sum().detach()
        log["Success/episode_success_events"] = (
            episode_outcome & self._completed & (~self._standing_reset)
        ).float().sum().detach()
        log["Success/unassisted_fraction"] = unassisted.float().mean().detach()
        log["Contact/total_foot_load_ratio_mean"] = (
            foot_load_ratio.sum(dim=-1).mean().detach()
        )
        log["Contact/min_foot_load_ratio_mean"] = (
            foot_load_ratio.amin(dim=-1).mean().detach()
        )
        log["State/root_height_mean"] = root_height.mean().detach()
        log["State/shoulder_height_mean"] = shoulder_height.mean().detach()
        log["State/upright_projection_mean"] = upright.mean().detach()
        log["Termination/unsafe_fraction"] = self._unsafe_termination.float().mean().detach()
        return self._unsafe_termination, time_out

    def _get_rewards(self) -> torch.Tensor:
        root_height, shoulder_height = self._terrain_relative_heights()
        linear_velocity_b, angular_velocity_b, projected_gravity_b = self._anchor_state_b()
        upright = -projected_gravity_b[:, 2]
        target_root_height = torch.full_like(root_height, float(self.cfg.target_root_height))
        target_shoulder_height = torch.full_like(
            shoulder_height, float(self.cfg.target_shoulder_height)
        )
        foot_contact, non_foot_contact, _, foot_vertical_force = self._support_state()
        foot_load_ratio = foot_vertical_force / self._robot_weight.unsqueeze(-1).clamp_min(
            1.0e-6
        )
        unassisted = self._last_assistance_force <= 1.0e-6
        reward_inputs = StandUpRewardInputs(
            root_height=root_height,
            shoulder_height=shoulder_height,
            rise_reference_root_height=torch.full_like(
                root_height, float(self.cfg.rise_reference_root_height)
            ),
            rise_reference_shoulder_height=torch.full_like(
                shoulder_height, float(self.cfg.rise_reference_shoulder_height)
            ),
            target_root_height=target_root_height,
            target_shoulder_height=target_shoulder_height,
            upright_projection=upright,
            rise_reference_upright_projection=torch.full_like(
                upright, float(self.cfg.rise_reference_upright_projection)
            ),
            base_linear_velocity_b=linear_velocity_b,
            base_angular_velocity_b=angular_velocity_b,
            joint_position=to_torch(self.robot.data.joint_pos),
            default_joint_position=to_torch(self.robot.data.default_joint_pos),
            soft_joint_position_limits=to_torch(self.robot.data.soft_joint_pos_limits),
            joint_velocity=to_torch(self.robot.data.joint_vel),
            joint_velocity_limits=to_torch(self.robot.data.joint_vel_limits),
            applied_torque=to_torch(self.robot.data.applied_torque),
            joint_effort_limits=to_torch(self.robot.data.joint_effort_limits),
            target_joint_position=self.action_processor.target_joint_position,
            previous_target_joint_position=self.action_processor.scale_action(
                self.action_processor.previous_applied_action
            ),
            left_foot_load_ratio=foot_load_ratio[:, 0],
            right_foot_load_ratio=foot_load_ratio[:, 1],
            non_foot_contact=non_foot_contact,
            unassisted=unassisted,
            completion_event=self._completion_event,
            standing_reset=self._standing_reset,
            unsafe_termination=self._unsafe_termination,
            settling=self._settling(),
        )
        terms = compute_stand_up_reward_terms(
            reward_inputs,
            self.cfg.rewards,
            step_dt=float(self.step_dt),
        )
        root_score, shoulder_score, upright_score, rise_score = stand_up_progress_scores(
            reward_inputs, self.cfg.rewards
        )
        physical_support_score = support_score(reward_inputs, self.cfg.rewards)
        log = self.extras.setdefault("log", {})
        for name, value in terms.items():
            log[f"Reward/{name}"] = value.mean().detach()
        log["Reward/rise_root_score"] = root_score.mean().detach()
        log["Reward/rise_shoulder_score"] = shoulder_score.mean().detach()
        log["Reward/rise_upright_score"] = upright_score.mean().detach()
        log["Reward/rise_score"] = rise_score.mean().detach()
        log["Reward/support_score"] = physical_support_score.mean().detach()
        log["Assistance/max_gravity_ratio"] = float(
            self._assistance_max_gravity_ratio
        )
        log["Assistance/episode_force_newtons_mean"] = (
            self._episode_assistance_force.mean().detach()
        )
        log["Assistance/episode_gravity_ratio_mean"] = (
            self._episode_assistance_force / self._robot_weight.clamp_min(1.0e-6)
        ).mean().detach()
        log["Assistance/zero_episode_fraction"] = (
            (self._episode_assistance_force <= 1.0e-6).float().mean().detach()
        )
        log["Assistance/applied_gravity_ratio_mean"] = (
            self._last_assistance_force / self._robot_weight.clamp_min(1.0e-6)
        ).mean().detach()
        log["Assistance/force_newtons_mean"] = (
            self._last_assistance_force.mean().detach()
        )
        log["Assistance/active_fraction"] = (
            (self._last_assistance_force > 1.0e-6).float().mean().detach()
        )
        target = self.action_processor.target_joint_position
        limits = to_torch(self.robot.data.joint_pos_limits)
        at_limit = (target <= limits[..., 0] + 1.0e-5) | (
            target >= limits[..., 1] - 1.0e-5
        )
        raw_action = self.action_processor.applied_action
        log["Action/raw_abs_gt_1_fraction"] = (raw_action.abs() > 1.0).float().mean().detach()
        log["Action/raw_abs_gt_3_fraction"] = (raw_action.abs() > 3.0).float().mean().detach()
        log["Action/q_target_clamp_fraction"] = at_limit.float().mean().detach()
        log["Contact/both_feet_support_fraction"] = foot_contact.all(dim=-1).float().mean().detach()
        log["Contact/non_foot_support_fraction"] = non_foot_contact.float().mean().detach()
        log["Reset/standing_fraction"] = self._standing_reset.float().mean().detach()
        return torch.stack(tuple(terms.values()), dim=-1).sum(dim=-1)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._reset_common(env_ids, joint_position_noise=0.0)
        default_root_state = to_torch(self.robot.data.default_root_state)[env_ids]
        root_state = default_root_state.clone()
        origins = to_torch(self.scene.env_origins)[env_ids]
        root_state[:, :3] = to_torch(self.scene.env_origins)[env_ids]
        root_state[:, :3] += torch.tensor(
            self.cfg.initial_root_position,
            device=self.device,
            dtype=root_state.dtype,
        )
        count = env_ids.numel()
        roll = torch.full((count,), float(self.cfg.initial_root_euler_xyz[0]), device=self.device)
        pitch = torch.full((count,), float(self.cfg.initial_root_euler_xyz[1]), device=self.device)
        yaw = torch.full((count,), float(self.cfg.initial_root_euler_xyz[2]), device=self.device)
        root_state[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        root_state[:, 7:] = 0.0
        standing = torch.rand(count, device=self.device) < float(
            self.cfg.standing_reset_fraction
        )
        root_state[standing, :3] = (
            origins[standing] + default_root_state[standing, :3]
        )
        root_state[standing, 3:7] = default_root_state[standing, 3:7]
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self._standing_reset[env_ids] = standing
        if self._fixed_assistance_force_newtons is None:
            minimum = float(self.cfg.assistance_minimum_force_fraction)
            fractions = minimum + (1.0 - minimum) * torch.rand(
                count, device=self.device, dtype=self._robot_weight.dtype
            )
            enabled = torch.rand(count, device=self.device) >= float(
                self.cfg.assistance_zero_probability
            )
            sampled_force = (
                self._robot_weight[env_ids]
                * float(self._assistance_max_gravity_ratio)
                * fractions
                * enabled.to(fractions.dtype)
            )
        else:
            sampled_force = torch.full(
                (count,),
                float(self._fixed_assistance_force_newtons),
                device=self.device,
                dtype=self._robot_weight.dtype,
            )
        sampled_force[standing] = 0.0
        self._episode_assistance_force[env_ids] = sampled_force
        self._last_assistance_force[env_ids] = 0.0
        self._stable_hold_steps[env_ids] = 0
        self._completed[env_ids] = False
        self._completion_event[env_ids] = False
        self._stable_stand[env_ids] = False
        self._unsafe_termination[env_ids] = False
        self._episode_outcome[env_ids] = False


__all__ = [
    "PAIR_PROPRIOCEPTION_DIM",
    "PAIR_TARGET_DIM",
    "STAND_UP_CRITIC_DIM",
    "STAND_UP_POLICY_DIM",
    "StandUpEnv",
    "stand_up_observation_contract",
]

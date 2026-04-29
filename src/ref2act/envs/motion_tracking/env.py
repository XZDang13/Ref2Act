from __future__ import annotations

import dataclasses
from importlib import import_module
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from ref2act.common.math import quat_mul
from ref2act.common.observation_spec import ObservationLayout
from ref2act.motion import MotionLib, MotionSampler, SamplingStrategy, SegmentSource

from .action import ActionProcessor
from .curriculum import TerminationThresholdCurriculum
from .observation import Observation
from .rewards import RewardSpec, Rewards
from .termination import Termination, TerminationSpec
from .tracking_quality import FallGuardResult, TrackingQualityGate, TrackingQualityResult
from .types import (
    JOINT_POSITION_RANGE,
    POSE_RANGE,
    VELOCITY_RANGE,
    ReferenceMotions,
    pose_noise,
    velocity_noise,
)
from .visualization import ReferenceMotionViewer

if TYPE_CHECKING:
    from ref2act.robots.g1 import G1MotionTrackingEnvCfg


def _resolve_cfg_factory(cfg_factory: str):
    module_name, _, attr_name = cfg_factory.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid cfg_factory: {cfg_factory!r}")
    module = import_module(module_name)
    cfg_cls = getattr(module, attr_name)
    return cfg_cls()


class MotionTrackingEnv(DirectRLEnv):
    cfg: G1MotionTrackingEnvCfg
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

    def __init__(self, cfg=None, render_mode=None, cfg_factory: str | None = None, **kwargs):
        if cfg is None:
            if cfg_factory is None:
                raise ValueError("Either cfg or cfg_factory must be provided.")
            cfg = _resolve_cfg_factory(cfg_factory)
        self._derive_observation_spaces(cfg)
        super().__init__(cfg, render_mode, **kwargs)

        self.action_processer = ActionProcessor(self.robot, self.cfg.action)
        self.action_processor = self.action_processer

        self.motion_lib = MotionLib(self.cfg.expert_motion_file, self.device)

        self.anchor_body_index = self._resolve_shared_body_index(self.cfg.anchor_body_name)
        self.key_body_indices = [self._resolve_shared_body_index(name) for name in self.cfg.key_body_names]
        self.end_effector_body_indices = [
            self._resolve_shared_body_index(name) for name in self.cfg.end_effector_body_names
        ]
        foot_body_indices = [self.robot.data.body_names.index(name) for name in self.cfg.foot_body_names]
        self.root_link_index = self._resolve_shared_body_index(self.cfg.root_link_name)

        collision_track_body_indices, _ = self.contact_sensor.find_bodies(self.cfg.collision_track_body_names)
        foot_contact_body_indices, _ = self.contact_sensor.find_bodies(self.cfg.foot_body_names, preserve_order=True)
        self.collision_track_body_indices = collision_track_body_indices

        sampling_strategy = self._get_sampling_strategy()
        cfg_init_failure_bins = getattr(self.cfg, "init_failure_bins", None)
        if cfg_init_failure_bins is None:
            enable_failure_bins = (
                self.cfg.segment_source == SegmentSource.Anchor
                or sampling_strategy == SamplingStrategy.FailureWeighted
                or bool(getattr(self.cfg.adaptive_sampler, "enabled", False))
            )
        else:
            enable_failure_bins = bool(cfg_init_failure_bins)

        self.sampler = MotionSampler(
            num_envs=self.cfg.scene.num_envs,
            motion_lib=self.motion_lib,
            dt=self.step_dt,
            bin_size=self.cfg.bin_size,
            failure_decay=self.cfg.failure_decay,
            failure_weight_uniform_mix=self.cfg.failure_weight_uniform_mix,
            failure_weight_max_uniform_ratio=self.cfg.failure_weight_max_uniform_ratio,
            failure_weight_exploration_bonus=self.cfg.failure_weight_exploration_bonus,
            segment_source=self.cfg.segment_source,
            adaptive_sampler=self.cfg.adaptive_sampler,
            enable_failure_bins=enable_failure_bins,
            device=self.device,
        )
        if (
            sampling_strategy == SamplingStrategy.FailureWeighted
            and not self.sampler.supports_failure_weighted_sampling
        ):
            if self.cfg.segment_source == SegmentSource.Anchor:
                raise ValueError(
                    "Failure-weighted anchor sampling requires motion clips with anchor metadata. "
                    "Reconvert the motion .npz files with `ref2act-convert --segment-method anchor`."
                )
            raise ValueError(
                "Failure-weighted sampling requires motion clips with segment metadata. "
                "Reconvert the motion .npz files with `ref2act-convert --segment-bin-size ...`."
            )

        observation_layout = ObservationLayout(
            joint_dim=int(self.robot.data.joint_pos.shape[1]),
            action_dim=int(self.robot.data.joint_pos.shape[1]),
            key_body_count=len(self.key_body_indices),
        )
        self.observation_model = Observation(
            spec=self.cfg.observation,
            layout=observation_layout,
            num_envs=self.cfg.scene.num_envs,
            device=self.device,
            anchor_body_index=self.anchor_body_index,
            key_body_indices=self.key_body_indices,
        )

        reward_spec = self._build_reward_spec(
            collision_track_body_indices=tuple(int(index) for index in collision_track_body_indices),
            foot_body_indices=tuple(int(index) for index in foot_body_indices),
            foot_contact_body_indices=tuple(int(index) for index in foot_contact_body_indices),
        )
        self.reward_model = Rewards(reward_spec)

        viewer_enabled = getattr(self.cfg, "reference_motion_viewer_enabled", True)
        self.reference_motion_viewer = ReferenceMotionViewer(self.key_body_indices) if viewer_enabled else None
        self.termination_model = Termination(self._build_termination_spec())
        self.termination_curriculum = TerminationThresholdCurriculum(
            self.termination_model,
            getattr(self.cfg, "termination_curriculum", None),
        )
        self.tracking_quality_gate = None
        self.tracking_quality_output: TrackingQualityResult | None = None
        self._tracking_quality_cache_step: int | None = None
        if self._tracking_quality_gate_enabled():
            self.tracking_quality_gate = TrackingQualityGate(
                self.cfg.robust_tracking.quality_gate,
                self.termination_model,
                num_envs=self.cfg.scene.num_envs,
                device=self.device,
            )
        self._apply_termination_curriculum(step=0)

    @staticmethod
    def _derive_observation_spaces(cfg) -> None:
        description = cfg.observation.describe(
            ObservationLayout(
                joint_dim=int(cfg.action_space),
                action_dim=int(cfg.action_space),
                key_body_count=len(cfg.key_body_names),
            )
        )
        cfg.motion_observation_space = description.group_dims.get("motion", 0)
        cfg.robot_observation_space = description.group_dims.get("robot", 0)
        cfg.critic_observation_space = description.group_dims.get("privilege", 0)
        cfg.policy_observation_space = sum(
            dim for group_name, dim in description.group_dims.items() if group_name != "privilege"
        )
        cfg.observation_space = cfg.policy_observation_space

    def _build_reward_spec(
        self,
        *,
        collision_track_body_indices: tuple[int, ...],
        foot_body_indices: tuple[int, ...],
        foot_contact_body_indices: tuple[int, ...],
    ) -> RewardSpec:
        terms = []
        for term_cfg in self.cfg.rewards.terms:
            updates = {}
            if hasattr(term_cfg, "anchor_body_index"):
                updates["anchor_body_index"] = self.anchor_body_index
            if hasattr(term_cfg, "key_body_indices"):
                updates["key_body_indices"] = tuple(self.key_body_indices)
            if hasattr(term_cfg, "end_effector_body_indices"):
                updates["end_effector_body_indices"] = tuple(self.end_effector_body_indices)
            if hasattr(term_cfg, "body_indices"):
                updates["body_indices"] = collision_track_body_indices
            if hasattr(term_cfg, "foot_body_indices"):
                updates["foot_body_indices"] = foot_body_indices
            if hasattr(term_cfg, "foot_contact_body_indices"):
                updates["foot_contact_body_indices"] = foot_contact_body_indices
            terms.append(dataclasses.replace(term_cfg, **updates) if updates else term_cfg)
        return dataclasses.replace(self.cfg.rewards, dt=self.step_dt, terms=tuple(terms))

    def _build_termination_spec(self) -> TerminationSpec:
        failure_rules = []
        for rule_cfg in self.cfg.termination.failure_rules:
            updates = {}
            if hasattr(rule_cfg, "anchor_body_index"):
                updates["anchor_body_index"] = self.anchor_body_index
            if hasattr(rule_cfg, "end_effector_body_indices"):
                updates["end_effector_body_indices"] = tuple(self.end_effector_body_indices)
            failure_rules.append(dataclasses.replace(rule_cfg, **updates) if updates else rule_cfg)
        return dataclasses.replace(self.cfg.termination, failure_rules=tuple(failure_rules))

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
                f"and motion clip ({motion_index}). ref2act requires matching body order "
                "for shared robot/reference bodies."
            )
        return robot_index

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None or len(env_ids) == self.num_envs:
            return self.robot._ALL_INDICES
        return env_ids

    def _build_reference_motion(self, env_ids: torch.Tensor) -> ReferenceMotions:
        motions = self.sampler.sample_motion_batch(
            env_ids,
            position_offsets=self.scene.env_origins[env_ids],
        )
        return ReferenceMotions(
            joint_pos=motions["joint_pos"],
            joint_vel=motions["joint_vel"],
            body_positions=motions["body_positions"],
            body_quaternions=motions["body_quaternions"],
            body_linear_velocities=motions["body_linear_velocities"],
            body_angular_velocities=motions["body_angular_velocities"],
            anchor_body_index=self.anchor_body_index,
            robot_body_positions=self.robot.data.body_pos_w[env_ids],
            robot_body_quaternions=self.robot.data.body_quat_w[env_ids],
        )

    def _apply_reset_noise(self, env_ids: torch.Tensor, reference_motion: ReferenceMotions) -> tuple[torch.Tensor, ...]:
        joint_pos = reference_motion.joint_pos.clone()
        joint_vel = reference_motion.joint_vel.clone()
        root_pos = reference_motion.body_positions[:, self.root_link_index].clone()
        root_quat = reference_motion.body_quaternions[:, self.root_link_index].clone()
        root_linear_vel = reference_motion.body_linear_velocities[:, self.root_link_index].clone()
        root_angular_vel = reference_motion.body_angular_velocities[:, self.root_link_index].clone()

        if self.cfg.add_reset_noise:
            root_pos_noise, root_quat_noise = pose_noise(len(env_ids), POSE_RANGE, root_pos.device)
            root_linear_vel_noise, root_angular_vel_noise = velocity_noise(
                len(env_ids),
                VELOCITY_RANGE,
                root_pos.device,
            )
            joint_pose_noise = torch.empty_like(joint_pos).uniform_(
                JOINT_POSITION_RANGE[0],
                JOINT_POSITION_RANGE[1],
            )
            root_pos += root_pos_noise
            root_quat = quat_mul(root_quat, root_quat_noise)
            root_linear_vel += root_linear_vel_noise
            root_angular_vel += root_angular_vel_noise
            joint_pos += joint_pose_noise

        return joint_pos, joint_vel, root_pos, root_quat, root_linear_vel, root_angular_vel

    def _initialize_robot_from_reference(self, env_ids: torch.Tensor, reference_motion: ReferenceMotions) -> None:
        joint_pos, joint_vel, root_pos, root_quat, root_linear_vel, root_angular_vel = self._apply_reset_noise(
            env_ids,
            reference_motion,
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = root_pos
        root_state[:, 2] += 0.05
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_linear_vel
        root_state[:, 10:13] = root_angular_vel

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _store_reference_motion(self, env_ids: torch.Tensor, reference_motion: ReferenceMotions) -> None:
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
        env_ids = self.robot._ALL_INDICES
        self.sampler.advance(env_ids)
        reference_motion = self._build_reference_motion(env_ids)
        self._store_reference_motion(env_ids, reference_motion)
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

    def _tracking_quality_gate_enabled(self) -> bool:
        robust_tracking = getattr(self.cfg, "robust_tracking", None)
        if robust_tracking is None or not getattr(robust_tracking, "enabled", False):
            return False
        quality_gate_cfg = getattr(robust_tracking, "quality_gate", None)
        return quality_gate_cfg is not None and getattr(quality_gate_cfg, "enabled", False)

    def get_joint_params(self):
        return {
            "joint_names": self.robot.data.joint_names,
            "joint_effort_limits": self.robot.data.joint_effort_limits[0],
            "joint_pos_limits": self.robot.data.default_joint_limits[0],
            "joint_stiffness": self.robot.data.default_joint_stiffness[0],
            "joint_damping": self.robot.data.default_joint_damping[0],
            "action_offset": self.action_processer.offset[0],
            "action_scale": self.action_processer.scale,
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
        self.scene._terrain = self.terrain

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._advance_reference_motion()
        self.action_processer.pre_process_action(actions)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.action_processer.target_joint_position)

    def _get_observations(self):
        if self.reference_motion_viewer is not None:
            self.reference_motion_viewer.visualize(self.reference_motion)
        return self.observation_model.get_default_observation(
            self.robot,
            self.reference_motion,
            self.scene,
            self.action_processer.applied_action,
        )

    def _get_rewards(self) -> torch.Tensor:
        tracking_quality = self._get_tracking_quality()
        return self.reward_model.get_task_reward(
            self.robot,
            self.reference_motion,
            self.contact_sensor,
            self.action_processer,
            tracking_quality=tracking_quality,
        )

    def _get_tracking_quality(self) -> TrackingQualityResult | None:
        if not self._tracking_quality_gate_enabled():
            return None
        if self.tracking_quality_gate is None:
            raise RuntimeError("Tracking quality gate is enabled but was not initialized.")

        step = int(self.common_step_counter)
        cached_output = getattr(self, "tracking_quality_output", None)
        if getattr(self, "_tracking_quality_cache_step", None) == step and cached_output is not None:
            return cached_output

        self._apply_termination_curriculum(step=step)
        context = self.termination_model.build_context(
            self.episode_length_buf,
            self.max_episode_length,
            self.robot,
            self.reference_motion,
            self.sampler,
        )
        self.tracking_quality_output = self.tracking_quality_gate.evaluate(context)
        self._tracking_quality_cache_step = step
        return self.tracking_quality_output

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._tracking_quality_gate_enabled():
            self._apply_termination_curriculum(step=self.common_step_counter)
            terminate, time_out = self.termination_model.get_dones(
                self.episode_length_buf,
                self.max_episode_length,
                self.robot,
                self.reference_motion,
                self.sampler,
            )
            self.sampler.record_failures(self.termination_model.terminated_env_ids)
            self._log_sampler_adaptive_stats()
            return terminate, time_out

        quality = self._get_tracking_quality()
        if quality is None:
            raise RuntimeError("Tracking quality gate is enabled but returned no output.")
        context = self.termination_model.build_context(
            self.episode_length_buf,
            self.max_episode_length,
            self.robot,
            self.reference_motion,
            self.sampler,
        )
        time_out = self.termination_model.evaluate_timeouts(context)
        fall_guard = self._evaluate_fall_guard(context)
        terminate = (
            quality.hard_tracking_failure_mask
            | quality.recovery_timeout_mask
            | fall_guard.fall_mask
        ).clone()
        self.termination_model.track_terminated_env_ids(terminate)

        record_failure_mask = quality.record_failure_mask | fall_guard.fall_mask
        record_env_ids = torch.nonzero(record_failure_mask, as_tuple=False).squeeze(-1)
        self.sampler.record_failures(record_env_ids)
        self._log_tracking_quality(quality, terminate, time_out, record_env_ids, fall_guard)
        self._log_sampler_adaptive_stats()
        return terminate, time_out

    def _evaluate_fall_guard(self, context) -> FallGuardResult:
        robust_tracking = getattr(self.cfg, "robust_tracking", None)
        fall_guard_cfg = getattr(robust_tracking, "fall_guard", None)
        fall_guard_enabled = fall_guard_cfg is not None and getattr(fall_guard_cfg, "enabled", False)
        if not fall_guard_enabled:
            episode_length_buf = getattr(context, "episode_length_buf", None)
            if episode_length_buf is None:
                episode_length_buf = self.episode_length_buf
            zeros = torch.zeros_like(episode_length_buf, dtype=torch.float32)
            return FallGuardResult(
                fall_mask=torch.zeros_like(episode_length_buf, dtype=torch.bool),
                anchor_height_drop=zeros,
                projected_gravity_error=zeros,
            )

        anchor_body_index = self.anchor_body_index
        robot_anchor_pos = context.robot.data.body_pos_w[:, anchor_body_index]
        reference_anchor_pos = context.reference_motion.body_positions[:, anchor_body_index]
        anchor_height_drop = reference_anchor_pos[:, 2] - robot_anchor_pos[:, 2]

        robot_anchor_quat = context.robot.data.body_quat_w[:, anchor_body_index]
        reference_anchor_quat = context.reference_motion.body_quaternions[:, anchor_body_index]
        robot_projected_gravity = quat_apply_inverse(robot_anchor_quat, context.robot.data.GRAVITY_VEC_W)
        reference_projected_gravity = quat_apply_inverse(reference_anchor_quat, context.robot.data.GRAVITY_VEC_W)
        projected_gravity_error = torch.abs(robot_projected_gravity[:, 2] - reference_projected_gravity[:, 2])

        fall_mask = (anchor_height_drop > float(fall_guard_cfg.max_anchor_height_drop)) & (
            projected_gravity_error > float(fall_guard_cfg.max_projected_gravity_error)
        )
        return FallGuardResult(
            fall_mask=fall_mask,
            anchor_height_drop=anchor_height_drop,
            projected_gravity_error=projected_gravity_error,
        )

    def _log_tracking_quality(
        self,
        quality: TrackingQualityResult,
        terminate: torch.Tensor,
        time_out: torch.Tensor,
        recorded_env_ids: torch.Tensor,
        fall_guard: FallGuardResult | None = None,
    ) -> None:
        score = quality.score.detach()
        self.extras["tracking_quality/score_mean"] = score.mean()
        self.extras["tracking_quality/score_p95"] = torch.quantile(score, 0.95)

        cfg = self.tracking_quality_gate.cfg if self.tracking_quality_gate is not None else None
        if cfg is None or cfg.log_quality_counts:
            self.extras["tracking_quality/ok_rate"] = (
                ~(
                    quality.soft_violation_mask
                    | quality.recovery_needed_mask
                    | quality.hard_tracking_failure_mask
                    | quality.recovery_timeout_mask
                )
            ).to(dtype=torch.float32).mean()
            self.extras["tracking_quality/soft_violation_rate"] = quality.soft_violation_mask.to(
                dtype=torch.float32
            ).mean()
            self.extras["tracking_quality/recovery_needed_rate"] = quality.recovery_needed_mask.to(
                dtype=torch.float32
            ).mean()
            self.extras["tracking_quality/hard_tracking_failure_rate"] = quality.hard_tracking_failure_mask.to(
                dtype=torch.float32
            ).mean()
            self.extras["tracking_quality/recovery_timeout_rate"] = quality.recovery_timeout_mask.to(
                dtype=torch.float32
            ).mean()

        if cfg is None or cfg.log_per_rule_errors:
            for rule_id, error in quality.per_rule_errors.items():
                self.extras[f"tracking_quality/{rule_id}_error_mean"] = error.detach().mean()

        self.extras["termination/hard_terminate_rate"] = terminate.to(dtype=torch.float32).mean()
        self.extras["termination/time_out_rate"] = time_out.to(dtype=torch.float32).mean()
        if fall_guard is not None:
            self.extras["termination/fall_guard_rate"] = fall_guard.fall_mask.to(dtype=torch.float32).mean()
            self.extras["fall_guard/anchor_height_drop_mean"] = fall_guard.anchor_height_drop.detach().mean()
            self.extras["fall_guard/projected_gravity_error_mean"] = fall_guard.projected_gravity_error.detach().mean()
        self.extras["sampler/recorded_recovery_failure_count"] = torch.as_tensor(
            recorded_env_ids.numel(),
            device=score.device,
            dtype=torch.float32,
        )

    def _log_sampler_adaptive_stats(self) -> None:
        for key, value in self.sampler.get_adaptive_stats().items():
            self.extras[f"sampler/{key}"] = value.detach() if isinstance(value, torch.Tensor) else value

    def _reset_idx(self, env_ids: torch.Tensor | None):
        env_ids = self._normalize_env_ids(env_ids)
        if self.tracking_quality_gate is not None:
            self.tracking_quality_gate.reset(env_ids)
        self.tracking_quality_output = None
        self._tracking_quality_cache_step = None
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self.action_processer.reset_action_buffer(env_ids)
        if self.cfg.action.noise_scale > 0.0:
            self.action_processer.set_random_offset_noise(env_ids)

        self.sampler.reset(
            env_ids,
            strategy=self._get_sampling_strategy(),
            temperature=self.cfg.failure_temperature,
        )
        self._log_sampler_adaptive_stats()
        reference_motion = self._build_reference_motion(env_ids)
        self._initialize_robot_from_reference(env_ids, reference_motion)
        self._store_reference_motion(env_ids, reference_motion)
        self.action_processer.set_reference_joint_position(reference_motion.joint_pos, env_ids)
        self.observation_model.reset(
            env_ids,
            self.robot,
            self.reference_motion,
            self.scene,
            self.action_processer.applied_action,
        )
        self.target_pos = self.reference_motion.joint_pos

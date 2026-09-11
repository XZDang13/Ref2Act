from __future__ import annotations


import torch

from ref2act.envs.base import LeggedRobotEnv, resolve_cfg_factory
from ref2act.isaac_compat import to_torch as _to_torch
from ref2act.common.math import quat_apply_inverse
from isaaclab.envs import DirectRLEnv
from .observation import StandUpObservation
from .recovery_reward import nominal_shoulder_height_from_relative_geometry, recovery_height_scores, recovery_upright_score
from .standup_support import hip_placement_quality, SupportContactReader, crouch_fall_state, support_quality, foot_preparation_quality, filtered_support_load, com_support_quality, sphere_support_points, bilateral_support_ready, com_preparation_quality, bilateral_load_quality, foot_placement_quality, sole_plant_quality
from .standup_episodes import EpisodeSampling
from .standup_hands import HandContactReader, hand_quality
from .standup_stages import StageInputs, stage_rewards, preparation_config, signed_torso_lean
from .static_physics import StaticPhysicsCache
from .standup_assistance import assistance_settings, draw_assistance, orientation_multiplier
from .standup_reward import assistance_height_multiplier, standup_finite_state, standup_hold_steps, standup_stable_state


class StandUpEnv(LeggedRobotEnv):
    """Recovery-based flat stand-up runtime with a fixed supine V1 reset."""

    def __init__(self, cfg=None, render_mode=None, cfg_factory=None, **kwargs):
        if cfg is None:
            cfg = resolve_cfg_factory(cfg_factory or "ref2act.robots.g1:G1StandUpEnvCfg")
        super().__init__(cfg, render_mode, **kwargs)
        self._observation_reset_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._native_observation = StandUpObservation(self)
        self._timeout_extractor = self._native_observation.extractor
        self.external_observations = False
        self._static_physics_cache = None
        self._ensure_static_physics_cache()

    def _ensure_static_physics_cache(self):
        view = self.robot.root_view
        cache = getattr(self, '_static_physics_cache', None)
        if cache is None or cache.view is not view:
            if cache is not None:
                cache.close()
            self._static_physics_cache = StaticPhysicsCache(
                view, on_invalidate=self._invalidate_parameter_dependents,
                enabled=getattr(self.cfg, 'cache_static_physics', True))

    def _invalidate_parameter_dependents(self):
        self._standup_total_mass = None
        # COM changes also affect derived link velocity and world COM buffers,
        # even when a setter runs within the current simulation timestamp.
        data = self.robot.data
        for name in ('_body_com_pose_b', '_body_com_pose_w', '_root_com_pose_w',
                     '_body_link_vel_w', '_root_link_vel_w', '_mass_matrix',
                     '_gravity_compensation_forces'):
            buffer = getattr(data, name, None)
            if buffer is not None and hasattr(buffer, 'timestamp'):
                buffer.timestamp = float('-inf')

    def invalidate_static_physics(self):
        """Call after direct USD/backend parameter edits that bypass setters."""
        self._ensure_static_physics_cache()
        self._static_physics_cache.invalidate()

    def close(self):
        cache = getattr(self, '_static_physics_cache', None)
        if cache is not None:
            cache.close()
        super().close()

    def _setup_scene(self):
        super()._setup_scene()
        if self.device != "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _get_observations(self):
        if self.external_observations:
            return {}
        result = self._native_observation.update(self._observation_reset_mask)
        self._observation_reset_mask.zero_()
        return result

    def _ensure_recovery_geometry(self) -> None:
        """Resolve shoulder links and standing-height targets once per runtime."""

        if hasattr(self, "_pair_recovery_shoulder_body_ids"):
            return
        shoulder_ids, shoulder_names = self.robot.find_bodies(
            list(self.cfg.pair_recovery_shoulder_body_names), preserve_order=True
        )
        if len(shoulder_ids) != 2:
            raise RuntimeError(
                "PAIR recovery requires exactly two shoulder bodies, got "
                f"{shoulder_names}."
            )
        self._pair_recovery_shoulder_body_ids = torch.tensor(
            shoulder_ids, dtype=torch.long, device=self.device
        )

        ratio = float(self.cfg.pair_recovery_target_height_ratio)
        default_root_height = _to_torch(self.robot.data.default_root_state)[:, 2]
        explicit_root = self.cfg.pair_recovery_explicit_root_target_height
        if explicit_root is None:
            root_target = ratio * float(default_root_height.median().item())
        else:
            root_target = float(explicit_root)

        explicit_shoulder = self.cfg.pair_recovery_explicit_shoulder_target_height
        if explicit_shoulder is None:
            body_positions = _to_torch(self.robot.data.body_link_pos_w)
            # At the first reset, PhysX body poses may still carry the USD
            # spawn translation rather than default_root_state. The relative
            # shoulder-to-root geometry is already valid, so reconstruct the
            # nominal standing height from that offset and the configured
            # default root height. Reading shoulder world Z directly produced
            # impossible targets (1.84 m for G1) and rewarded jumping toward
            # an unreachable state.
            nominal_shoulder_height = nominal_shoulder_height_from_relative_geometry(
                default_root_height=default_root_height,
                current_root_height=body_positions[:, self.root_body_index, 2],
                current_shoulder_height=body_positions[
                    :, self._pair_recovery_shoulder_body_ids, 2
                ].mean(dim=-1),
            )
            shoulder_target = ratio * float(nominal_shoulder_height.median().item())
        else:
            shoulder_target = float(explicit_shoulder)
        if root_target <= 0.0 or shoulder_target <= 0.0:
            raise RuntimeError("Automatically resolved recovery target heights are invalid.")
        if shoulder_target <= root_target:
            raise RuntimeError(
                "Recovery shoulder target must be above the root target; got "
                f"root={root_target:.3f} m, shoulder={shoulder_target:.3f} m."
            )
        self._pair_recovery_target_root_height = root_target
        self._pair_recovery_target_shoulder_height = shoulder_target

    def _recovery_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_recovery_geometry()
        _, _, projected_gravity_b = self._anchor_state_b()
        upright = -projected_gravity_b[:, 2]
        body_positions = _to_torch(self.robot.data.body_link_pos_w)
        root_height_w = body_positions[:, self.root_body_index, 2]
        shoulder_height_w = body_positions[
            :, self._pair_recovery_shoulder_body_ids, 2
        ].mean(dim=-1)
        origin_height = _to_torch(self.scene.env_origins)[:, 2]
        ground_height = origin_height
        sensor_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(getattr(self.cfg, "pair_recovery_height_sensor_enabled", False)):
            height_sensor = self.scene.sensors.get("pair_height_sensor")
            if height_sensor is None:
                raise RuntimeError("PAIR recovery height sensor is enabled but missing from the scene.")
            ray_hits = _to_torch(height_sensor.data.ray_hits_w)
            sensor_position = _to_torch(height_sensor.data.pos_w)
            hit_height = ray_hits[:, 0, 2]
            sensor_valid = torch.isfinite(hit_height)
            ground_height = torch.where(sensor_valid, hit_height, origin_height)
            root_position = _to_torch(self.robot.data.body_link_pos_w)[:, self.root_body_index]
            self._pair_height_sensor_xy_error = torch.linalg.vector_norm(
                sensor_position[:, :2] - root_position[:, :2], dim=-1
            )
        root_height = root_height_w - ground_height
        shoulder_height = shoulder_height_w - ground_height
        self._pair_height_sensor_valid = sensor_valid
        fallen = (
            (root_height < float(self.cfg.minimum_base_height))
            | (upright < float(self.cfg.minimum_upright_projection))
        )
        root_score, shoulder_score, body_height_score = recovery_height_scores(
            root_height,
            shoulder_height,
            target_root_height=float(self._pair_recovery_target_root_height),
            target_shoulder_height=float(self._pair_recovery_target_shoulder_height),
        )
        upright_score = recovery_upright_score(
            upright,
            minimum_upright_projection=float(self.cfg.pair_recovery_minimum_upright),
            target_upright_projection=float(self.cfg.pair_recovery_target_upright),
        )
        recovered = (
            (root_score >= 1.0)
            & (shoulder_score >= 1.0)
            & (upright >= float(self.cfg.pair_fall_recovery_upright))
        )
        return (
            root_height,
            shoulder_height,
            upright,
            fallen,
            recovered,
            body_height_score,
            upright_score,
        )

    def _anchor_state_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_quat_w = _to_torch(self.robot.data.body_link_quat_w)[:, self.anchor_body_index]
        anchor_lin_vel_b = quat_apply_inverse(
            anchor_quat_w,
            _to_torch(self.robot.data.body_link_lin_vel_w)[:, self.anchor_body_index],
        )
        anchor_ang_vel_b = quat_apply_inverse(
            anchor_quat_w,
            _to_torch(self.robot.data.body_link_ang_vel_w)[:, self.anchor_body_index],
        )
        projected_gravity_b = quat_apply_inverse(anchor_quat_w, _to_torch(self.robot.data.GRAVITY_VEC_W))
        return anchor_lin_vel_b, anchor_ang_vel_b, projected_gravity_b

    def _standup_settling_steps(self) -> int:
        return max(0, round(float(self.cfg.pair_standup_settling_s) / self.step_dt))

    def _standup_is_settling(self) -> torch.Tensor:
        return self.episode_length_buf < self._standup_settling_steps()

    def set_standup_assistance_ratio(self, gravity_ratio: float) -> None:
        gravity_ratio = float(gravity_ratio)
        maximum = float(self.cfg.pair_standup_assistance_max_gravity_ratio)
        if not 0.0 <= gravity_ratio <= maximum + 1.0e-8:
            raise ValueError("Stand-up assistance gravity ratio is outside its configured range.")
        if gravity_ratio > 0.0 and not bool(self.cfg.pair_standup_assistance_enabled):
            raise ValueError("Stand-up assistance is disabled for this environment.")
        self.cfg.pair_standup_assistance_current_gravity_ratio = gravity_ratio
        self._apply_standup_assistance()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_static_physics_cache()
        self.extras.pop('final_critic_observation', None)
        self.extras.pop('final_critic_mask', None)
        if getattr(self.cfg, "pair_standup_support_cfg", {}).get("enabled", False):
            if not hasattr(self, "_support_reader"):
                if self._physics_handles_decimation:
                    raise RuntimeError("Support sampling requires exposed PhysX substeps.")
                self._support_reader = SupportContactReader(self)
            self._support_substeps = 0
            self._support_force_sum = torch.zeros(self.num_envs, 2, device=self.device)
            self._support_hip_sum = torch.zeros_like(self._support_force_sum)
            support_cfg = self.cfg.pair_standup_support_cfg
            if support_cfg.get("reward_mode") == "staged" and support_cfg['stages'].get('hands', {}).get('enabled', False):
                if not hasattr(self, '_hand_reader'):
                    self._hand_reader = HandContactReader(self, support_cfg['stages']['hands']['body_names'])
                    self._hand_contact_steps = torch.zeros(self.num_envs,2,dtype=torch.long,device=self.device)
                self._hand_force_sum = torch.zeros(self.num_envs,2,3,device=self.device)
        previous_previous = getattr(self, "_standup_previous_previous_action", None)
        if previous_previous is None or previous_previous.shape != actions.shape:
            previous_previous = torch.zeros_like(actions)
            self._standup_previous_previous_action = previous_previous
        previous_previous.copy_(
            self._sim_to_policy_order(self.action_processor.previous_applied_action)
        )
        settling = self._standup_is_settling()
        if settling.any():
            actions = actions.clone()
            actions[settling] = 0.0
            if hasattr(self,'_mixed_group'):
                selected=settling & (self._mixed_group>0)
                proc=self.action_processor
                held=(self._mixed_target-proc.offset-proc.offset_noise)/proc.scale
                actions[selected]=self._sim_to_policy_order(held)[selected]
        self.action_processor.pre_process_action(self._policy_to_sim_order(actions))
        self._apply_standup_assistance()

    def _apply_action(self) -> None:
        if getattr(self.cfg, "pair_standup_support_cfg", {}).get("enabled", False):
            if self._support_substeps > 0:
                self._sample_support_contact()
            self._support_substeps += 1
        super()._apply_action()

    def _sample_support_contact(self) -> None:
        feet, hips = self._support_reader.read()
        self._support_force_sum += feet
        self._support_hip_sum += hips
        if hasattr(self, '_hand_reader'):
            self._hand_force_last = self._hand_reader.read()
            self._hand_force_sum += self._hand_force_last

    def _update_support_state(self) -> None:
        from isaaclab.utils.math import quat_apply
        self._sample_support_contact()  # Final substep; earlier ones sampled before the next action.
        if self._support_substeps != self.cfg.decimation:
            raise RuntimeError("Missing support contact substep samples.")
        data = self.robot.data
        cfg = self.cfg.pair_standup_support_cfg
        mass = _to_torch(data.body_mass)
        com = (_to_torch(data.body_com_pos_w) * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
        feet_ids = [self.robot.body_names.index(f"{s}_ankle_roll_link") for s in ("left", "right")]
        feet = _to_torch(data.body_link_pos_w)[:, feet_ids]
        quat = _to_torch(data.body_link_quat_w)[:, feet_ids]
        # Rotate sphere centers; their lowest points are always 5 mm below in world Z.
        corners = torch.tensor([[-.05, .025, -.030], [-.05, -.025, -.030],
                                [.12, .03, -.030], [.12, -.03, -.030]], device=self.device)
        points = quat_apply(quat[:, :, None].expand(-1, -1, 4, -1).reshape(-1, 4),
                            corners[None, None].expand(self.num_envs, 2, -1, -1).reshape(-1, 3))
        points = sphere_support_points(points.reshape(self.num_envs, 2, 4, 3) + feet[:, :, None])
        points[..., 2] -= _to_torch(self.scene.env_origins)[:, None, None, 2]
        if 'stance' in cfg.get('stages', {}):
            from ref2act.common.math import quat_apply_inverse, yaw_quat
            heading = yaw_quat(_to_torch(data.body_link_quat_w)[:, self.anchor_body_index])
            anchor = _to_torch(data.body_link_pos_w)[:,self.anchor_body_index].clone()
            anchor[:,2] -= _to_torch(self.scene.env_origins)[:,2]
            self._stance_sole_heading_xy = quat_apply_inverse(
                heading[:,None,None].expand(-1,2,4,-1).reshape(-1,4),
                (points-anchor[:,None,None]).reshape(-1,3)).reshape(self.num_envs,2,4,3)[...,:2]
        transfer_cfg = preparation_config(cfg) if cfg.get("reward_mode") == "staged" else (cfg.get("transfer") if cfg.get("reward_mode") == "transfer" else None)
        if transfer_cfg is not None:
            normals = quat_apply(quat.reshape(-1, 4), torch.tensor(
                [0., 0., 1.], device=self.device).expand(self.num_envs * 2, -1)).reshape(self.num_envs, 2, 3)
            knees_ids = [self.robot.body_names.index(f"{s}_knee_link") for s in ("left", "right")]
            shanks = _to_torch(data.body_link_pos_w)[:, knees_ids] - feet
            hip_relative_xy = None
            if transfer_cfg.get("placement_reference", "com") == "hips":
                from ref2act.common.math import quat_apply_inverse, yaw_quat
                hip_ids = [self.robot.body_names.index(f"{side}_hip_pitch_link") for side in ("left", "right")]
                hips_position = _to_torch(data.body_link_pos_w)[:, hip_ids]
                heading = yaw_quat(_to_torch(data.body_link_quat_w)[:, self.anchor_body_index])
                delta = points.mean(-2) - hips_position
                hip_relative_xy = quat_apply_inverse(heading[:, None].expand(-1, 2, -1).reshape(-1, 4),
                    delta.reshape(-1, 3)).reshape(self.num_envs, 2, 3)[..., :2]
        if not hasattr(self, "_support_contact_steps"):
            self._support_contact_steps = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
            self._support_previous_potential = torch.zeros(self.num_envs, device=self.device)
        fz = self._support_force_sum / self._support_substeps
        hips = self._support_hip_sum / self._support_substeps
        finite = standup_finite_state(fz, hips, points, com, mass)
        if hasattr(self, '_hand_reader'):
            hand_ids = self._hand_reader.body_ids
            hand_pos = _to_torch(data.body_link_pos_w)[:, hand_ids]
            height = hand_pos[...,2] - _to_torch(self.scene.env_origins)[:,None,2]
            velocity = _to_torch(data.body_link_lin_vel_w)[:,hand_ids]
            angular = _to_torch(data.body_ang_vel_w)[:,hand_ids]
            # Velocity at the ground projection of each rigid-body origin.
            offset = torch.zeros_like(hand_pos); offset[...,2] = -height
            slip = (velocity + torch.cross(angular,offset,dim=-1))[...,:2].norm(dim=-1)
            force = self._hand_force_sum/self._support_substeps
            finite &= standup_finite_state(force, self._hand_force_last, height, slip)
            force = torch.nan_to_num(force)
            # Immediate release: substep averaging cannot retain airborne credit.
            force[...,2] = torch.minimum(force[...,2].clamp_min(0), torch.nan_to_num(self._hand_force_last[...,2]).clamp_min(0))
            active = finite & ~self._standup_is_settling()
            force = torch.where(active[:,None,None],force,0.)
            self._hand_state = hand_quality(force,mass.sum(1)*9.81,torch.nan_to_num(slip),
                torch.nan_to_num(height),self._hand_contact_steps,step_dt=self.step_dt,cfg=cfg['stages']['hands'])
            self._hand_state['ground_force_z'] = torch.nan_to_num(self._hand_force_sum[...,2]/self._support_substeps)
            self._hand_contact_steps.copy_(self._hand_state['steps'])
            log = self.extras.setdefault('log',{})
            for key in ('ground_force_z','load','contact','persistent','support','slip','height','release'):
                for i,side in enumerate(('left','right')):
                    log[f'Hands/{side}_{key}'] = torch.where(active,self._hand_state[key][:,i].float(),0.).mean().detach()
        if transfer_cfg is not None:
            finite &= standup_finite_state(normals, shanks)
            if hip_relative_xy is not None:
                finite &= standup_finite_state(hip_relative_xy)
        self._standup_finite_this_step &= finite
        safe_points = torch.where(finite[:, None, None, None], points, torch.zeros_like(points))
        weight = torch.nan_to_num(mass.sum(1) * 9.81, nan=1., posinf=1., neginf=1.).clamp_min(1.e-3)
        safe_hips = torch.where(finite[:, None], hips, 0.0)
        safe_fz = torch.where(finite[:, None], fz, 0.0)
        self._support_state = support_quality(
            safe_fz, weight,
            safe_points, torch.where(finite[:, None], com[:, :2], 0.0), self._support_contact_steps,
            contact_force_ratio=cfg["contact_force_ratio"],
            persistence_steps=standup_hold_steps(cfg["persistence_s"], self.step_dt),
            balance_sigma=cfg["balance_sigma"], sole_height_tolerance=cfg["sole_height_tolerance"])
        self._support_contact_steps.copy_(self._support_state["contact_steps"])
        if cfg.get('stages',{}).get('rise_hold_v2',False):
            from .rise_hold import support_margin, support_frame_lean
            valid = (safe_points[...,2].abs()<cfg['sole_height_tolerance']) & (
                self._support_state['contact'] & (self._support_contact_steps >=
                standup_hold_steps(cfg['persistence_s'],self.step_dt)))[...,None]
            self._rise_com_margin = support_margin(torch.where(finite[:,None],com[:,:2],0.),
                safe_points[...,:2].flatten(1,2),valid.flatten(1))
            torso_q = _to_torch(data.body_link_quat_w)[:,self.robot.body_names.index('torso_link')]
            up = quat_apply(torso_q,torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1))
            self._rise_torso_lean = support_frame_lean(up,safe_points)
            self._rise_torso_upright = up[:,2]
        if transfer_cfg is not None:
            if not hasattr(self, "_transfer_filtered_load"):
                self._transfer_filtered_load = torch.zeros(self.num_envs, device=self.device)
                self._transfer_peak_load = torch.zeros_like(self._transfer_filtered_load)
                self._transfer_contact_load = torch.zeros_like(self._transfer_filtered_load)
            self._transfer_quality = foot_preparation_quality(
                safe_points, torch.where(finite[:, None, None], normals, 0.0),
                torch.where(finite[:, None, None], shanks, 0.0),
                flat_mode=transfer_cfg.get("flat_mode", "exponential"),
                **{k: transfer_cfg[k] for k in ("clearance_sigma", "clearance_tolerance",
                                               "flat_sigma", "shank_min_cos", "shank_sigma")})
            if transfer_cfg.get("foot_placement_scale", 0) > 0:
                self._transfer_quality.update(foot_placement_quality(
                    safe_points, torch.where(finite[:, None], com[:, :2], 0.),
                    **{k: transfer_cfg[f"placement_{k}"] for k in (
                        "xy_tolerance", "xy_sigma", "height_tolerance", "height_sigma")}))
                if hip_relative_xy is not None:
                    self._transfer_quality.update(hip_placement_quality(
                        torch.where(finite[:, None, None], hip_relative_xy, 0.), safe_points, cfg=transfer_cfg))
            if transfer_cfg.get("sole_plant_scale", 0) > 0:
                self._transfer_quality.update(sole_plant_quality(
                    safe_points, torch.where(finite[:, None, None], normals, 0.),
                    tolerance=transfer_cfg["sole_plant_tolerance"], sigma=transfer_cfg["sole_plant_sigma"]))
            # Require a usable sole as well as ground force. Inverted foot
            # contacts and high airborne forces cannot advance the stage.
            usable_foot_load = self._support_state["foot_load"] * self._transfer_quality["support_orientation_per_foot"]
            self._transfer_quality["usable_foot_load_per_foot"] = usable_foot_load
            if transfer_cfg.get("bilateral_squat", False):
                self._transfer_quality["persistent_contact_per_foot"] = (
                    self._support_state["contact"] & (self._support_contact_steps >=
                        standup_hold_steps(cfg["persistence_s"], self.step_dt)))
            self._transfer_usable_foot_load = usable_foot_load
            self._transfer_contact_load.copy_(filtered_support_load(
                self._support_state["foot_load"].sum(-1), self._transfer_contact_load,
                step_dt=self.step_dt, time_constant=transfer_cfg["load_filter_s"]))
            current_load = usable_foot_load.sum(-1)
            self._transfer_filtered_load.copy_(filtered_support_load(
                current_load, self._transfer_filtered_load, step_dt=self.step_dt,
                time_constant=transfer_cfg["load_filter_s"]))
            active = finite & ~self._standup_is_settling()
            self._transfer_filtered_load.masked_fill_(~active, 0.)
            self._transfer_contact_load.masked_fill_(~active, 0.)
            rise_transfer = cfg.get('stages', {}).get('rise_transfer')
            if rise_transfer is not None:
                actual = getattr(self, '_standup_assistance_actual_ratio', torch.zeros_like(current_load))
                remaining = (1-actual.clamp_min(0)).clamp_min(rise_transfer['minimum_ground_weight'])
                if not hasattr(self, '_transfer_ground_fraction_filter'):
                    self._transfer_ground_fraction_filter = torch.zeros_like(current_load)
                self._transfer_ground_fraction_filter.copy_(filtered_support_load(
                    current_load/remaining, self._transfer_ground_fraction_filter,
                    step_dt=self.step_dt, time_constant=transfer_cfg['load_filter_s']))
                self._transfer_ground_fraction_filter.masked_fill_(~active, 0.)
            log = self.extras.setdefault("log", {})
            if transfer_cfg.get("bilateral_load_scale", 0) > 0:
                self._transfer_quality["bilateral_load"] = bilateral_load_quality(
                    usable_foot_load, target=transfer_cfg["bilateral_load_target"])
            if transfer_cfg.get("com_scale", 0) > 0 or transfer_cfg.get("com_preparation_scale", 0) > 0:
                persistent = self._support_state["contact"] & (self._support_contact_steps >=
                    standup_hold_steps(cfg["persistence_s"], self.step_dt))
                distance = self._support_state["distance"]
                com_quality, com_gate, alignment = com_support_quality(
                    distance, usable_foot_load, persistent, self._transfer_filtered_load,
                    sigma=cfg["balance_sigma"], cfg=transfer_cfg)
                self._transfer_quality["com_alignment"] = alignment
                if transfer_cfg.get("com_preparation_scale", 0) > 0:
                    self._transfer_quality["com_preparation"] = com_preparation_quality(
                        distance, persistent, self._transfer_quality["support_orientation_per_foot"],
                        sigma=cfg["balance_sigma"])
                valid_region = torch.isfinite(distance) & finite
                log["Transfer/com_distance_m"] = (torch.where(valid_region, distance, 0.).sum()
                    / valid_region.sum().clamp_min(1)).detach()
                log["Transfer/com_region_valid_fraction"] = valid_region.float().mean().detach()
                log["Transfer/com_region_quality"] = com_quality.mean().detach()
                log["Transfer/com_support_gate"] = com_gate.mean().detach()
            for name, value in self._transfer_quality.items():
                log[f"Transfer/{name}"] = value.float().mean().detach()
            log["Transfer/contact_load"] = self._transfer_contact_load.mean().detach()
            for i, side in enumerate(("left", "right")):
                if transfer_cfg.get("foot_placement_scale", 0) > 0:
                    for name in ("foot_placement", "foot_com_distance", "foot_clearance"):
                        log[f"Transfer/{side}_{name}"] = self._transfer_quality[f"{name}_per_foot"][:, i].mean().detach()
                if transfer_cfg.get("placement_reference", "com") == "hips":
                    for name in ("hip_geometry", "foot_hip_distance", "foot_hip_forward", "foot_hip_lateral"):
                        log[f"Transfer/{side}_{name}"] = self._transfer_quality[f"{name}_per_foot"][:, i].mean().detach()
                if transfer_cfg.get("sole_plant_scale", 0) > 0:
                    for name in ("sole_plant", "sole_max_clearance"):
                        log[f"Transfer/{side}_{name}"] = self._transfer_quality[f"{name}_per_foot"][:, i].mean().detach()
                log[f"Transfer/{side}_contact_load"] = self._support_state["foot_load"][:, i].mean().detach()
                log[f"Transfer/{side}_support_orientation"] = self._transfer_quality["support_orientation_per_foot"][:, i].mean().detach()
                log[f"Transfer/{side}_usable_load"] = usable_foot_load[:, i].mean().detach()
                log[f"Transfer/{side}_flat_quality"] = self._transfer_quality["feet_flat_per_foot"][:, i].mean().detach()
            log["Transfer/filtered_load"] = self._transfer_filtered_load.mean().detach()
        self._support_hip_cost = (safe_hips.sum(-1) / weight - 0.1).clamp(0, 1).square()
        log = self.extras.setdefault("log", {})
        for name in ("load", "balance", "potential"):
            log[f"Support/{name}"] = self._support_state[name].mean().detach()
        log["Support/foot_load_mg"] = (safe_fz.sum(-1) / weight).mean().detach()
        log["Support/hip_collision_cost"] = self._support_hip_cost.mean().detach()
        log["Support/hip_force_N"] = safe_hips.sum(-1).mean().detach()
        log["Support/hip_roll_pelvis_force_N"] = log["Support/hip_force_N"]
        log["Support/com_horizontal_speed"] = torch.nan_to_num(
            (_to_torch(data.body_com_lin_vel_w) * mass[..., None]).sum(1)[:, :2]
            / mass.sum(1, keepdim=True)).norm(dim=-1).mean().detach()

    def _apply_standup_assistance(
        self, env_ids: torch.Tensor | None = None
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if int(env_ids.numel()) == 0:
            return
        body_id = getattr(self, "_standup_assistance_body_index", None)
        if body_id is None:
            name = getattr(self.cfg, "pair_standup_assistance_body_name", "pelvis")
            matches = [i for i, body in enumerate(self.robot.body_names) if body == name]
            if len(matches) != 1:
                raise ValueError(f"Assistance requires exactly one rigid body named {name!r}; found {len(matches)}.")
            body_id = matches[0]
            self._standup_assistance_body_index = body_id
        joint_position = _to_torch(self.robot.data.joint_pos)
        masses = getattr(self, "_standup_total_mass", None)
        if masses is None or int(masses.shape[0]) != self.num_envs:
            masses = _to_torch(self.robot.data.body_mass).sum(dim=-1)
            self._standup_total_mass = masses
        settings = getattr(self.cfg, "pair_standup_assistance_settings", assistance_settings({}))
        draws = getattr(self, "_standup_assistance_draw", None)
        if draws is None or int(draws.shape[0]) != self.num_envs:
            draws = draw_assistance(self.num_envs, self.device, settings)
            self._standup_assistance_draw = draws
        active = ~self._standup_is_settling()[env_ids]
        if hasattr(self,'_mixed_group'):
            active &= self._mixed_group[env_ids]==0
        ratio = float(self.cfg.pair_standup_assistance_current_gravity_ratio)
        root_height, shoulder_height, upright, _, _, _, _ = self._recovery_state()
        posture = (orientation_multiplier(upright[env_ids], settings) if upright is not None
                   else torch.ones_like(draws[env_ids]))
        if not hasattr(self, "_standup_assistance_orientation_multiplier"):
            self._standup_assistance_orientation_multiplier = torch.zeros(self.num_envs, device=self.device)
        self._standup_assistance_orientation_multiplier[env_ids] = posture
        body_score = self._body_height_score(root_height, shoulder_height)
        height_multiplier = assistance_height_multiplier(
            body_score[env_ids],
            start=float(self.cfg.pair_standup_assistance_height_gate_start),
            full=float(self.cfg.pair_standup_assistance_height_gate_full),
        )
        latched_off = getattr(self, "_standup_assistance_latched_off", None)
        if latched_off is None or int(latched_off.shape[0]) != self.num_envs:
            latched_off = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._standup_assistance_latched_off = latched_off
        # Reset writes can precede the next simulator data refresh.  Never
        # latch from potentially stale standing geometry during settling.
        latched_off[env_ids] |= active & (
            body_score[env_ids]
            >= float(self.cfg.pair_standup_assistance_height_gate_full)
        )
        height_multiplier *= (~latched_off[env_ids]).to(height_multiplier.dtype)
        multiplier_log = getattr(self, "_standup_assistance_multiplier", None)
        if multiplier_log is None or int(multiplier_log.shape[0]) != self.num_envs:
            multiplier_log = torch.zeros(self.num_envs, device=self.device)
            self._standup_assistance_multiplier = multiplier_log
        staged_cfg = getattr(self.cfg, "pair_standup_support_cfg", {})
        if staged_cfg.get("reward_mode") == "staged" and staged_cfg["stages"]["gate_assistance"]:
            gate = getattr(self, "_stage_assistance_gate", torch.zeros(self.num_envs, device=self.device))
            height_multiplier *= gate[env_ids]
        multiplier_log[env_ids] = height_multiplier
        forces = torch.zeros(
            (int(env_ids.numel()), 1, 3),
            dtype=joint_position.dtype,
            device=self.device,
        )
        forces[:, 0, 2] = (
            draws[env_ids]
            * ratio
            * masses[env_ids]
            * 9.81
            * active.to(dtype=forces.dtype)
            * height_multiplier
            * posture
        )
        if not hasattr(self, "_standup_assistance_actual_ratio"):
            self._standup_assistance_actual_ratio = torch.zeros(self.num_envs, device=self.device)
        self._standup_assistance_actual_ratio[env_ids] = forces[:, 0, 2] / (masses[env_ids] * 9.81)

        body_ids = torch.tensor(
            [body_id], dtype=torch.long, device=self.device
        )
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torch.zeros_like(forces),
            body_ids=body_ids,
            env_ids=env_ids,
            is_global=True,
        )

    def _body_height_score(
        self, root_height: torch.Tensor, shoulder_height: torch.Tensor
    ) -> torch.Tensor:
        root = torch.clamp(
            root_height / float(self._pair_recovery_target_root_height), 0.0, 1.0
        )
        shoulder = torch.clamp(
            shoulder_height / float(self._pair_recovery_target_shoulder_height),
            0.0,
            1.0,
        )
        return torch.minimum(root, shoulder)

    def _staged_reward_result(self, root_height, shoulder_height, upright,
                              linear_velocity, angular_velocity, joint_velocity):
        q = self._transfer_quality
        lean = None
        if 'rise_transfer' in self.cfg.pair_standup_support_cfg['stages']:
            torso_id = self.robot.body_names.index('torso_link')
            quat = _to_torch(self.robot.data.body_link_quat_w)[:, torso_id]
            lean = (self._rise_torso_lean if self.cfg.pair_standup_support_cfg['stages'].get('rise_hold_v2',False)
                    else signed_torso_lean(quat))
        inputs = StageInputs(
            root_height=torch.nan_to_num(root_height), shoulder_height=torch.nan_to_num(shoulder_height),
            root_target=self._pair_recovery_target_root_height,
            shoulder_target=self._pair_recovery_target_shoulder_height,
            upright=torch.nan_to_num(upright), foot_distance=q["foot_hip_distance_per_foot"],
            sole_plant=q["sole_plant_per_foot"], usable_load=q["usable_foot_load_per_foot"],
            persistent_contact=q["persistent_contact_per_foot"], filtered_load=self._transfer_filtered_load,
            com_distance=self._support_state["distance"],
            linear_speed=torch.nan_to_num(linear_velocity).norm(dim=-1),
            angular_speed=torch.nan_to_num(angular_velocity).norm(dim=-1),
            joint_speed=torch.nan_to_num(joint_velocity).square().mean(-1).sqrt(),
            torso_forward_lean=lean,
            assistance_ratio=getattr(self, '_standup_assistance_actual_ratio', torch.zeros_like(root_height)),
            filtered_ground_fraction=getattr(self, '_transfer_ground_fraction_filter', None),
            foot_low=q.get('feet_low_per_foot'),
            foot_clearance=q.get('foot_clearance_per_foot'),
            foot_flat=q.get('feet_signed_flat_per_foot'),
            com_margin=getattr(self,'_rise_com_margin',None),
            torso_upright=getattr(self,'_rise_torso_upright',None),
            sole_heading_xy=getattr(self, '_stance_sole_heading_xy', None))
        return stage_rewards(inputs, self.cfg.pair_standup_support_cfg["stages"], step_dt=self.step_dt,
            hand_state=getattr(self, "_hand_state", None),
            reward_form=("positive" if self.cfg.pair_standup_reward_cfg.get("mode") == "simple_v13" else "deficit"))

    def _get_rewards(self):
        if self.cfg.pair_standup_reward_cfg.get('mode') == 'simple_v13':
            from .simple_rewards import get_rewards
        else:
            from .legacy_rewards import get_rewards
        return get_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_state = _to_torch(self.robot.data.root_link_state_w)
        joint_position = _to_torch(self.robot.data.joint_pos)
        joint_velocity = _to_torch(self.robot.data.joint_vel)
        root_height, shoulder_height, upright, _, _, _, _ = self._recovery_state()
        base_linear_velocity, base_angular_velocity, _ = self._anchor_state_b()
        finite = standup_finite_state(
            root_state, joint_position, joint_velocity, root_height, shoulder_height,
            upright, base_linear_velocity, base_angular_velocity,
            _to_torch(self.robot.data.applied_torque),
            self.action_processor.applied_action,
            self.action_processor.previous_applied_action,
        )
        self._standup_finite_this_step = finite
        support_cfg = getattr(self.cfg, "pair_standup_support_cfg", {})
        if support_cfg.get("enabled", False):
            self._update_support_state()
            finite = self._standup_finite_this_step
        excessive_joint_velocity = joint_velocity.abs().amax(dim=-1) > float(
            self.cfg.pair_standup_max_joint_velocity
        )
        excessive_base_angular_velocity = root_state[:, 10:13].abs().amax(
            dim=-1
        ) > float(self.cfg.pair_standup_max_base_angular_velocity)
        terminated = (
            ~finite | excessive_joint_velocity | excessive_base_angular_velocity
        )
        fall_cfg = getattr(self.cfg, "pair_standup_fall_cfg", {})
        if fall_cfg.get("enabled", False):
            if not hasattr(self, "_standup_fall_steps"):
                self._standup_fall_steps = torch.zeros_like(self.episode_length_buf)
            fallen, fall_steps = crouch_fall_state(
                root_height, upright, self._standup_fall_steps,
                minimum_root_height=fall_cfg["minimum_root_height"],
                minimum_upright_projection=fall_cfg["minimum_upright_projection"],
                immediate_root_height=fall_cfg["immediate_root_height"],
                required_steps=standup_hold_steps(fall_cfg["persistence_s"], self.step_dt))
            self._standup_fall_steps.copy_(fall_steps)
            terminated |= fallen
            self.extras.setdefault("log", {})["StandUp/fall_termination_events"] = fallen.float().sum().detach()
        mixed_fall=(getattr(self.cfg,'pair_standup_mixed_reset',None) or {}).get('stand_fall')
        if mixed_fall is not None and hasattr(self,'_mixed_group'):
            from .mixed_reset import stand_reset_fall
            if not hasattr(self,'_stand_reset_fall_steps'):
                self._stand_reset_fall_steps=torch.zeros_like(self.episode_length_buf)
            fallen,self._stand_reset_fall_steps=stand_reset_fall(
                self._mixed_group,self._standup_is_settling(),root_height,upright,
                self._stand_reset_fall_steps,self._pair_recovery_target_root_height,mixed_fall,self.step_dt)
            terminated |= fallen
            self.extras.setdefault('log',{})['Reset/hold/fall_events']=fallen.float().sum().detach()
        sampler = getattr(self, '_episode_sampler', None)
        time_out = (self.episode_length_buf >= sampler.deadline if sampler is not None else
                    self.episode_length_buf >= (self.max_episode_length - 1))
        deadline_out = time_out.clone()
        self._support_boundary = terminated | time_out

        body_score = self._body_height_score(root_height, shoulder_height)
        standing = standup_stable_state(
            body_height_score=body_score,
            upright_projection=upright,
            base_linear_velocity=root_state[:, 7:10],
            base_angular_velocity=root_state[:, 10:13],
            joint_velocity=joint_velocity,
            minimum_body_height_score=1.0,
            minimum_upright_projection=float(self.cfg.pair_standup_success_upright),
            maximum_base_linear_velocity=float(
                self.cfg.pair_standup_success_max_base_linear_velocity
            ),
            maximum_base_angular_velocity=float(
                self.cfg.pair_standup_success_max_base_angular_velocity
            ),
            maximum_joint_velocity=float(
                self.cfg.pair_standup_success_max_joint_velocity
            ),
        )
        standing &= ~self._standup_is_settling() & ~terminated
        if support_cfg.get("enabled", False):
            supported = self._support_state["load"] >= support_cfg["success_min_load"]
            if support_cfg.get("reward_mode") in ("transfer", "staged"):
                supported &= self._transfer_filtered_load >= support_cfg["success_min_load"]
                if support_cfg.get("success_min_per_foot_load", 0) > 0:
                    persistent = self._support_state["contact"] & (self._support_contact_steps >=
                        standup_hold_steps(support_cfg["persistence_s"], self.step_dt))
                    supported &= bilateral_support_ready(self._transfer_usable_foot_load, persistent,
                        minimum_per_foot=support_cfg["success_min_per_foot_load"])
            self.extras.setdefault("log", {})["Support/supported_stable_fraction"] = (standing & supported).float().mean().detach()
            if support_cfg.get("require_support_for_success", True):
                standing &= supported
        if support_cfg.get('stages',{}).get('rise_hold_v2',False):
            from .standup_stance import stance_geometry
            standing &= (self._rise_com_margin>=0.) & (self._rise_torso_lean.abs()<=0.174533)
            standing &= self._rise_torso_upright>=self.cfg.pair_standup_success_upright
            standing &= self._hand_state['load'].sum(-1)<support_cfg['stages']['hands']['contact_load']
            standing &= self._transfer_usable_foot_load.amin(-1)>=.2
            if 'stance' in support_cfg['stages']:
                standing &= stance_geometry(self._stance_sole_heading_xy,support_cfg['stages']['stance'])['ready']>=.99
        hold = getattr(self, "_standup_success_hold", None)
        if hold is None or hold.shape != self.episode_length_buf.shape:
            hold = torch.zeros_like(self.episode_length_buf)
            self._standup_success_hold = hold
        hold.copy_(torch.where(standing, hold + 1, torch.zeros_like(hold)))
        success_now = hold >= standup_hold_steps(
            float(self.cfg.pair_standup_success_hold_s), float(self.step_dt)
        )
        latched = getattr(self, "_standup_success_latched", None)
        if latched is None or latched.shape != standing.shape:
            latched = torch.zeros_like(standing)
            self._standup_success_latched = latched
        success_events = success_now & ~latched
        latched |= success_now

        if sampler is not None:
            q=self._transfer_quality
            stage_cfg=support_cfg['stages']
            distance=((stage_cfg['distance_zero']-q['foot_hip_distance_per_foot'])/
                (stage_cfg['distance_zero']-stage_cfg['distance_full'])).clamp(0,1)
            features=torch.cat(((upright.clamp(-1,1)+1)[:,None]/2,
                ((shoulder_height-root_height)/self._pair_recovery_target_shoulder_height).clamp(0,1)[:,None],
                distance,q['sole_plant_per_foot'],q['usable_foot_load_per_foot'],
                self._transfer_filtered_load[:,None],body_score[:,None]),-1)
            if hasattr(self,'_hand_state'):
                features=torch.cat((features,self._hand_state['approach'],self._hand_state['support']),-1)
            if 'stance' in stage_cfg:
                from .standup_stance import stance_geometry
                from .standup_stages import hand_unload_quality
                stance=stance_geometry(self._stance_sole_heading_xy,stage_cfg['stance'])
                transfer=stage_cfg['rise_transfer']
                remaining=(1-self._standup_assistance_actual_ratio).clamp_min(transfer['minimum_ground_weight'])
                unload=hand_unload_quality(self._hand_state['load'].sum(-1)/remaining,transfer)
                # Count useful foot separation and supported hand unloading as
                # progress, rather than resetting while these repairs improve.
                takeover=unload*(q['usable_foot_load_per_foot'].amin(-1)/.08).clamp(0,1)
                features=torch.cat((features,stance['quality'][:,None],takeover[:,None]),-1)
            stalled=sampler.stagnated(torch.nan_to_num(features),
                (self.episode_length_buf-self._standup_settling_steps()).clamp_min(0),
                finite & ~terminated & ~self._standup_is_settling(), standing)
            time_out |= stalled
            log=self.extras.setdefault('log',{})
            log['Sampling/deadline_events']=(deadline_out & ~terminated).float().sum().detach()
            log['Sampling/stagnation_events']=(stalled & ~deadline_out & ~terminated).float().sum().detach()
            log['Sampling/initial_cut_events']=(time_out & sampler.initial & ~terminated).float().sum().detach()
            log['Sampling/reset_count']=(time_out|terminated).float().sum().detach()
            log['Sampling/idle_seconds']=sampler.idle.float().mean().detach()*self.step_dt
        self._support_boundary = terminated | time_out
        outcomes = time_out | terminated
        self._standup_holding_this_step = standing.clone()
        self._standup_success_this_step = success_events.clone()
        # A fixed-horizon timeout is the normal training boundary. Penalizing
        # every still-learning environment at that synchronized boundary adds
        # no ranking signal and creates a large periodic value target.
        self._standup_failure_this_step = terminated.clone()
        log = self.extras.setdefault("log", {})
        log["StandUp/success_fraction"] = latched.float().mean().detach()
        log["StandUp/stable_fraction"] = standing.float().mean().detach()
        log["StandUp/success_events"] = success_events.float().sum().detach()
        log["StandUp/outcomes"] = outcomes.float().sum().detach()
        log["StandUp/successful_outcomes"] = (
            outcomes & latched
        ).float().sum().detach()
        log["StandUp/stable_outcomes"] = (
            outcomes & success_now & ~terminated
        ).float().sum().detach()
        log["StandUp/nonfinite_fraction"] = (~finite).float().mean().detach()
        # Exclude episodes spanning a global force-level change. Probe episodes
        # have zero force throughout, not just after the height fade-out.
        episode_ratio = getattr(self, "_standup_assistance_episode_ratio", None)
        if episode_ratio is not None:
            eligible = outcomes & torch.isclose(episode_ratio, torch.full_like(
                episode_ratio, float(self.cfg.pair_standup_assistance_current_gravity_ratio)), rtol=0, atol=1.e-8)
            if hasattr(self,'_mixed_group'):
                eligible &= self._mixed_group==0
            if sampler is not None:
                eligible &= ~sampler.initial
            probe = self._standup_assistance_draw == 0
            stable_end = success_now & ~terminated
            for name, mask in (
                ("curriculum_outcomes", eligible),
                ("curriculum_stable_outcomes", eligible & stable_end),
                ("probe_outcomes", eligible & probe),
                ("probe_stable_outcomes", eligible & probe & stable_end),
            ):
                log[f"StandUp/{name}"] = mask.float().sum().detach()
        if hasattr(self,'_mixed_group'):
            from .mixed_reset import GROUPS
            for g,name in enumerate(GROUPS):
                mask=self._mixed_group==g
                log[f'Reset/{name}/occupancy']=mask.float().mean().detach()
                log[f'Reset/{name}/outcomes']=(outcomes&mask).float().sum().detach()
                log[f'Reset/{name}/success_outcomes']=(outcomes&mask&latched).float().sum().detach()
                log[f'Reset/{name}/stable_fraction']=(standing&mask).float().sum().detach()/mask.sum().clamp_min(1)
            ground=self._mixed_group==0
            log['StandUp/success_fraction']=(latched&ground).float().sum().detach()/ground.sum().clamp_min(1)
            log['StandUp/stable_fraction']=(standing&ground).float().sum().detach()/ground.sum().clamp_min(1)
            for name,mask in [('outcomes',outcomes),('successful_outcomes',outcomes&latched),('stable_outcomes',outcomes&success_now&~terminated),('success_events',success_events)]:
                log['StandUp/'+name]=(mask&ground).float().sum().detach()
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        self._ensure_static_physics_cache()
        normalized = self._normalize_env_ids(env_ids)
        # Infer standing targets from Ref2Act's default upright state before the
        # first fixed supine write changes the simulator state distribution.
        self._ensure_recovery_geometry()
        self.robot.reset(normalized)
        DirectRLEnv._reset_idx(self, normalized)
        self.action_processor.reset_action_buffer(normalized)
        if self.cfg.action.noise_scale > 0.0:
            self.action_processor.set_random_offset_noise(normalized)
        self._observation_reset_mask[normalized] = True
        sampling_cfg=getattr(self.cfg,'pair_standup_episode_sampling',{})
        if sampling_cfg.get('enabled',False):
            if not hasattr(self,'_episode_sampler'):
                self._episode_sampler=EpisodeSampling(self.num_envs,self.device,self.step_dt,sampling_cfg,
                    feature_count=(14 if self.cfg.pair_standup_support_cfg['stages'].get('hands',{}).get('enabled',False) else 10)
                    +(2 if 'stance' in self.cfg.pair_standup_support_cfg['stages'] else 0))
            self._episode_sampler.reset(normalized, initial=self.common_step_counter==0)

        from isaaclab.utils.math import quat_from_euler_xyz

        count = int(normalized.numel())
        origins = _to_torch(self.scene.env_origins)[normalized]
        root_state = _to_torch(self.robot.data.default_root_state)[normalized].clone()
        root_offset = torch.tensor(
            self.cfg.pair_standup_initial_root_position,
            dtype=root_state.dtype,
            device=self.device,
        )
        root_state[:, :3] = origins + root_offset
        roll, pitch, yaw = self.cfg.pair_standup_initial_root_euler_xyz
        angles = torch.full((count,), float(roll), device=self.device)
        pitches = torch.full((count,), float(pitch), device=self.device)
        yaws = torch.full((count,), float(yaw), device=self.device)
        root_state[:, 3:7] = quat_from_euler_xyz(angles, pitches, yaws)
        root_state[:, 7:] = 0.0
        joint_position = _to_torch(self.robot.data.default_joint_pos)[normalized].clone()
        joint_velocity = torch.zeros_like(joint_position)
        state = getattr(self.cfg, "pair_standup_crouch_state", None)
        if state is not None:
            if set(state["joint_names"]) != set(self.robot.joint_names):
                raise ValueError("Crouch snapshot joints do not match the live robot.")
            order = [state["joint_names"].index(n) for n in self.robot.joint_names]
            root_state[:] = torch.tensor(state["root_state"], device=self.device)
            root_state[:, :3] += origins
            joint_position[:] = torch.tensor(state["joint_position"], device=self.device)[order]
            joint_velocity[:] = torch.tensor(state["joint_velocity"], device=self.device)[order]
            noise = self.cfg.pair_standup_initial_joint_noise
            joint_position += (2 * torch.rand_like(joint_position) - 1) * noise
        mixed=getattr(self.cfg,'pair_standup_mixed_reset',None)
        if mixed is not None:
            from .mixed_reset import sample_states
            sample_states(self,normalized,root_state,joint_position,joint_velocity,origins)
        self.robot.write_root_link_pose_to_sim_index(
            root_pose=root_state[:, :7], env_ids=normalized
        )
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=root_state[:, 7:], env_ids=normalized
        )
        self.robot.write_joint_position_to_sim_index(
            position=joint_position, env_ids=normalized
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=joint_velocity, env_ids=normalized
        )
        self.action_processor.target_joint_position[normalized] = joint_position
        self.action_processor.applied_action[normalized] = 0.0
        self.action_processor.previous_applied_action[normalized] = 0.0
        if state is not None:
            offset = self.action_processor.offset.expand(self.num_envs, -1)[normalized]
            scale = self.action_processor.scale.expand(self.num_envs, -1)[normalized]
            initial_action = (joint_position - offset
                              - self.action_processor.offset_noise[normalized]) / scale
            self.action_processor.applied_action[normalized] = initial_action
            self.action_processor.previous_applied_action[normalized] = initial_action
        if mixed is not None:
            selected=normalized[self._mixed_group[normalized]>0]
            proc=self.action_processor
            initial=(self._mixed_target-proc.offset-proc.offset_noise)/proc.scale
            proc.target_joint_position[selected]=self._mixed_target[selected]
            proc.applied_action[selected]=initial[selected]
            proc.previous_applied_action[selected]=initial[selected]
        if hasattr(self, '_hand_contact_steps'):
            self._hand_contact_steps[normalized] = 0
            if hasattr(self, '_hand_state'):
                for value in self._hand_state.values():
                    value[normalized] = 0
        if hasattr(self, "_stage_assistance_gate"):
            self._stage_assistance_gate[normalized] = 0.
        if hasattr(self, "_support_contact_steps"):
            self._support_contact_steps[normalized] = 0
            self._support_previous_potential[normalized] = 0.0
        if hasattr(self, "_transfer_filtered_load"):
            self._transfer_filtered_load[normalized] = 0.0
            if hasattr(self, '_transfer_ground_fraction_filter'):
                self._transfer_ground_fraction_filter[normalized] = 0.0
            if hasattr(self, "_transfer_contact_load"):
                self._transfer_contact_load[normalized] = 0.0
            self._transfer_peak_load[normalized] = 0.0

        for name, dtype in (
            ("_standup_fall_steps", torch.long),
            ("_standup_success_hold", torch.long),
            ("_standup_success_latched", torch.bool),
            ("_standup_success_this_step", torch.bool),
            ("_standup_holding_this_step", torch.bool),
            ("_standup_failure_this_step", torch.bool),
            ("_standup_assistance_latched_off", torch.bool),
        ):
            value = getattr(self, name, None)
            if value is None or int(value.shape[0]) != self.num_envs:
                value = torch.zeros(self.num_envs, dtype=dtype, device=self.device)
                setattr(self, name, value)
            value[normalized] = False
        previous_previous = getattr(self, "_standup_previous_previous_action", None)
        if previous_previous is not None:
            previous_previous[normalized] = 0.0
            if mixed is not None:
                previous_previous[normalized]=self._sim_to_policy_order(self.action_processor.applied_action)[normalized]
        draws = getattr(self, "_standup_assistance_draw", None)
        if draws is not None:
            draws[normalized] = draw_assistance(count, self.device, getattr(
                self.cfg, "pair_standup_assistance_settings", assistance_settings({})))
        if not hasattr(self, "_standup_assistance_episode_ratio"):
            self._standup_assistance_episode_ratio = torch.zeros(self.num_envs, device=self.device)
        self._standup_assistance_episode_ratio[normalized] = float(
            self.cfg.pair_standup_assistance_current_gravity_ratio)
        multiplier = getattr(self, "_standup_assistance_multiplier", None)
        if multiplier is not None:
            multiplier[normalized] = 0.0
        self._apply_standup_assistance(normalized)

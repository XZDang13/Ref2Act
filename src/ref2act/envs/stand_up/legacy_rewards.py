from __future__ import annotations


import torch

from ref2act.isaac_compat import to_torch as _to_torch
from .recovery_reward import smooth_height_gate
from .standup_support import prepared_support_load, squat_support_quality, potential_reward, supported_pelvis_lift_quality
from .standup_transfer_rewards import (
    independent_height_deficits, shank_preparation_gate, righting_quality, support_transfer_rewards,
)
from .standup_observation import critic_terms, pack_critic_terms
from .standup_reward import compute_standup_reward_terms


def get_rewards(self) -> torch.Tensor:
    (
        root_height,
        shoulder_height,
        upright,
        _,
        _,
        _,
        upright_score,
    ) = self._recovery_state()
    body_score = self._body_height_score(root_height, shoulder_height)
    base_linear_velocity, base_angular_velocity, _ = self._anchor_state_b()

    joint_position = _to_torch(self.robot.data.joint_pos)
    joint_velocity = _to_torch(self.robot.data.joint_vel)
    joint_torque = _to_torch(self.robot.data.applied_torque)
    joint_position_limits = _to_torch(self.robot.data.joint_pos_limits)
    joint_velocity_limits = _to_torch(self.robot.data.soft_joint_vel_limits)
    if joint_velocity_limits.ndim == 1:
        joint_velocity_limits = joint_velocity_limits.unsqueeze(0).expand_as(
            joint_velocity
        )
    action = self._sim_to_policy_order(self.action_processor.applied_action)
    previous_action = self._sim_to_policy_order(
        self.action_processor.previous_applied_action
    )
    previous_previous_action = getattr(
        self, "_standup_previous_previous_action", torch.zeros_like(action)
    )
    completed = getattr(self, "_standup_success_this_step", None)
    if completed is None or completed.shape != body_score.shape:
        completed = torch.zeros_like(body_score, dtype=torch.bool)
    failed = getattr(self, "_standup_failure_this_step", None)
    if failed is None or failed.shape != body_score.shape:
        failed = torch.zeros_like(body_score, dtype=torch.bool)
    holding = getattr(self, "_standup_holding_this_step", None)
    if holding is None or holding.shape != body_score.shape:
        holding = torch.zeros_like(body_score, dtype=torch.bool)
    reward_cfg = self.cfg.pair_standup_reward_cfg
    terms = compute_standup_reward_terms(
        body_height_score=body_score,
        upright_score=upright_score,
        recovering=torch.ones_like(body_score, dtype=torch.bool),
        holding=holding,
        completed=completed,
        failed=failed,
        joint_position=joint_position,
        joint_position_lower=joint_position_limits[:, :, 0],
        joint_position_upper=joint_position_limits[:, :, 1],
        joint_velocity=joint_velocity,
        joint_velocity_limit=joint_velocity_limits,
        joint_torque=joint_torque,
        base_linear_velocity=base_linear_velocity,
        base_angular_velocity=base_angular_velocity,
        action=action,
        previous_action=previous_action,
        previous_previous_action=previous_previous_action,
        step_dt=float(self.step_dt),
        valid_state=self._standup_finite_this_step,
        **reward_cfg,
    )
    support_cfg = getattr(self.cfg, "pair_standup_support_cfg", {})
    transfer_cfg = support_cfg.get("transfer") if support_cfg.get("enabled", False) and support_cfg.get("reward_mode") == "transfer" else None
    if transfer_cfg is not None:
        from dataclasses import replace
        load = self._transfer_filtered_load
        root_score = torch.nan_to_num(root_height / self._pair_recovery_target_root_height).clamp(0, 1)
        shoulder_score = torch.nan_to_num(shoulder_height / self._pair_recovery_target_shoulder_height).clamp(0, 1)
        squat = squat_support_quality(self._transfer_quality, transfer_cfg) if transfer_cfg.get("bilateral_squat", False) else None
        if squat is not None:
            active_squat = self._standup_finite_this_step & ~failed & ~self._standup_is_settling()
            self.extras.setdefault("log", {})["Transfer/squat_support_quality"] = torch.where(active_squat, squat, 0.).mean().detach()
        if transfer_cfg.get("pelvis_lift_scale", 0) > 0:
            reward_load, geometry_gates = prepared_support_load(self._transfer_quality, load, transfer_cfg)
            lift_quality = supported_pelvis_lift_quality(root_score, reward_load, torch.nan_to_num(upright), cfg=transfer_cfg,
                hip_quality=self._transfer_quality.get("hip_geometry"), squat_quality=squat)
            active_lift = self._standup_finite_this_step & ~failed & ~self._standup_is_settling()
            self.extras.setdefault("log", {})["Transfer/pelvis_lift_quality"] = torch.where(active_lift, lift_quality, 0.).mean().detach()
            if geometry_gates is not None:
                self.extras["log"]["Transfer/prepared_load"] = torch.where(active_lift, reward_load, 0.).mean().detach()
                for i, side in enumerate(("left", "right")):
                    self.extras["log"][f"Transfer/{side}_support_geometry_gate"] = torch.where(active_lift, geometry_gates[:, i], 0.).mean().detach()
        stability_load = smooth_height_gate(load, start=transfer_cfg["stability_load_start"],
                                            full=transfer_cfg["stability_load_full"])
        height_weight = transfer_cfg["unloaded_height_weight"] + (1 - transfer_cfg["unloaded_height_weight"]) * load
        height_reward = terms.recovery_height_deficit * height_weight
        if transfer_cfg.get("height_mode") == "independent":
            components = independent_height_deficits(
                torch.nan_to_num(root_height / self._pair_recovery_target_root_height) if transfer_cfg.get("upper_relative_before_support", False) else root_score,
                torch.nan_to_num(shoulder_height / self._pair_recovery_target_shoulder_height) if transfer_cfg.get("upper_relative_before_support", False) else shoulder_score,
                step_dt=self.step_dt, cfg=transfer_cfg,
                upright_projection=torch.nan_to_num(upright), squat_quality=squat,
                root_to_shoulder_ratio=self._pair_recovery_target_root_height / self._pair_recovery_target_shoulder_height)
            components = {k: torch.where(self._standup_finite_this_step, v, 0.) for k, v in components.items()}
            height_reward = sum(components.values())
            active_log = self._standup_finite_this_step & ~(self._standup_is_settling() & ~failed)
            log = self.extras.setdefault("log", {})
            for key, value in components.items():
                log[f"Transfer/reward_height_{key}"] = torch.where(active_log, value, 0.).mean().detach()
            log["Transfer/shank_preparation_gate"] = shank_preparation_gate(shoulder_score, transfer_cfg, root_score).mean().detach()
            if transfer_cfg.get("righting_scale", 0) > 0:
                righting = righting_quality(torch.nan_to_num(upright), transfer_cfg)
                log["Transfer/righting_quality"] = (torch.where(active_log, righting, 0.).sum() / active_log.sum().clamp_min(1)).detach()
                log["Transfer/negative_up_fraction"] = ((active_log & (upright < 0)).float().sum() / active_log.sum().clamp_min(1)).detach()
        terms = replace(terms,
            recovery_height_deficit=height_reward,
            recovery_upright_deficit=terms.recovery_upright_deficit * stability_load,
            standing_stability=terms.standing_stability * stability_load * (squat if squat is not None else 1.),
            standing_motion_penalty=terms.standing_motion_penalty * stability_load)
    staged = None
    if support_cfg.get("enabled", False) and support_cfg.get("reward_mode") == "staged":
        from dataclasses import replace
        staged = self._staged_reward_result(root_height, shoulder_height, upright,
            base_linear_velocity, base_angular_velocity, joint_velocity)
        # Stages own all dense task/style objectives. Keep terminal events
        # and actuator regularization exactly once; no legacy height bypass.
        zero = torch.zeros_like(root_height)
        terms = replace(terms, recovery_height_deficit=zero, recovery_upright_deficit=zero,
            standing_stability=zero, standing_motion_penalty=zero)
        active = self._standup_finite_this_step & ~failed & ~self._standup_is_settling()
        self._stage_assistance_gate = torch.where(active, staged.assistance_gate, 0.).detach()
    reward = terms.total
    if support_cfg.get("enabled", False):
        if transfer_cfg is not None:
            transfer_terms, peak, preparation_weight = support_transfer_rewards(
                self._transfer_quality, self._transfer_filtered_load, self._transfer_peak_load,
                torch.nan_to_num(body_score), step_dt=self.step_dt, cfg=transfer_cfg,
                shoulder_score=shoulder_score, upright_projection=torch.nan_to_num(upright),
                root_score=root_score, contact_load=self._transfer_contact_load)
            active = self._standup_finite_this_step & ~failed & ~self._standup_is_settling()
            self._transfer_peak_load.copy_(torch.where(active, peak, self._transfer_peak_load))
            transfer_terms = {k: torch.where(active, v, 0.) for k, v in transfer_terms.items()}
            shaping = sum(transfer_terms.values())
            log = self.extras.setdefault("log", {})
            for name, value in transfer_terms.items():
                log[f"Transfer/reward_{name}"] = value.mean().detach()
            log["Transfer/peak_load"] = self._transfer_peak_load.mean().detach()
            log["Transfer/preparation_weight"] = preparation_weight.mean().detach()
            log["Transfer/stability_load_gate"] = stability_load.mean().detach()
        elif staged is not None:
            active = self._standup_finite_this_step & ~failed & ~self._standup_is_settling()
            staged_terms = {k: torch.where(active, v, 0.) for k,v in staged.rewards.items()}
            shaping = sum(staged_terms.values())
            log = self.extras.setdefault("log", {})
            for key, value in staged_terms.items():
                log[f"Stages/reward_{key}"] = value.mean().detach()
            for key, value in staged.diagnostics.items():
                log[f"Stages/{key}"] = torch.where(active, value, 0.).mean().detach()
            log["Stages/assistance_gate"] = self._stage_assistance_gate.mean().detach()
        else:
            shaping = potential_reward(
                self._support_previous_potential, self._support_state["potential"],
                self._support_boundary, gamma=self.cfg.pair_standup_support_gamma,
                scale=support_cfg["shaping_scale"])
        collision = -support_cfg["hip_collision_scale"] * self.step_dt * self._support_hip_cost
        reward_mask = self._standup_finite_this_step & ~(self._standup_is_settling() & ~failed)
        shaping = torch.where(reward_mask, shaping, 0.)
        collision = torch.where(reward_mask, collision, 0.)
        reward += shaping + collision
        self._support_previous_potential.copy_(self._support_state["potential"])
        self.extras.setdefault("log", {})["Support/reward_shaping"] = shaping.mean().detach()
        self.extras["log"]["Support/reward_hip_collision"] = collision.mean().detach()
    settling = self._standup_is_settling()
    reward = torch.where(settling & ~failed, torch.zeros_like(reward), reward)

    log = self.extras.setdefault("log", {})
    reward_term_names = (
        "recovery_height_deficit",
        "recovery_upright_deficit",
        "recovery_hold",
        "recovery_completion",
        "recovery_failure",
        "standing_stability",
        "standing_motion_penalty",
        "regularization",
    )
    for name in reward_term_names:
        log[f"StandUp/reward_{name}"] = torch.where(settling & ~failed, 0., getattr(terms, name)).mean().detach()
    log["StandUp/reward_recovery"] = torch.where(settling & ~failed, 0., terms.recovery).mean().detach()
    log["StandUp/reward_total"] = reward.mean().detach()

    def finite_mean(value: torch.Tensor) -> torch.Tensor:
        valid = self._standup_finite_this_step & torch.isfinite(value)
        return (torch.where(valid, value, 0.0).sum() / valid.sum().clamp_min(1)).detach()

    log["StandUp/root_height"] = finite_mean(root_height)
    log["StandUp/shoulder_height"] = finite_mean(shoulder_height)
    log["StandUp/upright_projection"] = finite_mean(upright)
    log["StandUp/body_height_score"] = finite_mean(body_score)
    log["StandUp/upright_score"] = finite_mean(upright_score)
    log["StandUp/upright_gate"] = finite_mean(
        smooth_height_gate(
            body_score,
            start=float(reward_cfg["upright_gate_start"]),
            full=float(reward_cfg["upright_gate_full"]),
        )
    )
    log["StandUp/target_root_height"] = torch.tensor(
        float(self._pair_recovery_target_root_height), device=self.device
    )
    log["StandUp/target_shoulder_height"] = torch.tensor(
        float(self._pair_recovery_target_shoulder_height), device=self.device
    )
    log["StandUp/settling_fraction"] = settling.float().mean().detach()
    log["StandUp/assistance_active_fraction"] = (
        (getattr(self, "_standup_assistance_actual_ratio", torch.zeros_like(reward)) > 1.e-8).float().mean().detach()
    )
    log["StandUp/assistance_actual_gravity_ratio"] = getattr(
        self, "_standup_assistance_actual_ratio", torch.zeros_like(reward)).mean().detach()
    log["StandUp/assistance_gravity_ratio"] = torch.tensor(
        float(self.cfg.pair_standup_assistance_current_gravity_ratio),
        device=self.device,
    )
    assistance_multiplier = getattr(
        self, "_standup_assistance_multiplier", torch.zeros_like(body_score)
    )
    log["StandUp/assistance_orientation_multiplier"] = getattr(
        self, "_standup_assistance_orientation_multiplier", torch.ones_like(body_score)
    ).mean().detach()
    log["StandUp/assistance_probe_fraction"] = (getattr(
        self, "_standup_assistance_draw", torch.ones_like(body_score)
    ) == 0).float().mean().detach()
    log["StandUp/assistance_height_multiplier"] = (
        assistance_multiplier.mean().detach()
    )
    assistance_latched = getattr(
        self,
        "_standup_assistance_latched_off",
        torch.zeros_like(body_score, dtype=torch.bool),
    )
    log["StandUp/assistance_latched_off_fraction"] = (
        assistance_latched.float().mean().detach()
    )
    if getattr(self.cfg,'pair_standup_episode_sampling',{}).get('enabled',False):
        mask=self.reset_time_outs & ~self.reset_terminated
        extractor=getattr(self,'_timeout_extractor',None)
        if extractor is not None and mask.any():
            final_terms=critic_terms(self,extractor)
            final,_=pack_critic_terms({k:v[mask] for k,v in final_terms.items()})
            self.extras['final_critic_observation']=final.detach().clone()
            self.extras['final_critic_mask']=mask.clone()
    return reward

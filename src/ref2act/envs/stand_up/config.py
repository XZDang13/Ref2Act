from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .standup_episodes import validate_episode_sampling
from .standup_stages import validate_stages, preparation_config
from .standup_observation import CRITIC_MODE
from .standup_assistance import assistance_settings
from .standup_reward import standup_hold_steps


def configure_task(
    cfg: Any,
    config: dict[str, Any],
    *,
    device: str,
    num_envs: int,
    seed: int,
) -> Any:
    """Configure the flat, non-goal G1 stand-up V1 environment."""

    terrain_type = str(getattr(cfg.terrain, "terrain_type", ""))
    if terrain_type != "plane":
        raise ValueError(
            f"PAIR stand-up V1 is flat-only and requires plane terrain, got {terrain_type!r}."
        )

    standup = config["standup"]
    initial = standup["initial_state"]
    initial_mode = str(initial.get("mode", "fixed_supine"))
    if initial_mode not in ("fixed_supine", "validated_crouch", "mixed_default_stand"):
        if initial_mode == "state_library":
            raise NotImplementedError(
                "state_library is reserved for stand-up V2; V1 supports fixed_supine only."
            )
        raise ValueError(f"Unknown stand-up initial-state mode: {initial_mode!r}.")
    root_euler = tuple(float(value) for value in initial["root_euler_xyz"])
    if len(root_euler) != 3:
        raise ValueError("standup.initial_state.root_euler_xyz must contain three values.")
    root_position = tuple(float(value) for value in initial["root_position"])
    if len(root_position) != 3:
        raise ValueError("standup.initial_state.root_position must contain three values.")

    cfg.episode_length_s = float(standup["episode_length_s"])
    cfg.pair_standup_episode_sampling = standup.get('episode_sampling', {})
    if cfg.pair_standup_episode_sampling.get('enabled', False):
        validate_episode_sampling(cfg.pair_standup_episode_sampling,standup.get('settling_s',.6))
        if standup.get('critic_observation') != CRITIC_MODE or standup.get('support',{}).get('reward_mode') != 'staged':
            raise ValueError('Episode sampling requires staged rewards and privileged single-frame critic')
    cfg.joint_position_reset_noise = 0.0
    cfg.action.mode = "offset"
    cfg.action.noise_scale = 0.0
    cfg.robot.spawn.articulation_props.enabled_self_collisions = bool(
        standup.get("enabled_self_collisions", True)
    )

    cfg.pair_standup_enabled = True
    cfg.pair_standup_initial_state_mode = initial_mode
    cfg.pair_standup_initial_root_position = root_position
    cfg.pair_standup_initial_root_euler_xyz = root_euler
    cfg.pair_standup_initial_joint_noise = float(
        initial.get("joint_position_noise", 0.0)
    )
    if initial_mode == "fixed_supine" and cfg.pair_standup_initial_joint_noise != 0.0:
        raise ValueError("Stand-up V1 fixed_supine requires zero joint-position noise.")
    if not math.isfinite(cfg.pair_standup_initial_joint_noise) or cfg.pair_standup_initial_joint_noise < 0:
        raise ValueError("Initial joint-position noise must be non-negative.")
    cfg.pair_standup_mixed_reset = None
    if initial_mode == 'mixed_default_stand':
        from .mixed_reset import load_mixed
        cfg.pair_standup_mixed_reset = load_mixed(initial)
        if not standup.get('episode_sampling',{}).get('enabled',False):
            raise ValueError('Mixed resets require asynchronous episode sampling')
    cfg.pair_standup_crouch_state = None
    if initial_mode == "validated_crouch":
        state_path = Path(initial["state_file"])
        if not state_path.is_absolute():
            state_path = Path(__file__).resolve().parents[3] / state_path
        state = json.loads(state_path.read_text())
        if state.get("quaternion_order") != "xyzw":
            raise ValueError("Validated crouch snapshot must use XYZW quaternions.")
        if len(state["root_state"]) != 13 or len(set(state["joint_names"])) != len(state["joint_names"]):
            raise ValueError("Invalid crouch root state or duplicate joint names.")
        for key in ("root_state", "joint_position", "joint_velocity"):
            values = torch.tensor(state[key])
            if not torch.isfinite(values).all():
                raise ValueError(f"Nonfinite crouch snapshot {key}.")
        if any(len(state[k]) != len(state["joint_names"]) for k in ("joint_position", "joint_velocity")):
            raise ValueError("Crouch joint arrays must match named joint count.")
        if abs(torch.tensor(state["root_state"][3:7]).norm().item() - 1.0) > 1.e-3:
            raise ValueError("Crouch quaternion must be normalized.")
        cfg.pair_standup_crouch_state = state
    cfg.pair_standup_settling_s = float(standup.get("settling_s", 0.6))
    if cfg.pair_standup_settling_s < 0.0:
        raise ValueError("standup.settling_s must be non-negative.")
    cfg.pair_standup_fall_cfg = dict(standup.get("fall_termination", {}))
    fall = cfg.pair_standup_fall_cfg
    if fall.get("enabled", False):
        if initial_mode != "validated_crouch":
            raise ValueError("Crouch fall termination is only valid for validated_crouch resets.")
        if not 0 < float(fall["immediate_root_height"]) < float(fall["minimum_root_height"]) < float(cfg.pair_standup_crouch_state["root_state"][2]):
            raise ValueError("Crouch fall heights must be positive and below the reset height.")
        if not 0 < float(fall["minimum_upright_projection"]) < 1:
            raise ValueError("Crouch fall upright threshold must be in (0, 1).")
        standup_hold_steps(float(fall["persistence_s"]), float(cfg.sim.dt) * int(cfg.decimation))
    cfg.pair_standup_support_cfg = dict(standup.get("support", {}))
    cfg.pair_standup_support_gamma = float(config["ppo"]["gamma"])
    if cfg.pair_standup_support_cfg.get("enabled", False):
        support = cfg.pair_standup_support_cfg
        mode = support.get("reward_mode", "potential")
        if mode not in ("potential", "transfer", "staged"):
            raise ValueError("support.reward_mode must be potential, transfer or staged.")
        if mode == "potential" and config["ppo"].get("bootstrap_timeouts", True):
            raise ValueError("Support pilot uses finite episodes: set bootstrap_timeouts=false.")
        if mode == "potential" and cfg.pair_standup_settling_s != 0:
            raise ValueError("Support shaping requires settling_s=0 to preserve transition rewards.")
        for key in ("persistence_s", "balance_sigma", "sole_height_tolerance", "contact_force_ratio"):
            if not math.isfinite(float(support[key])) or float(support[key]) <= 0:
                raise ValueError(f"support.{key} must be positive.")
        for key in ("shaping_scale", "hip_collision_scale"):
            if not math.isfinite(float(support[key])) or float(support[key]) < 0:
                raise ValueError(f"support.{key} must be non-negative.")
        if not 0 < float(support["success_min_load"]) <= 1:
            raise ValueError("support.success_min_load must be in (0, 1].")
        if not 0 <= float(support.get("success_min_per_foot_load", 0)) <= .5:
            raise ValueError("success_min_per_foot_load must be in [0, .5].")
        if not 0 < cfg.pair_standup_support_gamma <= 1:
            raise ValueError("Support discount must be in (0, 1].")
        if mode == "staged":
            validate_stages(support["stages"])
            preparation = preparation_config(support)
            for key in ("clearance_sigma", "flat_sigma", "shank_sigma", "load_filter_s",
                        "hip_forward_sigma", "hip_lateral_sigma", "sole_plant_sigma",
                        "placement_xy_sigma", "placement_height_sigma"):
                if not math.isfinite(float(preparation[key])) or preparation[key] <= 0:
                    raise ValueError(f"Invalid staged preparation {key}")
            for key, value in support["preparation"].items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise ValueError(f"Nonfinite staged preparation {key}")
            if not preparation['hip_forward_min'] < preparation['hip_forward_max']:
                raise ValueError("Invalid staged hip target")
        if mode == "transfer":
            transfer = support["transfer"]
            if transfer.get("height_mode", "legacy") not in ("legacy", "independent"):
                raise ValueError("Invalid transfer height_mode.")
            if transfer.get("flat_mode", "exponential") not in ("exponential", "signed_linear"):
                raise ValueError("Invalid flat_mode.")
            if transfer.get("shank_gate_source", "shoulder") not in ("shoulder", "pelvis"):
                raise ValueError("Invalid shank_gate_source.")
            if transfer.get("supported_rise_mode", "legacy") not in ("legacy", "joint"):
                raise ValueError("Invalid supported_rise_mode.")
            for key in ("com_preparation_scale", "bilateral_load_scale", "foot_placement_scale", "sole_plant_scale", "pelvis_lift_scale"):
                if not math.isfinite(float(transfer.get(key, 0))) or transfer.get(key, 0) < 0:
                    raise ValueError(f"Invalid {key}.")
            if transfer.get("bilateral_load_scale", 0) > 0 and not 0 < transfer.get("bilateral_load_target", 0) <= .5:
                raise ValueError("bilateral_load_target must be in (0, .5].")
            reference = transfer.get("placement_reference", "com")
            if reference not in ("com", "hips"):
                raise ValueError("placement_reference must be com or hips.")
            if transfer.get("placement_geometry", "directional") not in ("directional", "radial"):
                raise ValueError("Invalid placement_geometry.")
            if transfer.get("placement_geometry") == "radial":
                if reference != "hips" or not math.isfinite(float(transfer.get("hip_radial_tolerance", -1))) or transfer.get("hip_radial_tolerance", -1) < 0 or not math.isfinite(float(transfer.get("hip_radial_sigma", 0))) or transfer.get("hip_radial_sigma", 0) <= 0:
                    raise ValueError("Radial placement requires hip reference, finite tolerance and positive sigma.")
            if reference == "hips":
                for key in ("hip_forward_min", "hip_forward_max", "hip_forward_sigma", "hip_lateral_tolerance", "hip_lateral_sigma"):
                    if not math.isfinite(float(transfer[key])):
                        raise ValueError(f"{key} must be finite.")
                if not transfer["hip_forward_min"] < transfer["hip_forward_max"] or min(transfer["hip_forward_sigma"], transfer["hip_lateral_sigma"]) <= 0 or transfer["hip_lateral_tolerance"] < 0:
                    raise ValueError("Invalid hip placement target or scales.")
            if "pelvis_lift_hip_floor" in transfer:
                if reference != "hips" or transfer.get("foot_placement_scale", 0) <= 0 or not 0 < transfer["pelvis_lift_hip_floor"] <= 1:
                    raise ValueError("Hip-coupled lift requires hip placement and a floor in (0, 1].")
            if "sole_plant_hip_floor" in transfer:
                if reference != "hips" or transfer.get("foot_placement_scale", 0) <= 0 or not 0 <= transfer["sole_plant_hip_floor"] <= 1:
                    raise ValueError("Hip-coupled planting requires hip placement and a floor in [0, 1].")
            if "upper_preparation_full" in transfer:
                if transfer.get("height_mode") != "independent" or not 0 < transfer["upper_preparation_full"] <= 1:
                    raise ValueError("Upper preparation requires independent height and a threshold in (0, 1].")
            if "support_geometry_start" in transfer:
                if reference != "hips" or transfer.get("foot_placement_scale", 0) <= 0 or not 0 <= transfer["support_geometry_start"] < transfer.get("support_geometry_full", 0) <= 1:
                    raise ValueError("Prepared support requires hip placement and geometry thresholds in [0, 1].")
            if transfer.get("bilateral_squat", False):
                if (reference != "hips"
                    or transfer.get("sole_plant_scale", 0) <= 0
                    or transfer.get("allow_first_foot_preparation", False)
                    or transfer.get("height_mode") != "independent"
                    or transfer.get("supported_rise_mode") != "joint"):
                    raise ValueError("Bilateral squat requires bilateral hip/sole preparation and independent/joint height rewards.")
                for prefix in ("squat_plant", "squat_load"):
                    if not 0 <= transfer.get(prefix + "_start", -1) < transfer.get(prefix + "_full", 0) <= 1:
                        raise ValueError(f"Invalid {prefix} thresholds.")
                if not 0 <= transfer.get("squat_distance_full", -1) < transfer.get("squat_distance_start", 0) or not math.isfinite(float(transfer.get("squat_distance_start", 0))):
                    raise ValueError("Invalid squat distance thresholds.")
                if not 0 <= transfer.get("pelvis_preparation_full", -1) <= transfer["pelvis_lift_full"]:
                    raise ValueError("Invalid pelvis_preparation_full.")
            if transfer.get("foot_placement_scale", 0) > 0:
                for key in ("placement_xy_sigma", "placement_height_sigma"):
                    if not math.isfinite(float(transfer[key])) or transfer[key] <= 0:
                        raise ValueError(f"{key} must be finite and positive.")
                for key in ("placement_xy_tolerance", "placement_height_tolerance"):
                    if not math.isfinite(float(transfer[key])) or transfer[key] < 0:
                        raise ValueError(f"{key} must be finite and non-negative.")
            if transfer.get("sole_plant_scale", 0) > 0:
                if not math.isfinite(float(transfer["sole_plant_sigma"])) or transfer["sole_plant_sigma"] <= 0:
                    raise ValueError("sole_plant_sigma must be finite and positive.")
                if not math.isfinite(float(transfer["sole_plant_tolerance"])) or transfer["sole_plant_tolerance"] < 0:
                    raise ValueError("sole_plant_tolerance must be finite and non-negative.")
            if transfer.get("pelvis_lift_scale", 0) > 0:
                if not 0 <= transfer["pelvis_lift_start"] < transfer["pelvis_lift_full"] <= 1:
                    raise ValueError("Invalid pelvis lift interval.")
                if not 0 < transfer["pelvis_lift_load_full"] <= 1 or not 0 < transfer["righting_target_projection"] <= 1:
                    raise ValueError("Invalid pelvis lift support/righting threshold.")
            contact_scale = float(transfer.get("contact_scale", 0))
            if not math.isfinite(contact_scale) or contact_scale < 0:
                raise ValueError("contact_scale must be finite and non-negative.")
            righting_scale = float(transfer.get("righting_scale", 0))
            if not math.isfinite(righting_scale) or righting_scale < 0:
                raise ValueError("righting_scale must be finite and non-negative.")
            if righting_scale > 0:
                if transfer.get("height_mode") != "independent":
                    raise ValueError("Early righting requires independent height mode.")
                if not 0 < transfer["righting_target_projection"] < 1:
                    raise ValueError("righting_target_projection must be in (0, 1).")
            if transfer.get("height_mode") == "independent":
                for key in ("upper_height_scale", "pelvis_height_scale"):
                    if not math.isfinite(float(transfer[key])) or transfer[key] <= 0:
                        raise ValueError(f"support.transfer.{key} must be finite and positive.")
            if "shank_gate_start" in transfer:
                if not 0 <= transfer["shank_gate_start"] < transfer["shank_gate_full"] <= 1:
                    raise ValueError("Invalid shank height gate.")
            for key in ("clearance_sigma", "flat_sigma", "shank_sigma", "load_filter_s"):
                if not math.isfinite(float(transfer[key])) or transfer[key] <= 0:
                    raise ValueError(f"support.transfer.{key} must be positive.")
            for key in ("clearance_tolerance", "feet_low_scale", "feet_flat_scale",
                        "shank_upright_scale", "load_scale", "supported_rise_scale", "load_progress_scale"):
                if not math.isfinite(float(transfer[key])) or transfer[key] < 0:
                    raise ValueError(f"support.transfer.{key} must be non-negative.")
            if not 0 < transfer["shank_min_cos"] < 1:
                raise ValueError("support.transfer.shank_min_cos must be in (0, 1).")
            if not 0 <= transfer["rise_start"] < transfer["rise_full"] <= 1:
                raise ValueError("Invalid support.transfer rise interval.")
            if not 0 < transfer["unloaded_height_weight"] <= 1:
                raise ValueError("support.transfer.unloaded_height_weight must be in (0, 1].")
            if not 0 <= transfer["stability_load_start"] < transfer["stability_load_full"] <= 1:
                raise ValueError("Invalid support.transfer stability load interval.")
            if support["shaping_scale"] != 0:
                raise ValueError("Transfer rewards replace potential shaping; set shaping_scale=0.")
            com_scale = float(transfer.get("com_scale", 0))
            if not math.isfinite(com_scale) or com_scale < 0:
                raise ValueError("support.transfer.com_scale must be finite and non-negative.")
            if com_scale > 0:
                for prefix in ("com_load", "com_min_foot_load"):
                    if not 0 <= transfer[f"{prefix}_start"] < transfer[f"{prefix}_full"] <= 1:
                        raise ValueError(f"Invalid support.transfer.{prefix} interval.")

    targets = standup["targets"]
    target_source = str(targets.get("source", "default_standing"))
    if target_source != "default_standing":
        raise ValueError("Stand-up V1 targets.source must be 'default_standing'.")
    target_height_ratio = float(targets.get("height_ratio", 1.0))
    if not 0.0 < target_height_ratio <= 1.0:
        raise ValueError("Stand-up target height_ratio must be in (0, 1].")
    cfg.pair_recovery_target_height_ratio = target_height_ratio
    cfg.pair_recovery_explicit_root_target_height = None
    cfg.pair_recovery_explicit_shoulder_target_height = None
    cfg.pair_standup_target_source = target_source

    reward_cfg = dict(standup["reward"])
    policy_dt = float(cfg.sim.dt) * int(cfg.decimation)
    if reward_cfg.get('mode') == 'simple_v13':
        from .simple_rewards import validate_simple_rewards
        validate_simple_rewards(reward_cfg)
        if 'posture' in reward_cfg:
            support=cfg.pair_standup_support_cfg
            stages=support.get('stages',{})
            if not stages.get('hands',{}).get('enabled',False):
                raise ValueError('Posture reward requires ground-only hand sensing')
            if support['success_min_load'] >= stages['load_full']:
                raise ValueError('Posture foot-load ramp requires success_min_load < load_full')
        if not cfg.pair_standup_support_cfg.get('enabled') or cfg.pair_standup_support_cfg.get('reward_mode') != 'staged':
            raise ValueError('simple_v13 requires staged support measurements')
    else:
        # Old saved configs expressed the hold reward/window in policy steps.
        if "hold_reward_scale" not in reward_cfg:
            reward_cfg["hold_reward_scale"] = float(reward_cfg.pop("hold_reward")) / policy_dt
        start = float(reward_cfg["upright_gate_start"])
        full = float(reward_cfg["upright_gate_full"])
        if not 0.0 <= start < full <= 1.0:
            raise ValueError("Stand-up upright gate must satisfy 0 <= start < full <= 1.")
    cfg.pair_standup_reward_cfg = reward_cfg
    cfg.pair_standup_success_hold_s = float(
        standup["success_hold_s"] if "success_hold_s" in standup
        else float(standup["success_hold_steps"]) * policy_dt
    )
    standup_hold_steps(cfg.pair_standup_success_hold_s, policy_dt)
    cfg.pair_standup_success_upright = float(standup["success_upright_projection"])
    cfg.pair_standup_success_max_base_linear_velocity = float(
        standup["success_max_base_linear_velocity"]
    )
    cfg.pair_standup_success_max_base_angular_velocity = float(
        standup["success_max_base_angular_velocity"]
    )
    cfg.pair_standup_success_max_joint_velocity = float(
        standup["success_max_joint_velocity"]
    )
    if any(
        value <= 0.0
        for value in (
            cfg.pair_standup_success_max_base_linear_velocity,
            cfg.pair_standup_success_max_base_angular_velocity,
            cfg.pair_standup_success_max_joint_velocity,
        )
    ):
        raise ValueError("Stand-up success velocity limits must be positive.")
    cfg.pair_standup_max_joint_velocity = float(standup["max_joint_velocity"])
    cfg.pair_standup_max_base_angular_velocity = float(
        standup["max_base_angular_velocity"]
    )

    assistance = standup.get("assistance", {})
    cfg.pair_standup_assistance_settings = assistance_settings(assistance)
    cfg.pair_standup_assistance_body_name = assistance.get("body_name", "pelvis")
    if not isinstance(cfg.pair_standup_assistance_body_name, str) or not cfg.pair_standup_assistance_body_name:
        raise ValueError("assistance.body_name must be a nonempty exact rigid-body name.")
    cfg.pair_standup_assistance_enabled = bool(assistance.get("enabled", False))
    cfg.pair_standup_assistance_max_gravity_ratio = float(
        assistance.get("maximum_gravity_ratio", 0.60)
    )
    cfg.pair_standup_assistance_anneal_fraction = float(
        assistance.get("anneal_fraction", 0.70)
    )
    cfg.pair_standup_assistance_height_gate_start = float(
        assistance.get("height_gate_start", 0.65)
    )
    cfg.pair_standup_assistance_height_gate_full = float(
        assistance.get("height_gate_full", 0.85)
    )
    if not math.isfinite(cfg.pair_standup_assistance_max_gravity_ratio) or cfg.pair_standup_assistance_max_gravity_ratio < 0.0:
        raise ValueError("Stand-up assistance maximum_gravity_ratio must be finite and non-negative.")
    rise_transfer = standup.get('support', {}).get('stages', {}).get('rise_transfer')
    if rise_transfer is not None and cfg.pair_standup_assistance_max_gravity_ratio > 1-rise_transfer['minimum_ground_weight']:
        raise ValueError('Assistance exceeds the remaining-ground-weight safety floor')
    if not 0.0 < cfg.pair_standup_assistance_anneal_fraction <= 1.0:
        raise ValueError("Stand-up assistance anneal_fraction must be in (0, 1].")
    if not (
        0.0
        <= cfg.pair_standup_assistance_height_gate_start
        < cfg.pair_standup_assistance_height_gate_full
        <= 1.0
    ):
        raise ValueError("Stand-up assistance height gate must satisfy 0 <= start < full <= 1.")
    cfg.pair_standup_assistance_current_gravity_ratio = 0.0
    return cfg

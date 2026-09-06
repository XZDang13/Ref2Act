from __future__ import annotations

from importlib import import_module

__all__ = [
    "LocomotionEnv",
    "LocomotionObservation",
    "FlatLocomotionRewardCfg",
    "FlatLocomotionRewardInputs",
    "LocomotionRewardCfg",
    "LocomotionRewardInputs",
    "LOCOMOTION_COMMAND_CATEGORIES",
    "UniformVelocityCommandGenerator",
    "StratifiedVelocityCommandCfg",
    "StratifiedVelocityCommandGenerator",
    "VelocityCommandCfg",
    "make_velocity_command_generator",
    "Ref2ActFlatTerrainCfg",
    "Ref2ActSlopeTerrainCfg",
    "Ref2ActUnevenTerrainCfg",
    "compute_feet_air_time_positive_biped_reward",
    "compute_feet_phase_reward",
    "compute_feet_gait_reward",
    "compute_foot_clearance_reward",
    "compute_locomotion_gait_phase",
    "compute_locomotion_phase_features",
    "compute_locomotion_reward_terms",
    "compute_flat_locomotion_reward_terms",
    "phase_gait_targets",
    "phase_gait_signals",
    "expected_foot_height_from_phase",
    "default_locomotion_observation_spec",
    "make_locomotion_terrain_cfg",
]


_EXPORT_MODULES = {
    "LocomotionEnv": ".env",
    "LocomotionObservation": ".observation",
    "FlatLocomotionRewardCfg": ".task_rewards",
    "FlatLocomotionRewardInputs": ".task_rewards",
    "LocomotionRewardCfg": ".rewards",
    "LocomotionRewardInputs": ".rewards",
    "LOCOMOTION_COMMAND_CATEGORIES": ".commands",
    "UniformVelocityCommandGenerator": ".commands",
    "StratifiedVelocityCommandCfg": ".commands",
    "StratifiedVelocityCommandGenerator": ".commands",
    "VelocityCommandCfg": ".commands",
    "make_velocity_command_generator": ".commands",
    "Ref2ActFlatTerrainCfg": ".terrain",
    "Ref2ActSlopeTerrainCfg": ".terrain",
    "Ref2ActUnevenTerrainCfg": ".terrain",
    "compute_feet_air_time_positive_biped_reward": ".rewards",
    "compute_feet_phase_reward": ".rewards",
    "compute_feet_gait_reward": ".rewards",
    "compute_foot_clearance_reward": ".rewards",
    "compute_locomotion_gait_phase": ".rewards",
    "compute_locomotion_phase_features": ".rewards",
    "compute_locomotion_reward_terms": ".rewards",
    "compute_flat_locomotion_reward_terms": ".task_rewards",
    "phase_gait_targets": ".task_rewards",
    "phase_gait_signals": ".task_rewards",
    "expected_foot_height_from_phase": ".rewards",
    "default_locomotion_observation_spec": ".observation",
    "make_locomotion_terrain_cfg": ".terrain",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

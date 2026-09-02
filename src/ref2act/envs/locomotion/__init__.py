from __future__ import annotations

from importlib import import_module

__all__ = [
    "LocomotionEnv",
    "LocomotionObservation",
    "LocomotionRewardCfg",
    "LocomotionRewardInputs",
    "UniformVelocityCommandGenerator",
    "VelocityCommandCfg",
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
    "expected_foot_height_from_phase",
    "default_locomotion_observation_spec",
    "make_locomotion_terrain_cfg",
]


_EXPORT_MODULES = {
    "LocomotionEnv": ".env",
    "LocomotionObservation": ".observation",
    "LocomotionRewardCfg": ".rewards",
    "LocomotionRewardInputs": ".rewards",
    "UniformVelocityCommandGenerator": ".commands",
    "VelocityCommandCfg": ".commands",
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

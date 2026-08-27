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
    "compute_locomotion_reward_terms",
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
    "compute_locomotion_reward_terms": ".rewards",
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

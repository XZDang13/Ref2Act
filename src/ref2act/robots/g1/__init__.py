from __future__ import annotations

from importlib import import_module

from .spec import G1_23_DOF_JOINT_ORDER, G1_23_DOF_SPEC

__all__ = [
    "G1_CFG",
    "G1_23_DOF_JOINT_ORDER",
    "G1_23_DOF_SPEC",
    "G1DomainRandCfg",
    "G1FlatLocomotionEnvCfg",
    "G1FlatStandUpEnvCfg",
    "G1MixedTerrainLocomotionEnvCfg",
    "G1MotionTrackingEnvCfg",
    "G1MotionTrackingRoughEnvCfg",
    "G1TrainingEventCfg",
    "G1SlopeLocomotionEnvCfg",
    "G1UnevenLocomotionEnvCfg",
    "JOINT_ORDER",
    "MotionViewerCfg",
]


_EXPORT_MODULES = {
    "G1_CFG": ".articulation",
    "G1DomainRandCfg": ".randomization_presets",
    "G1FlatLocomotionEnvCfg": ".locomotion_env_cfg",
    "G1FlatStandUpEnvCfg": ".stand_up_env_cfg",
    "G1MixedTerrainLocomotionEnvCfg": ".locomotion_env_cfg",
    "G1MotionTrackingEnvCfg": ".env_cfg",
    "G1MotionTrackingRoughEnvCfg": ".env_cfg",
    "G1TrainingEventCfg": ".randomization_presets",
    "G1SlopeLocomotionEnvCfg": ".locomotion_env_cfg",
    "G1UnevenLocomotionEnvCfg": ".locomotion_env_cfg",
    "JOINT_ORDER": ".env_cfg",
    "MotionViewerCfg": ".env_cfg",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

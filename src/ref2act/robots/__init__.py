from __future__ import annotations

from .spec import RobotSpec, resolve_robot_spec

__all__ = [
    "G1FlatLocomotionEnvCfg",
    "G1FlatStandUpEnvCfg",
    "G1MixedTerrainLocomotionEnvCfg",
    "G1MotionTrackingEnvCfg",
    "G1MotionTrackingRoughEnvCfg",
    "G1SlopeLocomotionEnvCfg",
    "G1UnevenLocomotionEnvCfg",
    "RobotSpec",
    "resolve_robot_spec",
]


def __getattr__(name: str):
    if name in {
        "G1FlatLocomotionEnvCfg",
        "G1FlatStandUpEnvCfg",
        "G1MixedTerrainLocomotionEnvCfg",
        "G1MotionTrackingEnvCfg",
        "G1MotionTrackingRoughEnvCfg",
        "G1SlopeLocomotionEnvCfg",
        "G1UnevenLocomotionEnvCfg",
    }:
        from . import g1

        return getattr(g1, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

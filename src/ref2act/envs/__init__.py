from __future__ import annotations

__all__ = ["LeggedRobotEnv", "LocomotionEnv", "MotionTrackingEnv"]


def __getattr__(name: str):
    if name == "LeggedRobotEnv":
        from .base import LeggedRobotEnv

        return LeggedRobotEnv
    if name == "MotionTrackingEnv":
        from .motion_tracking.env import MotionTrackingEnv

        return MotionTrackingEnv
    if name == "LocomotionEnv":
        from .locomotion.env import LocomotionEnv

        return LocomotionEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

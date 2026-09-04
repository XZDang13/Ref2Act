from __future__ import annotations

__all__ = ["LeggedRobotEnv", "LocomotionEnv", "MotionTrackingEnv", "StandUpEnv"]


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
    if name == "StandUpEnv":
        from .stand_up.env import StandUpEnv

        return StandUpEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

__all__ = ["MotionTrackingEnv"]


def __getattr__(name: str):
    if name == "MotionTrackingEnv":
        from .env import MotionTrackingEnv

        return MotionTrackingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

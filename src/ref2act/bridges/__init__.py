from __future__ import annotations

__all__ = ["MujocoEnv"]


def __getattr__(name: str):
    if name == "MujocoEnv":
        from .mujoco.env import MujocoEnv

        return MujocoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

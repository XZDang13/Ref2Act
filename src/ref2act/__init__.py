from __future__ import annotations

__version__ = "0.2.1"


def _register_default_envs() -> None:
    from .envs.motion_tracking.registry import register_envs

    register_envs()


_register_default_envs()

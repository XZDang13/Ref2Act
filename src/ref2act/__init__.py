from __future__ import annotations

__version__ = "0.2.1"


def _register_default_envs() -> None:
    from .envs.locomotion.registry import register_envs as register_locomotion_envs
    from .envs.motion_tracking.registry import register_envs

    register_envs()
    register_locomotion_envs()


_register_default_envs()

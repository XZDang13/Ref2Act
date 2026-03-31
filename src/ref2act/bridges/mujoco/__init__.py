from __future__ import annotations

__all__ = [
    "IsaacLabMujocoAction",
    "IsaacLabMujocoObservation",
    "MujocoActionBuilder",
    "MujocoActionContext",
    "MujocoActionOutput",
    "MujocoEnv",
    "MujocoObservationBuilder",
    "MujocoObservationContext",
]


def __getattr__(name: str):
    if name == "MujocoEnv":
        from .env import MujocoEnv

        return MujocoEnv
    if name in {
        "IsaacLabMujocoAction",
        "MujocoActionBuilder",
        "MujocoActionContext",
        "MujocoActionOutput",
    }:
        from .action import IsaacLabMujocoAction, MujocoActionBuilder, MujocoActionContext, MujocoActionOutput

        exports = {
            "IsaacLabMujocoAction": IsaacLabMujocoAction,
            "MujocoActionBuilder": MujocoActionBuilder,
            "MujocoActionContext": MujocoActionContext,
            "MujocoActionOutput": MujocoActionOutput,
        }
        return exports[name]
    if name in {"IsaacLabMujocoObservation", "MujocoObservationBuilder", "MujocoObservationContext"}:
        from .observation import IsaacLabMujocoObservation, MujocoObservationBuilder, MujocoObservationContext

        exports = {
            "IsaacLabMujocoObservation": IsaacLabMujocoObservation,
            "MujocoObservationBuilder": MujocoObservationBuilder,
            "MujocoObservationContext": MujocoObservationContext,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

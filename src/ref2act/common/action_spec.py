from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch


@dataclass
class ActionModeContext:
    raw_action: torch.Tensor
    action_scale: torch.Tensor
    action_offset: torch.Tensor
    joint_pos_limits_lower: torch.Tensor
    joint_pos_limits_upper: torch.Tensor
    current_joint_pos_loader: Callable[[], torch.Tensor] | None = None
    reference_joint_pos_loader: Callable[[], torch.Tensor] | None = None
    _current_joint_pos: torch.Tensor | None = field(default=None, init=False, repr=False)
    _reference_joint_pos: torch.Tensor | None = field(default=None, init=False, repr=False)

    def get_current_joint_pos(self) -> torch.Tensor:
        if self.current_joint_pos_loader is None:
            raise ValueError("Current joint positions are not available for this action context.")
        if self._current_joint_pos is None:
            self._current_joint_pos = self.current_joint_pos_loader()
        return self._current_joint_pos

    def get_reference_joint_pos(self) -> torch.Tensor:
        if self.reference_joint_pos_loader is None:
            raise ValueError("Reference joint positions are not available for this action context.")
        if self._reference_joint_pos is None:
            self._reference_joint_pos = self.reference_joint_pos_loader()
        return self._reference_joint_pos


class ClampPolicy(Protocol):
    def clamp(self, context: ActionModeContext, target_joint_pos: torch.Tensor) -> torch.Tensor:
        ...


class JointLimitClampPolicy:
    def clamp(self, context: ActionModeContext, target_joint_pos: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            target_joint_pos,
            context.joint_pos_limits_lower,
            context.joint_pos_limits_upper,
        )


class ActionModeStrategy(Protocol):
    name: str

    def compute_target_joint_pos(
        self,
        context: ActionModeContext,
        applied_action: torch.Tensor,
    ) -> torch.Tensor:
        ...


class _AbsoluteActionModeStrategy:
    def __init__(self, name: str) -> None:
        self.name = name

    def compute_target_joint_pos(
        self,
        context: ActionModeContext,
        applied_action: torch.Tensor,
    ) -> torch.Tensor:
        return applied_action * context.action_scale + context.action_offset


class ResidualActionModeStrategy:
    name = "residual"

    def compute_target_joint_pos(
        self,
        context: ActionModeContext,
        applied_action: torch.Tensor,
    ) -> torch.Tensor:
        return context.get_reference_joint_pos() + applied_action * context.action_scale


class CurrentResidualActionModeStrategy:
    name = "current_residual"

    def compute_target_joint_pos(
        self,
        context: ActionModeContext,
        applied_action: torch.Tensor,
    ) -> torch.Tensor:
        return context.get_current_joint_pos() + applied_action * context.action_scale


ACTION_MODE_REGISTRY: dict[str, ActionModeStrategy] = {
    "median": _AbsoluteActionModeStrategy("median"),
    "offset": _AbsoluteActionModeStrategy("offset"),
    "absolute": _AbsoluteActionModeStrategy("absolute"),
    "residual": ResidualActionModeStrategy(),
    "current_residual": CurrentResidualActionModeStrategy(),
}

DEFAULT_CLAMP_POLICY = JointLimitClampPolicy()


def normalize_action_mode(action_mode: object | str) -> str:
    if isinstance(action_mode, str):
        normalized = action_mode
    elif hasattr(action_mode, "name"):
        normalized = str(action_mode.name)
    else:
        normalized = str(action_mode).split(".")[-1]

    normalized = normalized.replace("-", "_").lower()
    if normalized in {"currentresidual", "current_residual"}:
        return "current_residual"
    return normalized


def resolve_action_mode_strategy(action_mode: object | str) -> ActionModeStrategy:
    normalized = normalize_action_mode(action_mode)
    try:
        return ACTION_MODE_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported action mode: {action_mode}") from exc


__all__ = [
    "ACTION_MODE_REGISTRY",
    "ActionModeContext",
    "ActionModeStrategy",
    "ClampPolicy",
    "CurrentResidualActionModeStrategy",
    "DEFAULT_CLAMP_POLICY",
    "JointLimitClampPolicy",
    "ResidualActionModeStrategy",
    "normalize_action_mode",
    "resolve_action_mode_strategy",
]

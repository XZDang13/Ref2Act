from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RobotSpec:
    """Simulator-independent robot topology required by Ref2Act environments.

    Environment code consumes this contract instead of importing robot-specific
    joint or body names.  The articulation configuration remains backend-owned;
    this object only describes the topology shared by policies, rewards, data
    collection, and validation.
    """

    name: str
    joint_order: tuple[str, ...]
    root_body: str
    anchor_body: str
    foot_bodies: tuple[str, ...]
    key_bodies: tuple[str, ...] = ()
    end_effector_bodies: tuple[str, ...] = ()
    illegal_contact_patterns: tuple[str, ...] = ()
    leg_joint_patterns: tuple[str, ...] = ()
    torso_joint_patterns: tuple[str, ...] = ()
    arm_joint_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RobotSpec.name must not be empty.")
        if not self.joint_order:
            raise ValueError("RobotSpec.joint_order must not be empty.")
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("RobotSpec.joint_order contains duplicate names.")
        if not self.root_body or not self.anchor_body:
            raise ValueError("RobotSpec root_body and anchor_body must not be empty.")
        if not self.foot_bodies:
            raise ValueError("RobotSpec.foot_bodies must not be empty.")

    @property
    def action_dim(self) -> int:
        return len(self.joint_order)

    def validate_joint_names(self, actual_names: Sequence[str]) -> None:
        actual = tuple(actual_names)
        if len(set(actual)) != len(actual):
            raise ValueError(f"Robot '{self.name}' simulator joint names contain duplicates.")
        missing = sorted(set(self.joint_order).difference(actual))
        extra = sorted(set(actual).difference(self.joint_order))
        if not missing and not extra:
            return
        raise ValueError(
            f"Robot '{self.name}' joint topology mismatch: "
            f"missing={missing}, extra={extra}."
        )

    def policy_order_indices(self, actual_names: Sequence[str]) -> tuple[int, ...]:
        """Indices that convert a simulator-order tensor to policy order."""

        self.validate_joint_names(actual_names)
        actual = tuple(actual_names)
        return tuple(actual.index(name) for name in self.joint_order)

    def simulator_order_indices(self, actual_names: Sequence[str]) -> tuple[int, ...]:
        """Indices that convert a policy-order tensor to simulator order."""

        self.validate_joint_names(actual_names)
        return tuple(self.joint_order.index(name) for name in actual_names)

    def body_index(self, actual_names: Sequence[str], body_name: str) -> int:
        try:
            return tuple(actual_names).index(body_name)
        except ValueError as exc:
            raise ValueError(f"Robot '{self.name}' is missing required body '{body_name}'.") from exc

    def body_indices(self, actual_names: Sequence[str], body_names: Sequence[str]) -> tuple[int, ...]:
        return tuple(self.body_index(actual_names, name) for name in body_names)


def resolve_robot_spec(name: str) -> RobotSpec:
    if name == "g1_23dof":
        from ._g1_spec import G1_23_DOF_SPEC

        return G1_23_DOF_SPEC
    raise KeyError(f"Unknown robot spec: {name!r}.")


__all__ = ["RobotSpec", "resolve_robot_spec"]

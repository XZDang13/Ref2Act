from __future__ import annotations

import pytest

from ref2act.robots.g1.spec import G1_23_DOF_JOINT_ORDER, G1_23_DOF_SPEC
from ref2act.robots.spec import RobotSpec


def test_g1_23dof_spec_has_expected_policy_topology() -> None:
    assert G1_23_DOF_SPEC.action_dim == 23
    assert G1_23_DOF_SPEC.joint_order == G1_23_DOF_JOINT_ORDER
    assert G1_23_DOF_SPEC.root_body == "pelvis"
    assert G1_23_DOF_SPEC.anchor_body == "pelvis"
    assert G1_23_DOF_SPEC.foot_bodies == (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )


def test_robot_spec_validates_exact_joint_order() -> None:
    G1_23_DOF_SPEC.validate_joint_names(G1_23_DOF_JOINT_ORDER)

    reordered = list(G1_23_DOF_JOINT_ORDER)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    G1_23_DOF_SPEC.validate_joint_names(reordered)
    assert G1_23_DOF_SPEC.policy_order_indices(reordered)[:2] == (1, 0)
    assert G1_23_DOF_SPEC.simulator_order_indices(reordered)[:2] == (1, 0)


def test_robot_spec_rejects_duplicate_joint_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RobotSpec(
            name="bad",
            joint_order=("joint", "joint"),
            root_body="base",
            anchor_body="base",
            foot_bodies=("foot",),
        )


def test_robot_spec_resolves_required_body_indices() -> None:
    body_names = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link")
    assert G1_23_DOF_SPEC.body_indices(body_names, G1_23_DOF_SPEC.foot_bodies) == (1, 2)
    with pytest.raises(ValueError, match="missing required body"):
        G1_23_DOF_SPEC.body_index(body_names, "torso_link")

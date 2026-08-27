from __future__ import annotations

from ref2act.robots.spec import RobotSpec


G1_23_DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

G1_23_DOF_SPEC = RobotSpec(
    name="g1_23dof",
    joint_order=G1_23_DOF_JOINT_ORDER,
    root_body="pelvis",
    anchor_body="pelvis",
    foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    key_bodies=(
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_rubber_hand_link",
        "right_rubber_hand_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
    end_effector_bodies=(
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand_link",
        "right_rubber_hand_link",
    ),
    illegal_contact_patterns=(
        "pelvis",
        "torso_link",
        ".*_hip_.*_link",
        ".*_knee_link",
        ".*_shoulder_.*_link",
        ".*_elbow_link",
        ".*_wrist_.*|.*_wrist_roll",
        ".*_rubber_hand_link",
    ),
    leg_joint_patterns=(".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"),
    torso_joint_patterns=("waist_yaw_joint",),
    arm_joint_patterns=(".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"),
)


__all__ = ["G1_23_DOF_JOINT_ORDER", "G1_23_DOF_SPEC"]

from pathlib import Path

import pytest


pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdPhysics  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
G1_ASSET_ROOT = REPO_ROOT / "src" / "ref2act" / "assets" / "robots" / "g1"
G1_23_USD_PATH = G1_ASSET_ROOT / "g1_23_dof" / "g1_23dof.usda"
G1_29_USD_PATH = G1_ASSET_ROOT / "g1_29_dof" / "g1_mocap.usda"

G1_23_JOINT_NAMES = [
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
]

G1_23_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_toe_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_toe_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll",
    "left_rubber_hand_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll",
    "right_rubber_hand_link",
]

G1_29_EXTRA_JOINT_NAMES = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}

G1_29_TRACKING_BODY_NAMES = {
    "waist_yaw_link",
    "waist_roll_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
}


def _stage_names(path: Path) -> tuple[list[str], list[str]]:
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    assert stage is not None

    body_names = []
    joint_names = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_names.append(prim.GetName())
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_names.append(prim.GetName())
    return body_names, joint_names


def test_g1_23_usd_matches_retargeter_motion_topology() -> None:
    body_names, joint_names = _stage_names(G1_23_USD_PATH)

    assert body_names == G1_23_BODY_NAMES
    assert set(joint_names) == set(G1_23_JOINT_NAMES)
    assert len(joint_names) == len(G1_23_JOINT_NAMES)


def test_g1_29_usd_contains_retargeter_motion_topology() -> None:
    body_names, joint_names = _stage_names(G1_29_USD_PATH)

    assert len(joint_names) == 29
    assert set(G1_23_JOINT_NAMES).issubset(joint_names)
    assert G1_29_EXTRA_JOINT_NAMES.issubset(joint_names)
    assert G1_29_TRACKING_BODY_NAMES.issubset(body_names)


def test_flat_rough_configs_and_gym_registry_instantiate() -> None:
    import gymnasium as gym

    import ref2act  # noqa: F401
    from ref2act.robots.g1 import G1MotionTrackingEnvCfg, G1MotionTrackingRoughEnvCfg

    assert isinstance(G1MotionTrackingEnvCfg(), G1MotionTrackingEnvCfg)
    assert isinstance(G1MotionTrackingRoughEnvCfg(), G1MotionTrackingRoughEnvCfg)
    assert gym.spec("G1MotionTracking-v0").id == "G1MotionTracking-v0"
    assert gym.spec("G1MotionTrackingRough-v0").id == "G1MotionTrackingRough-v0"

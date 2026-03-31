from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
G1_MJCF_PATH = REPO_ROOT / "src" / "ref2act" / "assets" / "robots" / "g1" / "g1_23dof_rubber_hand.xml"
G1_SCENE_PATH = REPO_ROOT / "src" / "ref2act" / "assets" / "scenes" / "g1" / "scene.xml"


def _load_g1_mjcf_root() -> ET.Element:
    return ET.parse(G1_MJCF_PATH).getroot()


def test_g1_mjcf_uses_expected_rubber_hand_body_names() -> None:
    root = _load_g1_mjcf_root()
    body_names = {body.attrib["name"] for body in root.iter("body") if "name" in body.attrib}

    assert "left_rubber_hand" in body_names
    assert "right_rubber_hand" in body_names


def test_g1_mjcf_motor_ranges_match_joint_force_ranges() -> None:
    root = _load_g1_mjcf_root()

    joint_force_ranges = {
        joint.attrib["name"]: joint.attrib["actuatorfrcrange"]
        for joint in root.iter("joint")
        if "name" in joint.attrib and "actuatorfrcrange" in joint.attrib
    }

    motors = [
        motor for motor in root.iter("motor") if "joint" in motor.attrib and "ctrlrange" in motor.attrib
    ]

    assert motors, "Expected the G1 MJCF to define motors."

    for motor in motors:
        joint_name = motor.attrib["joint"]
        assert joint_name in joint_force_ranges
        assert motor.attrib["ctrlrange"] == joint_force_ranges[joint_name]


def test_g1_scene_include_resolves_to_packaged_robot_mjcf() -> None:
    root = ET.parse(G1_SCENE_PATH).getroot()
    include = root.find("include")

    assert include is not None
    include_file = include.attrib["file"]
    include_path = (G1_SCENE_PATH.parent / include_file).resolve()

    assert include_path == G1_MJCF_PATH.resolve()
    assert include_path.exists()


def test_g1_scene_can_be_loaded_by_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(G1_SCENE_PATH))

    assert model.nq > 0
    assert model.nv > 0
    assert model.nu > 0

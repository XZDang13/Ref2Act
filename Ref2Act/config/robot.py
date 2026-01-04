from importlib import resources as importlib_resources
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

def _default_assets_root() -> Path:
    try:
        assets_root = Path(importlib_resources.files("Ref2Act") / "assets")
        if assets_root.exists():
            return assets_root
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent / "assets"

_assets_root = _default_assets_root()
g1_usd_path = str(
    _assets_root
    / "G1"
    / "g1_23_dof_rubber_hand"
    / "g1_23dof_rubber_hand"
    / "g1_23dof_rubber_hand.usd"
)

g1_static_usd_path = str(
    _assets_root
    / "G1"
    / "g1_23_dof_rubber_hand_static"
    / "g1_23dof_rubber_hand"
    / "g1_23dof_rubber_hand.usd"
)

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=g1_usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos={
                ".*_hip_pitch_joint": -0.20,
                ".*_knee_joint": 0.42,
                ".*_ankle_pitch_joint": -0.23,
                "left_shoulder_roll_joint": 0.16,
                "left_shoulder_pitch_joint": 0.35,
                "right_shoulder_roll_joint": -0.16,
                "right_shoulder_pitch_joint": 0.35,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                    ".*_ankle_pitch_joint",
                    ".*_ankle_roll_joint"
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0,
                    ".*_knee_joint": 139.0,
                    ".*_ankle_pitch_joint": 50.0,
                    ".*_ankle_roll_joint": 50.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                    ".*_ankle_pitch_joint": 37.0,
                    ".*_ankle_roll_joint": 37.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 40.179238471,
                    ".*_hip_roll_joint": 99.098427777,
                    ".*_hip_pitch_joint": 40.179238471,
                    ".*_knee_joint": 99.098427777,
                    ".*_ankle_pitch_joint": 28.501246196,
                    ".*_ankle_roll_joint": 28.501246196,
                },
                damping={
                    ".*_hip_yaw_joint": 2.557889765,
                    ".*_hip_roll_joint": 6.308801854,
                    ".*_hip_pitch_joint": 2.557889765,
                    ".*_knee_joint": 6.308801854,
                    ".*_ankle_pitch_joint": 1.814445687,
                    ".*_ankle_roll_joint": 1.814445687,
                },
                friction=0.003,
                armature={
                    ".*_hip_yaw_joint": 0.010177520,
                    ".*_hip_roll_joint": 0.025101925,
                    ".*_hip_pitch_joint": 0.010177520,
                    ".*_knee_joint": 0.025101925,
                    ".*_ankle_pitch_joint": 0.007219450,
                    ".*_ankle_roll_joint": 0.007219450,
                },
            ),
            "bodies": ImplicitActuatorCfg(
                joint_names_expr=[
                    "waist_yaw_joint",
                ],
                effort_limit_sim={
                    "waist_yaw_joint": 88.0,
                },
                velocity_limit_sim={
                    "waist_yaw_joint": 32.0,
                },
                stiffness={
                    "waist_yaw_joint": 40.179238471,
                },
                damping={
                    "waist_yaw_joint": 2.557889765,
                },
                friction=0.003,
                armature={
                    "waist_yaw_joint": 0.007219450,
                },
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_.*": 25.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 37.0,
                    ".*_shoulder_roll_joint": 37.0,
                    ".*_shoulder_yaw_joint": 37.0,
                    ".*_elbow_joint": 37.0,
                    ".*_wrist_.*": 37.0,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": 14.250623098,
                    ".*_shoulder_roll_joint": 14.250623098,
                    ".*_shoulder_yaw_joint": 14.250623098,
                    ".*_elbow_joint": 14.250623098,
                    ".*_wrist_.*": 14.250623098,
                },
                damping={
                    ".*_shoulder_pitch_joint": 0.907222843,
                    ".*_shoulder_roll_joint": 0.907222843,
                    ".*_shoulder_yaw_joint": 0.907222843,
                    ".*_elbow_joint": 0.907222843,
                    ".*_wrist_.*": 0.907222843,
                },
                friction=0.003,
                armature={
                    ".*_shoulder_pitch_joint": 0.003609725,
                    ".*_shoulder_roll_joint": 0.003609725,
                    ".*_shoulder_yaw_joint": 0.003609725,
                    ".*_elbow_joint": 0.003609725,
                    ".*_wrist_.*": 0.003609725,
                },
            ),
        },
)

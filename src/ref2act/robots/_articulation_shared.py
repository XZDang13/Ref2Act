import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from ref2act.assets import robot_asset_path

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2  # 14.25062309787429
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2  # 40.17923847137318
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2  # 99.09842777666113
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2  # 16.77832748089279

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ  # 0.907222843292423
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # 2.5578897650279457
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # 6.3088018534966395
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ  # 1.06814150219

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(
            robot_asset_path(
                "g1",
                "g1_23_dof_rubber_hand",
                "g1_23dof_rubber_hand",
                "g1_23dof_rubber_hand.usd",
            )
        ),
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
            pos=(0.0, 0.0, 0.76),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": 0.6,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
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
                    ".*_hip_yaw_joint": STIFFNESS_7520_14,
                    ".*_hip_roll_joint": STIFFNESS_7520_22,
                    ".*_hip_pitch_joint": STIFFNESS_7520_14,
                    ".*_knee_joint": STIFFNESS_7520_22,
                    ".*_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
                    ".*_ankle_roll_joint": 2.0 * STIFFNESS_5020,
                },
                damping={
                    ".*_hip_yaw_joint": DAMPING_7520_14,
                    ".*_hip_roll_joint":DAMPING_7520_22,
                    ".*_hip_pitch_joint": DAMPING_7520_14,
                    ".*_knee_joint": DAMPING_7520_22,
                    ".*_ankle_pitch_joint": 2.0 * DAMPING_5020,
                    ".*_ankle_roll_joint": 2.0 * DAMPING_5020,
                },
                armature={
                    ".*_hip_yaw_joint": ARMATURE_7520_14,
                    ".*_hip_roll_joint": ARMATURE_7520_22,
                    ".*_hip_pitch_joint": ARMATURE_7520_14,
                    ".*_knee_joint": ARMATURE_7520_22,
                    ".*_ankle_pitch_joint": 2.0 * ARMATURE_5020,
                    ".*_ankle_roll_joint": 2.0 * ARMATURE_5020,
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
                    "waist_yaw_joint": STIFFNESS_7520_14,
                },
                damping={
                    "waist_yaw_joint": DAMPING_7520_14,
                },
                armature={
                    "waist_yaw_joint": ARMATURE_7520_14,
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
                    ".*_shoulder_pitch_joint": STIFFNESS_5020,
                    ".*_shoulder_roll_joint": STIFFNESS_5020,
                    ".*_shoulder_yaw_joint": STIFFNESS_5020,
                    ".*_elbow_joint": STIFFNESS_5020,
                    ".*_wrist_.*": STIFFNESS_5020,
                },
                damping={
                    ".*_shoulder_pitch_joint": DAMPING_5020,
                    ".*_shoulder_roll_joint": DAMPING_5020,
                    ".*_shoulder_yaw_joint": DAMPING_5020,
                    ".*_elbow_joint": DAMPING_5020,
                    ".*_wrist_.*": DAMPING_5020,
                },
                armature={
                    ".*_shoulder_pitch_joint": ARMATURE_5020,
                    ".*_shoulder_roll_joint": ARMATURE_5020,
                    ".*_shoulder_yaw_joint": ARMATURE_5020,
                    ".*_elbow_joint": ARMATURE_5020,
                    ".*_wrist_.*": ARMATURE_5020,
                },
            ),
        },
)


ARMATURE_4438 = 0.008419
ARMATURE_5047 = 0.044277

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_4438 = 30
STIFFNESS_5047 = 80

DAMPING_4438 = 0.6
DAMPING_5047 = 1.1


PI_PLUS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(robot_asset_path("hightorque", "pi_plus", "urdf", "pi_plus_20dof.urdf")),
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
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.351),
        joint_pos={
            ".*_hip_pitch_joint": -0.25,
            ".*_calf_joint": 0.65,
            ".*_ankle_pitch_joint": -0.4,
            ".*_elbow_joint": 0.0,
            "l_shoulder_roll_joint": 0.0,
            "l_shoulder_pitch_joint": 0.0,
            "r_shoulder_roll_joint": 0.0,
            "r_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_thigh_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_calf_joint",
            ],
            effort_limit_sim={
                ".*_thigh_joint": 30.0,
                ".*_hip_roll_joint": 30.0,
                ".*_hip_pitch_joint": 30.0,
                ".*_calf_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_thigh_joint": 8.0,
                ".*_hip_roll_joint": 8.0,
                ".*_hip_pitch_joint": 8.0,
                ".*_calf_joint": 8.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_5047,
                ".*_hip_roll_joint": STIFFNESS_5047,
                ".*_thigh_joint": STIFFNESS_5047,
                ".*_calf_joint": STIFFNESS_5047,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_5047,
                ".*_hip_roll_joint": DAMPING_5047,
                ".*_thigh_joint": DAMPING_5047,
                ".*_calf_joint": DAMPING_5047,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_5047,
                ".*_hip_roll_joint": ARMATURE_5047,
                ".*_thigh_joint": ARMATURE_5047,
                ".*_calf_joint": ARMATURE_5047,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=30.0,
            velocity_limit_sim=17.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_5047,
            damping=DAMPING_5047,
            armature=ARMATURE_5047,
        ),
        
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_upper_arm_joint",
                ".*_elbow_joint",
                ".*_wrist_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 20.0,
                ".*_shoulder_roll_joint": 20.0,
                ".*_upper_arm_joint": 20.0,
                ".*_elbow_joint": 20.0,
                ".*_wrist_joint": 20.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 17.0,
                ".*_shoulder_roll_joint": 17.0,
                ".*_upper_arm_joint": 17.0,
                ".*_elbow_joint": 17.0,
                ".*_wrist_joint": 17.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_4438,
                ".*_shoulder_roll_joint": STIFFNESS_4438,
                ".*_upper_arm_joint": STIFFNESS_4438,
                ".*_elbow_joint": STIFFNESS_4438,
                ".*_wrist_joint": STIFFNESS_4438,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_4438,
                ".*_shoulder_roll_joint": DAMPING_4438,
                ".*_upper_arm_joint": DAMPING_4438,
                ".*_elbow_joint": DAMPING_4438,
                ".*_wrist_joint": DAMPING_4438,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_4438,
                ".*_shoulder_roll_joint": ARMATURE_4438,
                ".*_upper_arm_joint": ARMATURE_4438,
                ".*_elbow_joint": ARMATURE_4438,
                ".*_wrist_joint": ARMATURE_4438,
            },
        ),
    },
)

from enum import Enum
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from .robot import G1_CFG, PI_PLUS_CFG
from ..curriculum import TerminationCurriculumCfg
from ..domain_randomization import (
    randomize_action_latency,
    randomize_group_actuator_gains,
    randomize_group_body_masses,
    randomize_rigid_body_collider_offsets_by_body,
    randomize_rigid_body_com_from_default,
)
from ..sampler import SamplerMod, SamplingStrategy

G1_ACTION_LATENCY_RANGE = (0, 2)
PI_PLUS_ACTION_LATENCY_RANGE = (0, 2)

G1_PUSH_VELOCITY_RANGE = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "yaw": (-0.4, 0.4),
}

PI_PLUS_PUSH_VELOCITY_RANGE = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "yaw": (-0.3, 0.3),
}

G1_CONTACT_BODIES = "left_ankle_roll_link|right_ankle_roll_link|left_rubber_hand|right_rubber_hand"
PI_PLUS_CONTACT_BODIES = "l_ankle_roll_link|r_ankle_roll_link|l_wrist_link|r_wrist_link"

G1_LEG_JOINTS = ".*_hip_.*_joint|.*_knee_joint|.*_ankle_.*_joint"
G1_TORSO_JOINTS = "waist_yaw_joint"
G1_ARM_JOINTS = ".*_shoulder_.*_joint|.*_elbow_joint|.*_wrist_.*"

PI_PLUS_LEG_JOINTS = ".*_thigh_joint|.*_hip_roll_joint|.*_hip_pitch_joint|.*_calf_joint|.*_ankle_.*_joint"
PI_PLUS_ARM_JOINTS = ".*_shoulder_.*_joint|.*_upper_arm_joint|.*_elbow_joint|.*_wrist_joint"


def _rough_terrain_importer_cfg() -> TerrainImporterCfg:
    terrain_generator_cfg = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=1.0,
                stone_height_max=0.2,
                stone_width_range=(0.25, 1.575),
                stone_distance_range=(0.05, 0.1),
                holes_depth=-2.0,
                platform_width=1.5,
                border_width=0.0,
            ),
        },
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=terrain_generator_cfg.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


class ActionMod(Enum):
    Median = 0
    Offset = 1
    Residual = 2
    CurrentResidual = 3


@configclass
class G1DomainRandCfg:
    """Structured domain randomization for the G1 robot."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    rand_robot_mass = EventTerm(
        func=randomize_group_body_masses,
        mode="startup",
        params={
            "base_cfg": SceneEntityCfg("robot", body_names="pelvis|torso_link"),
            "base_scale_range": (0.8, 1.2),
            "legs_cfg": SceneEntityCfg("robot", body_names=".*hip.*|.*knee.*|.*ankle.*"),
            "legs_scale_range": (0.9, 1.1),
            "arms_cfg": SceneEntityCfg("robot", body_names=".*shoulder.*|.*elbow.*|.*wrist.*|.*hand.*"),
            "arms_scale_range": (0.9, 1.1),
        },
    )

    rand_base_com = EventTerm(
        func=randomize_rigid_body_com_from_default,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis|torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    rand_contact_offsets = EventTerm(
        func=randomize_rigid_body_collider_offsets_by_body,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=G1_CONTACT_BODIES),
            "rest_offset_range": (0.0, 0.002),
            "contact_offset_range": (0.004, 0.012),
            "min_contact_gap": 5e-4,
        },
    )

    rand_leg_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LEG_JOINTS),
            "friction_distribution_params": (0.0, 0.04),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    rand_arm_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    rand_torso_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_TORSO_JOINTS),
            "friction_distribution_params": (0.0, 0.03),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    rand_leg_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_LEG_JOINTS),
            "armature_distribution_params": (0.92, 1.08),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    rand_arm_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    rand_torso_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=G1_TORSO_JOINTS),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    rand_robot_joint_stiffness_and_damping = EventTerm(
        func=randomize_group_actuator_gains,
        mode="startup",
        params={
            "legs_cfg": SceneEntityCfg("robot", joint_names=G1_LEG_JOINTS),
            "legs_scale_range": (0.8, 1.2),
            "torso_cfg": SceneEntityCfg("robot", joint_names=G1_TORSO_JOINTS),
            "torso_scale_range": (0.8, 1.2),
            "arms_cfg": SceneEntityCfg("robot", joint_names=G1_ARM_JOINTS),
            "arms_scale_range": (0.8, 1.2),
        },
    )

    rand_action_latency = EventTerm(
        func=randomize_action_latency,
        mode="reset",
        params={"latency_range": G1_ACTION_LATENCY_RANGE},
    )


@configclass
class G1TrainingEventCfg(G1DomainRandCfg):
    """Training events for the G1 robot."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": G1_PUSH_VELOCITY_RANGE},
    )

@configclass
class PiPlusDomainRandCfg:
    """Structured domain randomization for the PiPlus robot."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=PI_PLUS_CONTACT_BODIES),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.08),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    rand_robot_mass = EventTerm(
        func=randomize_group_body_masses,
        mode="startup",
        params={
            "base_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "base_scale_range": (0.95, 1.05),
            "legs_cfg": SceneEntityCfg("robot", body_names=".*hip.*|.*thigh.*|.*calf.*|.*ankle.*"),
            "legs_scale_range": (0.9, 1.1),
            "arms_cfg": SceneEntityCfg("robot", body_names=".*shoulder.*|.*upper_arm.*|.*elbow.*|.*wrist.*"),
            "arms_scale_range": (0.9, 1.1),
        },
    )

    rand_base_com = EventTerm(
        func=randomize_rigid_body_com_from_default,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.015, 0.015), "z": (-0.01, 0.01)},
        },
    )

    rand_contact_offsets = EventTerm(
        func=randomize_rigid_body_collider_offsets_by_body,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=PI_PLUS_CONTACT_BODIES),
            "rest_offset_range": (0.0, 0.0015),
            "contact_offset_range": (0.003, 0.01),
            "min_contact_gap": 5e-4,
        },
    )

    rand_leg_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_LEG_JOINTS),
            "friction_distribution_params": (0.0, 0.03),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    rand_arm_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_ARM_JOINTS),
            "friction_distribution_params": (0.0, 0.02),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    rand_leg_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_LEG_JOINTS),
            "armature_distribution_params": (0.94, 1.06),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    rand_arm_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_ARM_JOINTS),
            "armature_distribution_params": (0.92, 1.08),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    rand_robot_joint_stiffness_and_damping = EventTerm(
        func=randomize_group_actuator_gains,
        mode="startup",
        params={
            "legs_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_LEG_JOINTS),
            "legs_scale_range": (0.92, 1.08),
            "arms_cfg": SceneEntityCfg("robot", joint_names=PI_PLUS_ARM_JOINTS),
            "arms_scale_range": (0.92, 1.08),
        },
    )

    rand_action_latency = EventTerm(
        func=randomize_action_latency,
        mode="reset",
        params={"latency_range": PI_PLUS_ACTION_LATENCY_RANGE},
    )


@configclass
class PiPlusTrainingEventCfg(PiPlusDomainRandCfg):
    """Training events for the PiPlus robot."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.5, 4.5),
        params={"velocity_range": PI_PLUS_PUSH_VELOCITY_RANGE},
    )


@configclass
class G1MotionTrackingEnvCfg(DirectRLEnvCfg):
    expert_motion_file = None
    episode_length_s = 10.0

    decimation = 4

    observation_space = 124
    policy_observation_space = 124
    motion_observation_space = 49
    robot_observation_space = 75

    critic_observation_space = 256
    action_space = 23
    state_space = 0

    action_buffer_length = G1_ACTION_LATENCY_RANGE[1] + 1
    action_latency_range = G1_ACTION_LATENCY_RANGE
    action_mod = ActionMod.Median
    action_noise = 0.025

    expert_motion_file = None

    bin_size = 0.3
    failure_decay = 0.99
    failure_weight_min = 0.001
    failure_temperature = 1.0
    sampling_strategy: SamplingStrategy | None = None
    sampler_mod:SamplerMod = SamplerMod.Clamp

    root_link_name = "pelvis"
    anchor_body_name = "pelvis"

    key_body_names = [
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    
    collision_track_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    end_effector_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ]

    foot_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]

    foot_slip_weight = -0.1
    foot_slip_force_threshold = 1.0

    anchor_pos_error_threshold = 0.5
    anchor_ori_error_threshold = 1.5
    end_effector_pos_error_threshold = 0.5
    #termination_curriculum: TerminationCurriculumCfg = TerminationCurriculumCfg()
    probabilistic_error_termination = True
    error_termination_ramp_multiplier = 2.0
    error_termination_sigmoid_steepness = 8.0
    height_only = True
    end_effector_height_only = True

    training = True
    add_obs_noise = True
    add_action_noise = True
    add_reset_noise = True
    random_start = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0
        ),
        debug_vis=False,
    )

    scene:InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    events: G1TrainingEventCfg = G1TrainingEventCfg()

    robot:ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0,
    )

@configclass
class PiPlusMotionTrackingEnvCfg(DirectRLEnvCfg):
    expert_motion_file = None
    episode_length_s = 10.0

    decimation = 4

    observation_space = 124
    policy_observation_space = 124
    critic_observation_space = 256
    action_space = 23
    state_space = 0

    action_buffer_length = PI_PLUS_ACTION_LATENCY_RANGE[1] + 1
    action_latency_range = PI_PLUS_ACTION_LATENCY_RANGE
    action_mod = ActionMod.Median
    action_noise = 0.01

    expert_motion_file = None

    bin_size = 0.2
    failure_decay = 1.0
    failure_weight_min = 0.001
    failure_temperature = 1.0
    sampling_strategy: SamplingStrategy | None = None
    sampler_mod:SamplerMod = SamplerMod.Clamp

    root_link_name = "base_link"
    anchor_body_name = "base_link"

    key_body_names = [
        "base_link",
        "l_hip_roll_link",
        "l_calf_link",
        "l_ankle_roll_link",
        "r_hip_roll_link",
        "r_calf_link",
        "r_ankle_roll_link",
            
        "l_shoulder_roll_link",
        "l_elbow_link",
        "l_wrist_link",
        "r_shoulder_roll_link",
        "r_elbow_link",
        "r_wrist_link",
    ]
    
    collision_track_body_names = [
        "l_ankle_roll_link",
        "r_ankle_roll_link", 
        "l_wrist_link",
        "r_wrist_link",
    ]

    end_effector_body_names = [
        "l_ankle_roll_link",
        "r_ankle_roll_link",
        "l_wrist_link",
        "r_wrist_link",
    ]

    foot_body_names = [
        "l_ankle_roll_link",
        "r_ankle_roll_link",
    ]

    foot_slip_weight = -0.1
    foot_slip_force_threshold = 1.0

    anchor_pos_error_threshold = 0.25
    anchor_ori_error_threshold = 0.8
    end_effector_pos_error_threshold = 0.15
    termination_curriculum: TerminationCurriculumCfg = TerminationCurriculumCfg()
    probabilistic_error_termination = True
    error_termination_ramp_multiplier = 2.0
    error_termination_sigmoid_steepness = 8.0
    height_only = True
    end_effector_height_only = False

    training = True
    add_obs_noise = True
    add_action_noise = True
    add_reset_noise = True
    random_start = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0
        ),
        debug_vis=False,
    )

    scene:InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    events: PiPlusTrainingEventCfg = PiPlusTrainingEventCfg()

    robot:ArticulationCfg = PI_PLUS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0,
    )


@configclass
class G1MotionTrackingRoughEnvCfg(G1MotionTrackingEnvCfg):
    terrain = _rough_terrain_importer_cfg()


@configclass
class PiPlusMotionTrackingRoughEnvCfg(PiPlusMotionTrackingEnvCfg):
    terrain = _rough_terrain_importer_cfg()

@configclass
class MotionViewerCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = G1_CFG.replace(prim_path="/World/Robot")

JOINT_ORDER = [
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

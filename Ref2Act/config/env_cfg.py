from enum import Enum
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
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
from ..sampler import SamplerMod

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

class ActionMod(Enum):
    Median = 0
    Offset = 1

@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    rand_robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"
                ),
                "mass_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "distribution": "uniform"
            },
        )

    rand_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    
    rand_robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=200,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

@configclass
class G1MotionTrackingEnvCfg(DirectRLEnvCfg):
    expert_motion_file = None
    episode_length_s = 10.0

    decimation = 4

    observation_space = 124
    policy_observation_space = 124
    critic_observation_space = 256
    action_space = 23
    state_space = 0

    action_buffer_length = 1
    action_mod = ActionMod.Median
    action_noise = 0.01

    expert_motion_file = None

    bin_size = 0.3
    failure_decay = 0.99
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

    anchor_pos_error_threshold = 0.25
    anchor_ori_error_threshold = 0.8
    end_effector_pos_error_threshold = 0.25
    height_only = True

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

    action_buffer_length = 1
    action_mod = ActionMod.Median
    action_noise = 0.01

    expert_motion_file = None

    bin_size = 0.2
    failure_decay = 1.0
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

    anchor_pos_error_threshold = 0.25
    anchor_ori_error_threshold = 0.8
    end_effector_pos_error_threshold = 0.25
    height_only = True

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

    robot:ArticulationCfg = PI_PLUS_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0,
    )

@configclass
class MotionViewerCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    robot = G1_CFG.replace(prim_path="/World/Robot")

    contact_sensor = ContactSensorCfg(
        prim_path="/World/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0,
    )

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

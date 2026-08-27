import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from ref2act.envs.motion_tracking.action import ActionSpec
from ref2act.envs.motion_tracking.curriculum import TerminationCurriculumCfg
from ref2act.envs.motion_tracking.observation import default_training_observation_spec
from ref2act.envs.motion_tracking.randomization import (
    randomize_group_actuator_gains,
    randomize_group_body_masses,
    randomize_rigid_body_collider_offsets_by_body,
    randomize_rigid_body_com_from_default,
)
from ref2act.envs.motion_tracking.rewards import default_reward_spec
from ref2act.envs.motion_tracking.termination import default_termination_spec
from ref2act.motion.sampling import SamplerMod, SamplingStrategy, SegmentSource
from ref2act.robots._articulation_shared import G1_CFG

G1_ACTION_LATENCY_RANGE = (0, 0)

G1_PUSH_VELOCITY_RANGE = {
    "x": (-0.3, 0.3),
    "y": (-0.3, 0.3),
    "yaw": (-0.4, 0.4),
}

G1_CONTACT_BODIES = (
    "left_ankle_roll_link|right_ankle_roll_link|left_rubber_hand_link|right_rubber_hand_link"
)

G1_LEG_JOINTS = ".*_hip_.*_joint|.*_knee_joint|.*_ankle_.*_joint"
G1_TORSO_JOINTS = "waist_yaw_joint"
G1_ARM_JOINTS = ".*_shoulder_.*_joint|.*_elbow_joint|.*_wrist_.*"


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
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


def _g1_reward_spec():
    return default_reward_spec(dt=0.0, anchor_height_only=False)


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
class G1MotionTrackingEnvCfg(DirectRLEnvCfg):
    expert_motion_file = None
    episode_length_s = 10.0

    decimation = 4

    observation_space = 0
    policy_observation_space = 0
    motion_observation_space = 0
    robot_observation_space = 0
    critic_observation_space = 0
    action_space = 23
    state_space = 0

    observation = default_training_observation_spec(add_noise=True)
    action = ActionSpec(
        mode="median",
        buffer_length=1,
        latency_range=None,
        noise_scale=0.025,
    )

    bin_size = 0.3
    weight_fail = 0.5
    weight_novel = 0.3
    cap_beta = 2.0
    adaptive_uniform_ratio = 0.1
    adaptive_alpha = 0.001
    adaptive_kernel_size = 1
    adaptive_lambda = 0.8
    motion_sampling_warmup_s = 0.0
    motion_sampling_ramp_s = 0.0
    motion_sampling_schedule = "cosine"
    segment_source: SegmentSource = SegmentSource.Time
    sampling_strategy: SamplingStrategy | None = None
    sampler_mod: SamplerMod = SamplerMod.Clamp
    init_failure_bins: bool | None = None
    compact_motion_storage: bool = True

    root_link_name = "pelvis"
    anchor_body_name = "pelvis"

    key_body_names = [
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
    ]

    # Bodies that must not carry contact force.  The ankle/toe support assembly
    # is deliberately excluded so normal foot-ground contact is not penalized.
    # With the all-body contact sensor below, this captures both self-contact
    # and unsafe non-foot terrain contact.
    collision_track_body_names = [
        "pelvis",
        "torso_link",
        ".*_hip_.*_link",
        ".*_knee_link",
        ".*_shoulder_.*_link",
        ".*_elbow_link",
        ".*_wrist_.*|.*_wrist_roll",
        ".*_rubber_hand_link",
    ]

    end_effector_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_rubber_hand_link",
        "right_rubber_hand_link",
    ]

    foot_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]

    rewards = _g1_reward_spec()
    termination = default_termination_spec(
        anchor_height_only=True,
        end_effector_height_only=False,
    )
    termination_curriculum: TerminationCurriculumCfg | None = None

    training = True
    add_reset_noise = True
    random_start = True
    reset_root_height_offset = 0.05

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
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
        physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    events: G1TrainingEventCfg = G1TrainingEventCfg()

    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        class_type="ref2act.nested_contact_sensor:NestedContactSensor",
        # Query only legal support bodies plus bodies whose contact is penalized.
        # Contact reports are still activated recursively by the custom USD
        # spawner, but excluding ankle-pitch/toe bodies from this view avoids
        # transferring force data that neither reward consumes.
        prim_path=(
            "/World/envs/env_.*/Robot/"
            "(pelvis|torso_link|.*_hip_.*_link|.*_knee_link|"
            ".*_shoulder_.*_link|.*_elbow_link|.*_wrist_.*|"
            ".*_rubber_hand_link|.*_ankle_roll_link)"
        ),
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
    )


@configclass
class G1MotionTrackingRoughEnvCfg(G1MotionTrackingEnvCfg):
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

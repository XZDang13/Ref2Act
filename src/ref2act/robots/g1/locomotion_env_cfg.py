from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from ref2act.envs.locomotion.commands import StratifiedVelocityCommandCfg
from ref2act.envs.locomotion.observation import default_locomotion_observation_spec
from ref2act.envs.locomotion.task_rewards import FlatLocomotionRewardCfg
from ref2act.envs.locomotion.terrain import make_locomotion_terrain_cfg
from ref2act.envs.motion_tracking.action import ActionSpec
from ref2act.robots._articulation_shared import G1_CFG
from ref2act.robots._env_cfg_shared import G1DomainRandCfg
from ref2act.robots._g1_spec import G1_23_DOF_SPEC


def _base_height_sensor_cfg() -> RayCasterCfg:
    """Create the local terrain grid used only by generated-terrain rewards."""

    return RayCasterCfg(
        class_type="ref2act.direct_leaf_ray_caster:DirectLeafRayCaster",
        # This sensor is instantiated directly by LeggedRobotEnv rather than
        # through InteractiveSceneCfg, so use the resolved global regex.
        prim_path="/World/envs/env_.*/Robot/Geometry/pelvis",
        spawn=None,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            # A small local grid supports terrain-relative pelvis and swing-foot
            # heights without exposing terrain perception to the policy.
            resolution=0.3,
            size=(0.6, 0.6),
            direction=(0.0, 0.0, -1.0),
        ),
        max_distance=5.0,
        debug_vis=False,
    )


@configclass
class G1FlatLocomotionEnvCfg(DirectRLEnvCfg):
    """G1 23-DoF blind velocity locomotion on a flat plane."""

    robot_spec_name = G1_23_DOF_SPEC.name

    episode_length_s = 20.0
    decimation = 4

    observation_space = 0
    policy_observation_space = 0
    command_observation_space = 0
    robot_observation_space = 0
    critic_observation_space = 0
    action_space = G1_23_DOF_SPEC.action_dim
    state_space = 0

    # Actor input is command/phase + robot history supplied by the downstream
    # PAIR adapter.  Velocity feedback remains available only through history.
    observation = default_locomotion_observation_spec(
        add_noise=True,
        include_gait_phase=True,
        include_velocity_feedback=False,
    )
    action = ActionSpec(
        mode="offset",
        buffer_length=1,
        latency_range=None,
        noise_scale=0.025,
    )
    command = StratifiedVelocityCommandCfg()
    rewards = FlatLocomotionRewardCfg()

    # Flat terrain uses the environment origin as ground height. Generated
    # terrains override this with one downward ray per environment so the
    # reward follows the local slope/uneven surface without entering policy
    # observations.
    base_height_sensor: RayCasterCfg | None = None

    joint_position_reset_noise = 0.05
    minimum_base_height = 0.20
    minimum_upright_projection = math.cos(0.8)
    illegal_contact_force_threshold = 1.0
    unique_terrain_origins = True
    # Generated terrains reserve one additional tile on every outer edge.  In
    # combination with the 10 m importer border this covers the full 20 s
    # command horizon without reintroducing per-patch resets.
    terrain_guard_tiles = 1
    terrain_curriculum = False
    terrain_out_of_bounds_distance = None

    # Legacy task-free environments still use these three selections through
    # LocomotionEnv's compatibility reward path.
    reward_hip_joint_names = [".*_hip_yaw_joint", ".*_hip_roll_joint"]
    reward_arm_joint_names = [
        ".*_shoulder_.*_joint",
        ".*_elbow_joint",
    ]
    reward_torso_joint_names = ["waist_yaw_joint"]
    reward_leg_joint_names = [
        ".*_hip_.*_joint",
        ".*_knee_joint",
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ]
    reward_ankle_joint_names = [
        ".*_ankle_pitch_joint",
        ".*_ankle_roll_joint",
    ]
    # Feet are the only legal support surface for nominal locomotion.
    termination_body_names = [
        "pelvis",
        "torso_link",
        ".*_knee_link",
        ".*_rubber_hand_link",
    ]

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
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )
    # Keep startup domain randomization, but do not add interval pushes until
    # nominal command tracking is established.
    events: G1DomainRandCfg = G1DomainRandCfg()
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Locomotion-only upright reference. Keep the shared motion-tracking G1
    # defaults intact. Height is measured from this USD pose to the sole point.
    robot.init_state.joint_pos.update({
        ".*_hip_pitch_joint": -0.15,
        ".*_knee_joint": 0.35,
        ".*_ankle_pitch_joint": -0.20,
    })
    robot.init_state.pos = (0.0, 0.0, rewards.base_height_target)
    robot.spawn.articulation_props.enabled_self_collisions = True

    contact_sensor = ContactSensorCfg(
        class_type="ref2act.nested_contact_sensor:NestedContactSensor",
        prim_path=(
            "/World/envs/env_.*/Robot/"
            "(pelvis|torso_link|.*_hip_.*_link|.*_knee_link|"
            ".*_shoulder_.*_link|.*_elbow_link|.*_wrist_.*|"
            ".*_rubber_hand_link|.*_ankle_pitch_link|.*_ankle_roll_link)"
        ),
        history_length=3,
        track_air_time=True,
        # Match IsaacLab G1FlatEnvCfg. The backend default is used for
        # air/contact mode tracking; termination has its own explicit 1 N test.
        force_threshold=None,
    )


@configclass
class G1SlopeLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion over a static distribution of slope difficulties."""

    terrain = make_locomotion_terrain_cfg(mode="slope")
    base_height_sensor = _base_height_sensor_cfg()
    terrain_curriculum = False
    terrain_out_of_bounds_distance = None


@configclass
class G1UnevenLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion over a static distribution of uneven terrain."""

    terrain = make_locomotion_terrain_cfg(mode="uneven")
    base_height_sensor = _base_height_sensor_cfg()
    terrain_curriculum = False
    terrain_out_of_bounds_distance = None


@configclass
class G1MixedTerrainLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion over unique flat/slope/uneven patches."""

    terrain = make_locomotion_terrain_cfg(mode="mixed")
    base_height_sensor = _base_height_sensor_cfg()
    terrain_curriculum = False
    terrain_out_of_bounds_distance = None


__all__ = [
    "G1FlatLocomotionEnvCfg",
    "G1MixedTerrainLocomotionEnvCfg",
    "G1SlopeLocomotionEnvCfg",
    "G1UnevenLocomotionEnvCfg",
]

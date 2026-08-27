from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from ref2act.envs.locomotion.commands import VelocityCommandCfg
from ref2act.envs.locomotion.observation import default_locomotion_observation_spec
from ref2act.envs.locomotion.rewards import LocomotionRewardCfg
from ref2act.envs.locomotion.terrain import make_locomotion_terrain_cfg
from ref2act.envs.motion_tracking.action import ActionSpec
from ref2act.robots._articulation_shared import G1_CFG
from ref2act.robots._env_cfg_shared import G1TrainingEventCfg
from ref2act.robots._g1_spec import G1_23_DOF_SPEC


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

    observation = default_locomotion_observation_spec(add_noise=True)
    action = ActionSpec(
        mode="median",
        buffer_length=1,
        latency_range=None,
        noise_scale=0.025,
    )
    command = VelocityCommandCfg()
    rewards = LocomotionRewardCfg()

    joint_position_reset_noise = 0.05
    minimum_base_height = 0.42
    minimum_upright_projection = 0.35
    terrain_curriculum = False
    terrain_out_of_bounds_distance = None

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
    events: G1TrainingEventCfg = G1TrainingEventCfg()
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_sensor = ContactSensorCfg(
        class_type="ref2act.nested_contact_sensor:NestedContactSensor",
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
class G1SlopeLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion on continuous uphill and downhill terrain."""

    terrain = make_locomotion_terrain_cfg(mode="slope")
    terrain_curriculum = True
    terrain_out_of_bounds_distance = 3.25


@configclass
class G1UnevenLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion on smooth uneven terrain."""

    terrain = make_locomotion_terrain_cfg(mode="uneven")
    terrain_curriculum = True
    terrain_out_of_bounds_distance = 3.25


@configclass
class G1MixedTerrainLocomotionEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 blind locomotion on a flat/slope/uneven curriculum."""

    terrain = make_locomotion_terrain_cfg(mode="mixed")
    terrain_curriculum = True
    terrain_out_of_bounds_distance = 3.25


__all__ = [
    "G1FlatLocomotionEnvCfg",
    "G1MixedTerrainLocomotionEnvCfg",
    "G1SlopeLocomotionEnvCfg",
    "G1UnevenLocomotionEnvCfg",
]

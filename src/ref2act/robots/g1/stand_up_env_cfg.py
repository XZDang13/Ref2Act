from __future__ import annotations
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from ref2act.envs.motion_tracking.action import ActionSpec
from ref2act.robots._articulation_shared import G1_CFG
from ref2act.robots._env_cfg_shared import G1DomainRandCfg
from ref2act.robots._g1_spec import G1_23_DOF_SPEC
from ref2act.envs.stand_up.defaults import default_config
from ref2act.envs.stand_up.runtime_config import configure_standup_cfg

@configclass
class G1StandUpEnvCfg(DirectRLEnvCfg):
    """G1 supine-to-standing task with V12 staged support rewards."""
    robot_spec_name = G1_23_DOF_SPEC.name
    cache_static_physics = True
    episode_length_s = 10.0
    decimation = 4
    observation_space = 312
    policy_observation_space = 312
    critic_observation_space = 181
    action_space = G1_23_DOF_SPEC.action_dim
    state_space = 0
    action = ActionSpec(mode="offset", buffer_length=1, latency_range=(0, 0), noise_scale=0.)
    base_height_sensor = None
    minimum_base_height = .20
    minimum_upright_projection = math.cos(.8)
    task = default_config()
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

    def __post_init__(self):
        configure_standup_cfg(self, self.task, device=self.sim.device,
                             num_envs=self.scene.num_envs, seed=self.seed or 0)

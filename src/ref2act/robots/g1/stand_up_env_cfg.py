from __future__ import annotations

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from ref2act.envs.motion_tracking.action import ActionSpec
from ref2act.envs.stand_up.rewards import StandUpRewardCfg
from ref2act.robots._articulation_shared import G1_CFG

from .locomotion_env_cfg import G1FlatLocomotionEnvCfg


@configclass
class G1FlatStandUpEnvCfg(G1FlatLocomotionEnvCfg):
    """G1 stand-up with mostly-supine resets and temporary pelvis assistance."""

    # Discovery is expected to finish the maneuver and hold standing within
    # four seconds. Failed attempts reset promptly instead of spending most of
    # a rollout motionless on the floor.
    episode_length_s = 4.0
    observation_space = {"policy": 78, "critic": 87}
    policy_observation_space = 78
    critic_observation_space = 87

    action = ActionSpec(
        mode="offset",
        buffer_length=1,
        latency_range=None,
        noise_scale=0.0,
    )
    rewards = StandUpRewardCfg()
    events = None

    # V1 intentionally contains no dynamics or pose randomization. Ten percent
    # of episodes start from the nominal standing pose to teach the hold phase.
    # The GPU smoke measured the settled horizontal pelvis at about 0.248 m.
    # Spawn directly at that contact height so PPO never records transitions
    # for actions that an in-episode settling mask would ignore.
    initial_root_position = (0.0, 0.0, 0.25)
    initial_root_euler_xyz = (0.0, math.pi / 2.0, 0.0)
    settling_steps = 0
    standing_reset_fraction = 0.10
    rise_reference_root_height = 0.25
    rise_reference_shoulder_height = 0.20
    rise_reference_upright_projection = 0.0

    # Assistance is sampled once per supine attempt and follows an exogenous
    # time profile. It never depends on a state the policy can manipulate.
    assistance_body_name = "pelvis"
    assistance_initial_max_gravity_ratio = 0.35
    assistance_zero_probability = 0.20
    assistance_minimum_force_fraction = 0.50
    assistance_full_duration_s = 1.50
    assistance_fade_duration_s = 1.00

    target_root_height = 0.74
    target_shoulder_height = 1.16
    success_height_ratio = 0.95
    success_upright_projection = 0.90
    success_linear_velocity = 0.30
    success_angular_velocity = 0.60
    success_joint_velocity = 2.0
    success_hold_steps = 25

    support_force_threshold = 10.0
    success_total_foot_load_ratio = 0.70
    success_each_foot_load_ratio = 0.10
    support_force_observation_scale = 400.0
    maximum_linear_velocity = 20.0
    maximum_angular_velocity = 30.0
    maximum_joint_velocity = 60.0
    non_foot_support_body_names = [
        "pelvis",
        "torso_link",
        ".*_hip_.*_link",
        ".*_knee_link",
        ".*_shoulder_.*_link",
        ".*_elbow_link",
        ".*_wrist_.*|.*_wrist_roll",
        ".*_rubber_hand_link",
    ]

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
        force_threshold=None,
    )


__all__ = ["G1FlatStandUpEnvCfg"]

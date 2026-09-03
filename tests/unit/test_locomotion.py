from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import ref2act  # noqa: F401
from ref2act.common.observation_spec import ObservationLayout
from ref2act.envs.locomotion.commands import UniformVelocityCommandGenerator, VelocityCommandCfg
from ref2act.envs.locomotion.commands import (
    LOCOMOTION_COMMAND_CATEGORIES,
    StratifiedVelocityCommandCfg,
    StratifiedVelocityCommandGenerator,
)
from ref2act.envs.base import LeggedRobotEnv
from ref2act.envs.locomotion.env import LocomotionEnv, compute_locomotion_termination
from ref2act.envs.locomotion.observation import (
    compute_locomotion_velocity_feedback,
    default_locomotion_observation_spec,
)
from ref2act.envs.locomotion.rewards import (
    LocomotionRewardCfg,
    LocomotionRewardInputs,
    compute_feet_air_time_reward,
    compute_feet_air_time_positive_biped_reward,
    compute_feet_phase_reward,
    compute_feet_gait_reward,
    compute_foot_clearance_reward,
    compute_locomotion_gait_phase,
    compute_locomotion_phase_features,
    compute_locomotion_reward_terms,
    expected_foot_height_from_phase,
)
from ref2act.envs.locomotion.task_rewards import (
    FlatLocomotionRewardCfg,
    FlatLocomotionRewardInputs,
    compute_flat_locomotion_reward_terms,
    phase_gait_targets,
)
from ref2act.envs.locomotion.terrain import (
    Ref2ActSlopeTerrainCfg,
    Ref2ActUnevenTerrainCfg,
    make_locomotion_terrain_cfg,
    slope_terrain,
    uneven_terrain,
)
from ref2act.robots.g1 import (
    G1FlatLocomotionEnvCfg,
    G1MixedTerrainLocomotionEnvCfg,
    G1SlopeLocomotionEnvCfg,
    G1UnevenLocomotionEnvCfg,
)


def test_locomotion_observation_contract_matches_shared_g1_proprioception() -> None:
    description = default_locomotion_observation_spec(add_noise=False).describe(
        ObservationLayout(joint_dim=23, action_dim=23, key_body_count=0, command_dim=3)
    )
    assert description.group_dims == {
        "command": 7,
        "feedback": 3,
        "robot": 78,
        "privilege": 88,
    }
    collection_description = default_locomotion_observation_spec(
        add_noise=False,
        include_gait_phase=False,
        include_velocity_feedback=False,
    ).describe(
        ObservationLayout(joint_dim=23, action_dim=23, key_body_count=0, command_dim=3)
    )
    assert collection_description.group_dims == {
        "command": 3,
        "robot": 78,
        "privilege": 84,
    }


def test_locomotion_velocity_feedback_matches_commanded_axes() -> None:
    half_angle = torch.tensor(torch.pi / 4.0)
    feedback = compute_locomotion_velocity_feedback(
        torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, torch.sin(half_angle), torch.cos(half_angle)]]
        ),
        torch.tensor([[0.4, -0.2, 3.0], [0.0, 1.0, -4.0]]),
        torch.tensor([[1.0, 2.0, -0.3], [0.0, 0.0, 2.0]]),
    )
    torch.testing.assert_close(
        feedback,
        torch.tensor([[0.4, -0.2, -0.3], [1.0, 0.0, 2.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_velocity_command_generator_samples_in_range_and_resamples() -> None:
    torch.manual_seed(7)
    cfg = VelocityCommandCfg(
        linear_x_range=(-1.0, 1.0),
        linear_y_range=(-0.5, 0.5),
        yaw_rate_range=(-0.25, 0.25),
        linear_x_limit=(-1.0, 1.0),
        linear_y_limit=(-0.5, 0.5),
        yaw_rate_limit=(-0.25, 0.25),
        resampling_time_range_s=(0.02, 0.02),
        standing_fraction=0.0,
    )
    generator = UniformVelocityCommandGenerator(cfg=cfg, num_envs=8, step_dt=0.02, device="cpu")
    generator.reset()
    first = generator.commands.clone()
    assert torch.all(first[:, 0].abs() <= 1.0)
    assert torch.all(first[:, 1].abs() <= 0.5)
    assert torch.all(first[:, 2].abs() <= 0.25)
    generator.step()
    assert not torch.equal(generator.commands, first)


def test_velocity_command_curriculum_expands_only_successful_axes() -> None:
    generator = UniformVelocityCommandGenerator(
        cfg=VelocityCommandCfg(), num_envs=2, step_dt=0.02, device="cpu"
    )
    changed = generator.update_curriculum(linear_score=0.81, yaw_score=0.79)
    assert changed
    assert generator.current_linear_x_range == (-0.2, 0.2)
    assert generator.current_linear_y_range == (-0.2, 0.2)
    assert generator.current_yaw_rate_range == (-0.1, 0.1)


def test_stratified_commands_have_explicit_modes_and_no_tiny_motion() -> None:
    torch.manual_seed(7)
    generator = StratifiedVelocityCommandGenerator(
        cfg=StratifiedVelocityCommandCfg(),
        num_envs=20000,
        step_dt=0.02,
        device="cpu",
    )
    generator.reset()
    assert generator.category_names == LOCOMOTION_COMMAND_CATEGORIES
    moving = generator.category_ids != 0
    assert torch.all(torch.linalg.vector_norm(generator.commands[moving], dim=-1) >= 0.15)
    fractions = torch.bincount(generator.category_ids, minlength=5).float() / 20000
    torch.testing.assert_close(
        fractions,
        torch.tensor(generator.cfg.category_fractions),
        atol=0.015,
        rtol=0.0,
    )


def test_flat_tracking_reward_is_positive_and_category_normalized() -> None:
    batch = 4
    command = torch.tensor(
        [
            [0.6, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.0, 0.0, 0.5],
            [0.6, 0.3, 0.5],
        ]
    )
    zeros3 = torch.zeros(batch, 3)
    common = dict(
        commands=command,
        base_linear_velocity_b=zeros3,
        base_angular_velocity_b=zeros3,
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        base_height=torch.full((batch,), 0.76),
        gait_phase=torch.full((batch, 2), torch.pi),
        feet_height=torch.zeros(batch, 2),
        feet_contact=torch.ones(batch, 2, dtype=torch.bool),
        joint_acc=torch.zeros(batch, 12),
        applied_torque=torch.zeros(batch, 12),
        action=torch.zeros(batch, 23),
        previous_action=torch.zeros(batch, 23),
        terminated=torch.zeros(batch, dtype=torch.bool),
        feet_air_time=torch.zeros(batch),
        feet_slide=torch.zeros(batch),
        dof_pos_limits=torch.zeros(batch),
    )
    stationary = compute_flat_locomotion_reward_terms(
        FlatLocomotionRewardInputs(
            **common,
            base_linear_velocity_yaw_frame=zeros3,
        ),
        FlatLocomotionRewardCfg(),
    )
    matching = compute_flat_locomotion_reward_terms(
        FlatLocomotionRewardInputs(
            **{
                **common,
                "base_linear_velocity_yaw_frame": torch.tensor(
                    [
                        [0.6, 0.0, 0.0],
                        [0.0, 0.3, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.6, 0.3, 0.0],
                    ]
                ),
                "base_angular_velocity_b": torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.5],
                        [0.0, 0.0, 0.5],
                    ]
                ),
            }
        ),
        FlatLocomotionRewardCfg(),
    )
    assert torch.all(stationary["track_command_exp"] > 0.0)
    assert torch.all(matching["track_command_exp"] > stationary["track_command_exp"])
    torch.testing.assert_close(
        matching["track_command_exp"],
        torch.full((batch,), FlatLocomotionRewardCfg().track_command_exp),
    )
    torch.testing.assert_close(matching["stand_still_exp"], torch.zeros(batch))


def test_flat_tracking_uses_explicit_stand_reward_and_penalizes_inactive_drift() -> None:
    batch = 2
    zeros3 = torch.zeros(batch, 3)
    common = dict(
        commands=torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        base_linear_velocity_b=zeros3,
        base_angular_velocity_b=zeros3,
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        base_height=torch.full((batch,), 0.76),
        gait_phase=torch.full((batch, 2), torch.pi),
        feet_height=torch.zeros(batch, 2),
        feet_contact=torch.ones(batch, 2, dtype=torch.bool),
        joint_acc=torch.zeros(batch, 12),
        applied_torque=torch.zeros(batch, 12),
        action=torch.zeros(batch, 23),
        previous_action=torch.zeros(batch, 23),
        terminated=torch.zeros(batch, dtype=torch.bool),
        feet_air_time=torch.zeros(batch),
        feet_slide=torch.zeros(batch),
        dof_pos_limits=torch.zeros(batch),
    )
    terms = compute_flat_locomotion_reward_terms(
        FlatLocomotionRewardInputs(
            **common,
            base_linear_velocity_yaw_frame=torch.tensor(
                [[0.0, 0.0, 0.0], [0.6, 0.3, 0.0]]
            ),
        ),
        FlatLocomotionRewardCfg(),
    )
    assert terms["stand_still_exp"][0] == pytest.approx(2.0)
    assert terms["stand_still_exp"][1] == pytest.approx(0.0)
    assert terms["inactive_command_axes"][0] == pytest.approx(0.0)
    assert terms["inactive_command_axes"][1] == pytest.approx(
        FlatLocomotionRewardCfg().inactive_command_axes
        * (1.0 - np.exp(-1.0))
        / 2.0
    )


def test_flat_phase_targets_encode_alternating_support_without_target_flight() -> None:
    phase = torch.tensor(
        [[0.0, -torch.pi], [-torch.pi, 0.0], [torch.pi, torch.pi]]
    )
    height, contact = phase_gait_targets(
        phase, stance_ratio=0.55, swing_height=0.09
    )
    torch.testing.assert_close(
        height,
        torch.tensor([[0.09, 0.0], [0.0, 0.09], [0.0, 0.0]]),
    )
    assert torch.equal(
        contact,
        torch.tensor([[False, True], [True, False], [True, True]]),
    )
    assert torch.all(contact.any(dim=-1))


def test_flat_gait_reward_prefers_phase_match_and_penalizes_moving_flight() -> None:
    cfg = FlatLocomotionRewardCfg()
    batch = 3
    inputs = FlatLocomotionRewardInputs(
        commands=torch.tensor(
            [[0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]
        ),
        base_linear_velocity_b=torch.zeros(batch, 3),
        base_angular_velocity_b=torch.zeros(batch, 3),
        base_linear_velocity_yaw_frame=torch.tensor(
            [[0.6, 0.0, 0.0], [0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]
        ),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        base_height=torch.tensor([0.76, 0.66, 0.76]),
        gait_phase=torch.tensor(
            [[0.0, -torch.pi], [0.0, -torch.pi], [0.0, -torch.pi]]
        ),
        feet_height=torch.tensor([[0.09, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        feet_contact=torch.tensor(
            [[False, True], [True, True], [False, False]]
        ),
        joint_acc=torch.zeros(batch, 12),
        applied_torque=torch.zeros(batch, 12),
        action=torch.zeros(batch, 23),
        previous_action=torch.zeros(batch, 23),
        terminated=torch.zeros(batch, dtype=torch.bool),
        feet_air_time=torch.zeros(batch),
        feet_slide=torch.zeros(batch),
        dof_pos_limits=torch.zeros(batch),
    )
    terms = compute_flat_locomotion_reward_terms(inputs, cfg)
    assert terms["swing_contact_penalty"][0] == pytest.approx(0.0)
    assert terms["swing_contact_penalty"][1] == pytest.approx(
        cfg.swing_contact_penalty
    )
    assert terms["unexpected_double_support_penalty"][1] == pytest.approx(
        cfg.unexpected_double_support_penalty
    )
    assert terms["stance_missing_contact_penalty"][2] == pytest.approx(
        cfg.stance_missing_contact_penalty
    )
    assert terms["swing_foot_under_clearance_penalty"][0] == pytest.approx(0.0)
    assert terms["swing_foot_under_clearance_penalty"][1] == pytest.approx(
        cfg.swing_foot_under_clearance_penalty
    )
    assert terms["moving_flight"][0] == pytest.approx(0.0)
    assert terms["moving_flight"][2] == pytest.approx(cfg.moving_flight)
    assert terms["base_height_l2"][0] == pytest.approx(0.0)
    assert terms["base_height_l2"][1] == pytest.approx(-0.05)


def _reward_inputs(*, command_error: float) -> LocomotionRewardInputs:
    batch = 3
    joints = 23
    commands = torch.zeros(batch, 3)
    commands[:, 0] = command_error
    return LocomotionRewardInputs(
        commands=commands,
        base_linear_velocity_b=torch.zeros(batch, 3),
        base_angular_velocity_b=torch.zeros(batch, 3),
        base_linear_velocity_yaw_frame=torch.zeros(batch, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        base_height=torch.full((batch,), 0.76),
        joint_velocity=torch.zeros(batch, joints),
        joint_acc=torch.zeros(batch, joints),
        applied_torque=torch.zeros(batch, joints),
        applied_action=torch.zeros(batch, joints),
        previous_applied_action=torch.zeros(batch, joints),
        terminated=torch.zeros(batch, dtype=torch.bool),
        feet_air_time=torch.zeros(batch),
        feet_phase=torch.ones(batch),
        pose=torch.zeros(batch),
        close_feet_xy=torch.zeros(batch),
        feet_orientation=torch.zeros(batch),
        gait=torch.zeros(batch),
        feet_clearance=torch.ones(batch),
        feet_slide=torch.zeros(batch),
        undesired_contacts=torch.zeros(batch),
        dof_pos_limits=torch.zeros(batch),
        joint_deviation_hip=torch.zeros(batch),
        joint_deviation_arms=torch.zeros(batch),
        joint_deviation_torso=torch.zeros(batch),
    )


def test_locomotion_reward_prefers_matching_velocity() -> None:
    cfg = LocomotionRewardCfg()
    matching = compute_locomotion_reward_terms(_reward_inputs(command_error=0.0), cfg)
    mismatching = compute_locomotion_reward_terms(_reward_inputs(command_error=1.0), cfg)
    assert torch.all(matching["track_lin_vel_xy_exp"] > mismatching["track_lin_vel_xy_exp"])
    assert torch.allclose(matching["flat_orientation_l2"], torch.zeros(3))


def test_locomotion_reward_tightens_lateral_but_keeps_holosoma_yaw_basin() -> None:
    cfg = LocomotionRewardCfg()
    inputs = _reward_inputs(command_error=0.0)
    lateral_commands = inputs.commands.clone()
    lateral_commands[:, 1] = 0.3
    lateral = compute_locomotion_reward_terms(
        LocomotionRewardInputs(**{**inputs.__dict__, "commands": lateral_commands}), cfg
    )
    yaw_commands = inputs.commands.clone()
    yaw_commands[:, 2] = 0.2
    yaw = compute_locomotion_reward_terms(
        LocomotionRewardInputs(**{**inputs.__dict__, "commands": yaw_commands}), cfg
    )
    assert torch.all(lateral["track_lin_vel_xy_exp"] < 0.25)
    expected_yaw = cfg.track_ang_vel_z_exp * torch.exp(torch.tensor(-(0.2**2) / 0.25))
    torch.testing.assert_close(
        yaw["track_ang_vel_z_exp"], expected_yaw.expand_as(yaw["track_ang_vel_z_exp"])
    )


def test_feet_air_time_rewards_only_bounded_completed_flights() -> None:
    reward = compute_feet_air_time_reward(
        torch.tensor([[0.10, 0.30], [0.80, 0.40]]),
        torch.tensor([[True, True], [True, False]]),
        threshold=0.15,
        maximum=0.50,
    )
    assert torch.allclose(reward, torch.tensor([0.15, 0.35]))


def test_feet_air_time_positive_biped_matches_isaaclab_definition() -> None:
    reward = compute_feet_air_time_positive_biped_reward(
        current_air_time=torch.tensor([[0.0, 0.3], [0.0, 0.0], [0.2, 0.4]]),
        current_contact_time=torch.tensor([[0.2, 0.0], [0.2, 0.3], [0.0, 0.0]]),
        commands=torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        threshold=0.4,
    )
    assert torch.allclose(reward, torch.tensor([0.2, 0.0, 0.0]))


def test_locomotion_reward_includes_feet_air_time_term() -> None:
    inputs = _reward_inputs(command_error=0.0)
    inputs = LocomotionRewardInputs(
        **{**inputs.__dict__, "feet_air_time": torch.full((3,), 0.25)}
    )
    terms = compute_locomotion_reward_terms(inputs, LocomotionRewardCfg(feet_air_time=2.0))
    assert torch.allclose(terms["feet_air_time"], torch.full((3,), 0.5))


def test_feet_gait_rewards_alternating_contacts_and_yaw_commands() -> None:
    reward = compute_feet_gait_reward(
        current_contact_time=torch.tensor([[0.2, 0.0], [0.0, 0.2], [0.2, 0.0]]),
        commands=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]),
        episode_step=torch.tensor([4, 24, 0]),
        step_dt=0.02,
        period=0.8,
        offsets=(0.0, 0.5),
        stance_threshold=0.55,
    )
    assert torch.allclose(reward, torch.tensor([2.0, 2.0, 0.0]))


def test_observable_locomotion_phase_is_alternating_and_stands_grounded() -> None:
    phase = compute_locomotion_gait_phase(
        episode_step=torch.tensor([0, 25, 0]),
        phase_offset=torch.zeros(3),
        commands=torch.tensor(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        step_dt=0.02,
        period=1.0,
        offsets=(0.0, 0.5),
    )
    assert torch.allclose(
        torch.cos(phase[:2, 1] - phase[:2, 0]),
        -torch.ones(2),
        atol=1e-6,
    )
    assert torch.allclose(phase[2], torch.full((2,), torch.pi))
    features = compute_locomotion_phase_features(phase)
    assert features.shape == (3, 4)
    assert torch.allclose(features[2], torch.tensor([0.0, -1.0, 0.0, -1.0]), atol=1e-6)


def test_feet_phase_reward_tracks_cubic_bezier_height_target() -> None:
    phase = torch.tensor([[-torch.pi, 0.0], [0.0, -torch.pi]])
    expected = expected_foot_height_from_phase(phase, swing_height=0.09)
    assert torch.allclose(expected, torch.tensor([[0.0, 0.09], [0.09, 0.0]]))
    target = expected
    reward = compute_feet_phase_reward(
        torch.cat((target[:1], torch.zeros((1, 2))), dim=0),
        phase,
        stance_height=0.0,
        swing_height=0.09,
        tracking_sigma=0.008,
    )
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] < reward[0]


def test_foot_clearance_only_shapes_horizontally_moving_feet() -> None:
    reward = compute_foot_clearance_reward(
        foot_clearance=torch.tensor([[0.0, 0.1], [0.0, 0.1]]),
        foot_velocity_xy=torch.tensor(
            [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]]
        ),
        target_height=0.1,
        std=0.05,
        tanh_mult=2.0,
    )
    assert reward[0] == pytest.approx(1.0)
    assert reward[1] < reward[0]


def test_locomotion_reward_penalizes_base_height_error() -> None:
    inputs = _reward_inputs(command_error=0.0)
    inputs = LocomotionRewardInputs(
        **{**inputs.__dict__, "base_height": torch.tensor([0.76, 0.56, 0.96])}
    )
    terms = compute_locomotion_reward_terms(
        inputs,
        LocomotionRewardCfg(base_height_l2=-10.0, base_height_target=0.76),
    )
    assert torch.allclose(terms["base_height_l2"], torch.tensor([0.0, -0.4, -0.4]))


def test_g1_flat_locomotion_config_and_registry_contract() -> None:
    cfg = G1FlatLocomotionEnvCfg()
    assert cfg.action_space == 23
    assert cfg.robot_spec_name == "g1_23dof"
    assert cfg.terrain.terrain_type == "plane"
    assert not cfg.robot.spawn.articulation_props.enabled_self_collisions
    assert ".*_knee_link" in cfg.contact_sensor.prim_path
    assert ".*_ankle_roll_link" in cfg.contact_sensor.prim_path
    assert cfg.contact_sensor.force_threshold is None
    assert cfg.illegal_contact_force_threshold == 1.0
    assert cfg.action.mode == "offset"
    assert cfg.command.linear_x_range == (-0.4, 1.0)
    assert cfg.command.linear_y_range == (-0.3, 0.3)
    assert cfg.command.yaw_rate_range == (-0.5, 0.5)
    assert cfg.command.linear_x_limit == (-0.4, 1.0)
    assert not cfg.command.curriculum_enabled
    assert cfg.command.resampling_time_range_s == (10.0, 10.0)
    assert isinstance(cfg.rewards, FlatLocomotionRewardCfg)
    assert cfg.rewards.termination_penalty == -200.0
    assert cfg.rewards.track_command_exp == 2.0
    assert cfg.rewards.stand_still_exp == 2.0
    assert cfg.rewards.inactive_command_axes == -0.3
    assert cfg.rewards.swing_contact_penalty == -0.25
    assert cfg.rewards.stance_missing_contact_penalty == -0.25
    assert cfg.rewards.unexpected_double_support_penalty == -0.25
    assert cfg.rewards.swing_foot_under_clearance_penalty == -0.25
    assert cfg.rewards.moving_flight == -0.5
    assert cfg.rewards.base_height_l2 == -5.0
    assert cfg.rewards.gait_stance_ratio == pytest.approx(0.55)
    assert cfg.rewards.feet_air_time == 0.5
    assert cfg.rewards.linear_velocity_scales == pytest.approx((0.5, 0.3))
    assert cfg.rewards.yaw_rate_scale == pytest.approx(0.5)
    assert cfg.rewards.action_rate_l2 == -0.002
    assert cfg.rewards.ang_vel_xy_l2 == -0.02
    assert cfg.termination_body_names == [
        "pelvis", "torso_link", ".*_knee_link", ".*_rubber_hand_link"
    ]
    description = cfg.observation.describe(
        ObservationLayout(joint_dim=23, action_dim=23, key_body_count=0, command_dim=3)
    )
    assert description.group_dims == {"command": 7, "robot": 78, "privilege": 88}
    assert cfg.minimum_base_height == 0.2
    assert cfg.minimum_upright_projection == pytest.approx(np.cos(0.8))
    spec = gym.spec("G1FlatLocomotion-v0")
    assert spec.entry_point == "ref2act.envs.locomotion.env:LocomotionEnv"


def test_locomotion_termination_uses_terrain_relative_height() -> None:
    fallen, tilted, terminated = compute_locomotion_termination(
        terrain_relative_base_height=torch.tensor([0.19, 0.80, 0.80]),
        upright_projection=torch.tensor([1.0, 0.60, 1.0]),
        illegal_contact=torch.tensor([False, False, True]),
        minimum_base_height=0.20,
        minimum_upright_projection=float(np.cos(0.8)),
    )
    assert torch.equal(fallen, torch.tensor([True, False, False]))
    assert torch.equal(tilted, torch.tensor([False, True, False]))
    assert torch.equal(terminated, torch.tensor([True, True, True]))


def test_locomotion_scene_explicitly_filters_physx_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class _Scene:
        physics_backend = "physx"
        cfg = type("SceneCfg", (), {"filter_collisions": True})()

        @staticmethod
        def filter_collisions(*, global_prim_paths: list[str]) -> None:
            calls.append(global_prim_paths)

    monkeypatch.setattr(LeggedRobotEnv, "_setup_scene", lambda self: None)
    env = object.__new__(LocomotionEnv)
    env._is_closed = True
    env.scene = _Scene()
    env.cfg = type(
        "EnvCfg",
        (),
        {"terrain": type("TerrainCfg", (), {"prim_path": "/World/ground"})()},
    )()

    LocomotionEnv._setup_scene(env)

    assert calls == [["/World/ground"]]


def test_locomotion_scene_does_not_apply_physx_filter_to_newton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Scene:
        physics_backend = "newton"
        cfg = type("SceneCfg", (), {"filter_collisions": True})()

        @staticmethod
        def filter_collisions(*, global_prim_paths: list[str]) -> None:
            raise AssertionError(f"unexpected PhysX filter: {global_prim_paths}")

    monkeypatch.setattr(LeggedRobotEnv, "_setup_scene", lambda self: None)
    env = object.__new__(LocomotionEnv)
    env._is_closed = True
    env.scene = _Scene()
    env.cfg = type(
        "EnvCfg",
        (),
        {"terrain": type("TerrainCfg", (), {"prim_path": "/World/ground"})()},
    )()

    LocomotionEnv._setup_scene(env)


def test_generated_locomotion_assigns_one_unique_origin_per_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_envs = 70
    terrain_origins = torch.arange(4 * 66 * 3, dtype=torch.float32).reshape(4, 66, 3)
    calls: list[list[str]] = []

    class _Scene:
        physics_backend = "physx"
        cfg = type("SceneCfg", (), {"filter_collisions": True, "num_envs": num_envs})()

        @staticmethod
        def filter_collisions(*, global_prim_paths: list[str]) -> None:
            calls.append(global_prim_paths)

    generator = type("GeneratorCfg", (), {"num_rows": 8, "num_cols": 20})()
    terrain_cfg = type(
        "TerrainCfg",
        (),
        {"prim_path": "/World/ground", "terrain_generator": generator},
    )()
    monkeypatch.setattr(LeggedRobotEnv, "_setup_scene", lambda self: None)
    env = object.__new__(LocomotionEnv)
    env._is_closed = True
    env.scene = _Scene()
    env.sim = type("Sim", (), {"device": "cpu"})()
    env.cfg = type(
        "EnvCfg",
        (),
        {
            "terrain": terrain_cfg,
            "unique_terrain_origins": True,
            "terrain_guard_tiles": 1,
        },
    )()
    env.terrain = type("Terrain", (), {"terrain_origins": terrain_origins})()

    LocomotionEnv._setup_scene(env)

    assert generator.num_rows == 4
    assert generator.num_cols == 66
    assert torch.equal(
        env.terrain.terrain_levels,
        torch.cat((torch.ones(64), torch.full((6,), 2))).long(),
    )
    assert torch.equal(
        env.terrain.terrain_types,
        torch.cat((torch.arange(1, 65), torch.arange(1, 7))),
    )
    assert torch.unique(env.terrain.env_origins, dim=0).shape[0] == num_envs
    assert calls == [["/World/ground"]]


def test_ref2act_slope_is_continuous_and_supports_both_directions() -> None:
    uphill_cfg = Ref2ActSlopeTerrainCfg(size=(8.0, 8.0), slope_range_deg=(12.0, 12.0), inverted=True)
    downhill_cfg = Ref2ActSlopeTerrainCfg(size=(8.0, 8.0), slope_range_deg=(12.0, 12.0), inverted=False)
    uphill_meshes, uphill_origin = slope_terrain(1.0, uphill_cfg)
    downhill_meshes, downhill_origin = slope_terrain(1.0, downhill_cfg)

    uphill_vertices = uphill_meshes[0].vertices
    downhill_vertices = downhill_meshes[0].vertices
    edge_mask = (
        np.isclose(downhill_vertices[:, 0], 0.0)
        | np.isclose(downhill_vertices[:, 0], 8.0)
        | np.isclose(downhill_vertices[:, 1], 0.0)
        | np.isclose(downhill_vertices[:, 1], 8.0)
    )
    assert np.allclose(downhill_vertices[edge_mask, 2], 0.0)
    assert np.allclose(uphill_vertices[edge_mask, 2], 0.0)
    assert downhill_origin[2] > 0.0
    assert uphill_origin[2] < 0.0
    assert downhill_meshes[0].face_normals[:, 2].min() > 0.9
    assert uphill_meshes[0].face_normals[:, 2].min() > 0.9


def test_ref2act_uneven_is_bounded_reproducible_and_edge_continuous() -> None:
    cfg = Ref2ActUnevenTerrainCfg(
        size=(8.0, 8.0),
        amplitude_range=(0.06, 0.06),
        resolution=0.2,
    )
    first_meshes, first_origin = uneven_terrain(1.0, cfg)
    second_meshes, second_origin = uneven_terrain(1.0, cfg)
    vertices = first_meshes[0].vertices
    edge_mask = (
        np.isclose(vertices[:, 0], 0.0)
        | np.isclose(vertices[:, 0], 8.0)
        | np.isclose(vertices[:, 1], 0.0)
        | np.isclose(vertices[:, 1], 8.0)
    )
    assert np.max(np.abs(vertices[:, 2])) <= 0.06 + 1.0e-9
    assert np.allclose(vertices[edge_mask, 2], 0.0)
    assert np.array_equal(first_meshes[0].vertices, second_meshes[0].vertices)
    assert np.array_equal(first_origin, second_origin)


def test_locomotion_terrain_modes_and_registrations() -> None:
    mixed = make_locomotion_terrain_cfg(mode="mixed")
    proportions = {
        name: cfg.proportion for name, cfg in mixed.terrain_generator.sub_terrains.items()
    }
    assert proportions == {
        "flat": 0.4,
        "uphill": 0.15,
        "downhill": 0.15,
        "uneven": 0.3,
    }
    assert mixed.terrain_generator.curriculum
    assert mixed.max_init_terrain_level == 1

    configs_and_ids = (
        (G1SlopeLocomotionEnvCfg(), "G1SlopeLocomotion-v0"),
        (G1UnevenLocomotionEnvCfg(), "G1UnevenLocomotion-v0"),
        (G1MixedTerrainLocomotionEnvCfg(), "G1MixedTerrainLocomotion-v0"),
    )
    for cfg, env_id in configs_and_ids:
        assert cfg.terrain.terrain_type == "generator"
        assert cfg.unique_terrain_origins
        assert cfg.terrain_guard_tiles == 1
        assert not cfg.terrain_curriculum
        assert cfg.terrain_out_of_bounds_distance is None
        assert gym.spec(env_id).entry_point == "ref2act.envs.locomotion.env:LocomotionEnv"

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import ref2act  # noqa: F401
from ref2act.common.observation_spec import ObservationLayout
from ref2act.envs.locomotion.commands import UniformVelocityCommandGenerator, VelocityCommandCfg
from ref2act.envs.base import LeggedRobotEnv
from ref2act.envs.locomotion.env import LocomotionEnv, compute_locomotion_termination
from ref2act.envs.locomotion.observation import default_locomotion_observation_spec
from ref2act.envs.locomotion.rewards import (
    LocomotionRewardCfg,
    LocomotionRewardInputs,
    compute_feet_air_time_reward,
    compute_feet_air_time_positive_biped_reward,
    compute_feet_gait_reward,
    compute_foot_clearance_reward,
    compute_locomotion_reward_terms,
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
        "command": 3,
        "robot": 78,
        "privilege": 84,
    }


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
    assert cfg.command.linear_x_range == (-0.1, 0.1)
    assert cfg.command.linear_y_range == (-0.1, 0.1)
    assert cfg.command.yaw_rate_range == (-0.1, 0.1)
    assert cfg.command.linear_x_limit == (-0.5, 1.0)
    assert cfg.command.curriculum_enabled
    assert cfg.command.resampling_time_range_s == (10.0, 10.0)
    assert cfg.rewards.termination_penalty == 0.0
    assert cfg.rewards.track_ang_vel_z_exp == 0.5
    assert cfg.rewards.alive == 0.15
    assert cfg.rewards.base_height_l2 == -10.0
    assert cfg.rewards.base_height_target == 0.76
    assert cfg.rewards.feet_air_time == 0.0
    assert cfg.rewards.gait == 0.5
    assert cfg.rewards.feet_clearance == 1.0
    assert cfg.rewards.undesired_contacts == -1.0
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
    terrain_origins = torch.arange(2 * 64 * 3, dtype=torch.float32).reshape(2, 64, 3)

    class _Scene:
        physics_backend = "physx"
        cfg = type("SceneCfg", (), {"filter_collisions": True, "num_envs": num_envs})()

        @staticmethod
        def filter_collisions(*, global_prim_paths: list[str]) -> None:
            raise AssertionError(f"unique origins must not need collision groups: {global_prim_paths}")

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
        {"terrain": terrain_cfg, "unique_terrain_origins": True},
    )()
    env.terrain = type("Terrain", (), {"terrain_origins": terrain_origins})()

    LocomotionEnv._setup_scene(env)

    assert generator.num_rows == 2
    assert generator.num_cols == 64
    assert torch.equal(env.terrain.terrain_levels, torch.cat((torch.zeros(64), torch.ones(6))).long())
    assert torch.equal(env.terrain.terrain_types, torch.cat((torch.arange(64), torch.arange(6))))
    assert torch.unique(env.terrain.env_origins, dim=0).shape[0] == num_envs


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
        assert not cfg.terrain_curriculum
        assert gym.spec(env_id).entry_point == "ref2act.envs.locomotion.env:LocomotionEnv"

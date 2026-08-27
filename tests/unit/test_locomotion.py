from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import ref2act  # noqa: F401
from ref2act.common.observation_spec import ObservationLayout
from ref2act.envs.locomotion.commands import UniformVelocityCommandGenerator, VelocityCommandCfg
from ref2act.envs.locomotion.observation import default_locomotion_observation_spec
from ref2act.envs.locomotion.rewards import (
    LocomotionRewardCfg,
    LocomotionRewardInputs,
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


def _reward_inputs(*, command_error: float) -> LocomotionRewardInputs:
    batch = 3
    joints = 23
    commands = torch.zeros(batch, 3)
    commands[:, 0] = command_error
    return LocomotionRewardInputs(
        commands=commands,
        base_linear_velocity_b=torch.zeros(batch, 3),
        base_angular_velocity_b=torch.zeros(batch, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
        joint_pos=torch.zeros(batch, joints),
        default_joint_pos=torch.zeros(batch, joints),
        joint_acc=torch.zeros(batch, joints),
        applied_torque=torch.zeros(batch, joints),
        applied_action=torch.zeros(batch, joints),
        previous_applied_action=torch.zeros(batch, joints),
    )


def test_locomotion_reward_prefers_matching_velocity() -> None:
    cfg = LocomotionRewardCfg()
    matching = compute_locomotion_reward_terms(_reward_inputs(command_error=0.0), cfg)
    mismatching = compute_locomotion_reward_terms(_reward_inputs(command_error=1.0), cfg)
    assert torch.all(matching["track_linear_velocity"] > mismatching["track_linear_velocity"])
    assert torch.allclose(matching["orientation"], torch.zeros(3))


def test_g1_flat_locomotion_config_and_registry_contract() -> None:
    cfg = G1FlatLocomotionEnvCfg()
    assert cfg.action_space == 23
    assert cfg.robot_spec_name == "g1_23dof"
    assert cfg.terrain.terrain_type == "plane"
    spec = gym.spec("G1FlatLocomotion-v0")
    assert spec.entry_point == "ref2act.envs.locomotion.env:LocomotionEnv"


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
        assert cfg.terrain_curriculum
        assert gym.spec(env_id).entry_point == "ref2act.envs.locomotion.env:LocomotionEnv"

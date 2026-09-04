from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import torch

from ref2act.envs.stand_up.env import (
    STAND_UP_CRITIC_DIM,
    STAND_UP_POLICY_DIM,
    stand_up_observation_contract,
)
from ref2act.envs.stand_up.rewards import (
    StandUpRewardCfg,
    StandUpRewardInputs,
    compute_stand_up_reward_terms,
    normalized_exp_progress,
    stand_up_progress_scores,
    support_score,
)
from ref2act.robots.g1 import G1FlatStandUpEnvCfg


def _inputs(batch: int = 2, joints: int = 3) -> StandUpRewardInputs:
    root = torch.full((batch,), 0.74)
    shoulder = torch.full((batch,), 1.16)
    limits = torch.tensor([[[-1.0, 1.0]]]).repeat(batch, joints, 1)
    zeros3 = torch.zeros(batch, 3)
    zerosj = torch.zeros(batch, joints)
    return StandUpRewardInputs(
        root_height=root,
        shoulder_height=shoulder,
        rise_reference_root_height=torch.full((batch,), 0.25),
        rise_reference_shoulder_height=torch.full((batch,), 0.20),
        target_root_height=root.clone(),
        target_shoulder_height=shoulder.clone(),
        upright_projection=torch.ones(batch),
        rise_reference_upright_projection=torch.zeros(batch),
        base_linear_velocity_b=zeros3,
        base_angular_velocity_b=zeros3,
        joint_position=zerosj,
        default_joint_position=zerosj,
        soft_joint_position_limits=limits,
        joint_velocity=zerosj,
        joint_velocity_limits=torch.full_like(zerosj, 10.0),
        applied_torque=zerosj,
        joint_effort_limits=torch.full_like(zerosj, 100.0),
        target_joint_position=zerosj,
        previous_target_joint_position=zerosj,
        left_foot_load_ratio=torch.full((batch,), 0.50),
        right_foot_load_ratio=torch.full((batch,), 0.50),
        non_foot_contact=torch.zeros(batch, dtype=torch.bool),
        unassisted=torch.ones(batch, dtype=torch.bool),
        completion_event=torch.zeros(batch, dtype=torch.bool),
        standing_reset=torch.zeros(batch, dtype=torch.bool),
        unsafe_termination=torch.zeros(batch, dtype=torch.bool),
        settling=torch.zeros(batch, dtype=torch.bool),
    )


def _terms(values: StandUpRewardInputs, cfg: StandUpRewardCfg | None = None):
    return compute_stand_up_reward_terms(
        values, StandUpRewardCfg() if cfg is None else cfg, step_dt=0.02
    )


def test_stand_up_observation_contract_is_pair_current_plus_privilege() -> None:
    contract = stand_up_observation_contract()
    assert STAND_UP_POLICY_DIM == 55 + 23 == 78
    assert STAND_UP_CRITIC_DIM == 78 + 9 == 87
    assert contract["policy_dim"] == 78
    assert contract["critic_dim"] == 87


def test_g1_stand_up_v5_is_registered_and_deterministic() -> None:
    spec = gym.spec("G1FlatStandUp-v0")
    assert spec.entry_point == "ref2act.envs.stand_up.env:StandUpEnv"
    cfg = G1FlatStandUpEnvCfg()
    assert cfg.observation_space == {"policy": 78, "critic": 87}
    assert cfg.action.mode == "offset"
    assert cfg.action.buffer_length == 1
    assert cfg.action.latency_range is None
    assert cfg.action.noise_scale == 0.0
    assert cfg.events is None
    assert cfg.initial_root_position == (0.0, 0.0, 0.25)
    assert cfg.standing_reset_fraction == 0.10
    assert cfg.rise_reference_root_height == 0.25
    assert cfg.rise_reference_shoulder_height == 0.20
    assert cfg.rise_reference_upright_projection == 0.0
    assert cfg.settling_steps == 0
    assert cfg.initial_root_euler_xyz == (0.0, torch.pi / 2.0, 0.0)
    assert cfg.assistance_body_name == "pelvis"
    assert cfg.assistance_initial_max_gravity_ratio == 0.35
    assert cfg.assistance_zero_probability == 0.20
    assert cfg.assistance_minimum_force_fraction == 0.50
    assert cfg.assistance_full_duration_s == 1.50
    assert cfg.assistance_fade_duration_s == 1.00
    assert cfg.episode_length_s == 4.0
    contract = cfg.rewards.contract()
    assert contract["type"] == "fixed_supine_stand_up_v5_unassisted_support"
    assert contract["success_terminates"] is False


def test_normalized_exp_progress_is_positive_normalized_and_not_early_saturated() -> None:
    score = normalized_exp_progress(
        torch.tensor([-0.5, 0.0, 0.5, 0.8, 1.0, 1.5]), exponent=2.0
    )
    torch.testing.assert_close(score[[0, 1]], torch.zeros(2))
    torch.testing.assert_close(score[4:], torch.ones(2))
    assert 0.0 < score[2] < score[3] < 0.8


def test_rise_objective_requires_root_shoulder_and_upright_together() -> None:
    cfg = StandUpRewardCfg(stand_quality=0.0)
    lying = replace(
        _inputs(),
        root_height=torch.full((2,), 0.25),
        shoulder_height=torch.full((2,), 0.20),
        upright_projection=torch.zeros(2),
    )
    raised_shoulder = replace(lying, shoulder_height=torch.full((2,), 0.80))
    raised_root = replace(lying, root_height=torch.full((2,), 0.60))
    raised_upright = replace(lying, upright_projection=torch.full((2,), 0.50))
    coordinated = replace(
        lying,
        root_height=torch.full((2,), 0.60),
        shoulder_height=torch.full((2,), 0.80),
        upright_projection=torch.full((2,), 0.50),
    )
    lying_terms = _terms(lying, cfg)
    shoulder_terms = _terms(raised_shoulder, cfg)
    root_terms = _terms(raised_root, cfg)
    upright_terms = _terms(raised_upright, cfg)
    coordinated_terms = _terms(coordinated, cfg)
    for terms in (lying_terms, shoulder_terms, root_terms, upright_terms):
        torch.testing.assert_close(terms["rise_progress"], torch.zeros(2))
    assert torch.all(coordinated_terms["rise_progress"] > 0.0)


def test_stand_quality_requires_unassisted_physical_support() -> None:
    cfg = StandUpRewardCfg()
    standing = _inputs()
    without_support = replace(
        standing,
        left_foot_load_ratio=torch.zeros(2),
        right_foot_load_ratio=torch.zeros(2),
    )
    other_support = replace(standing, non_foot_contact=torch.ones(2, dtype=torch.bool))
    assisted = replace(standing, unassisted=torch.zeros(2, dtype=torch.bool))
    standing_reward = _terms(standing, cfg)["stand_quality"]
    unsupported_reward = _terms(without_support, cfg)["stand_quality"]
    assert torch.all(standing_reward > 0.0)
    torch.testing.assert_close(unsupported_reward, torch.zeros(2))
    torch.testing.assert_close(
        _terms(other_support, cfg)["stand_quality"], torch.zeros(2)
    )
    torch.testing.assert_close(_terms(assisted, cfg)["stand_quality"], torch.zeros(2))


def test_support_score_uses_total_load_and_both_feet() -> None:
    cfg = StandUpRewardCfg()
    full = _inputs()
    one_foot = replace(
        full,
        left_foot_load_ratio=torch.ones(2),
        right_foot_load_ratio=torch.zeros(2),
    )
    partial = replace(
        full,
        left_foot_load_ratio=torch.full((2,), 0.05),
        right_foot_load_ratio=torch.full((2,), 0.35),
    )
    torch.testing.assert_close(support_score(full, cfg), torch.ones(2))
    torch.testing.assert_close(support_score(one_foot, cfg), torch.zeros(2))
    expected = torch.full((2,), (0.40 / 0.70) * (0.05 / 0.10))
    torch.testing.assert_close(support_score(partial, cfg), expected)


def test_completion_bonus_excludes_assistance_and_standing_resets() -> None:
    cfg = StandUpRewardCfg()
    completed = replace(_inputs(), completion_event=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(
        _terms(completed, cfg)["completion"], torch.full((2,), cfg.completion)
    )
    assisted = replace(completed, unassisted=torch.zeros(2, dtype=torch.bool))
    standing = replace(completed, standing_reset=torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(_terms(assisted, cfg)["completion"], torch.zeros(2))
    torch.testing.assert_close(_terms(standing, cfg)["completion"], torch.zeros(2))


def test_default_pose_and_stability_live_inside_stand_quality() -> None:
    cfg = StandUpRewardCfg()
    nominal = _terms(_inputs(), cfg)["stand_quality"]
    disturbed = replace(
        _inputs(),
        joint_position=torch.ones(2, 3),
        base_linear_velocity_b=torch.ones(2, 3),
        base_angular_velocity_b=torch.ones(2, 3),
        joint_velocity=torch.ones(2, 3),
    )
    assert torch.all(_terms(disturbed, cfg)["stand_quality"] < nominal)


def test_settling_masks_persistent_terms_but_not_events() -> None:
    cfg = StandUpRewardCfg()
    values = replace(
        _inputs(),
        settling=torch.ones(2, dtype=torch.bool),
        completion_event=torch.ones(2, dtype=torch.bool),
        unsafe_termination=torch.ones(2, dtype=torch.bool),
    )
    terms = _terms(values, cfg)
    for name, value in terms.items():
        if name == "unsafe_termination":
            torch.testing.assert_close(value, torch.full((2,), cfg.unsafe_termination))
        elif name == "completion":
            torch.testing.assert_close(value, torch.full((2,), cfg.completion))
        else:
            torch.testing.assert_close(value, torch.zeros(2))

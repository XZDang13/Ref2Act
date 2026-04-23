import dataclasses
import importlib
import sys
import types

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return vec.clone()


def _quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return torch.zeros(q1.shape[:-1], dtype=q1.dtype, device=q1.device)


def _load_rewards_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
        "isaaclab.sensors": sys.modules.get("isaaclab.sensors"),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
        "sensors": getattr(isaaclab, "sensors", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply = _quat_apply
    math_mod.quat_error_magnitude = _quat_error_magnitude
    math_mod.quat_inv = lambda q: q
    math_mod.quat_mul = lambda q1, q2: q1
    math_mod.yaw_quat = lambda q: q

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.ContactSensor = type("ContactSensor", (), {})

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["isaaclab.sensors"] = sensors_mod
    isaaclab.assets = assets_mod
    isaaclab.utils = utils_mod
    isaaclab.sensors = sensors_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.rewards", None)
        return importlib.import_module("ref2act.envs.motion_tracking.rewards")
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        for attr_name, previous_attr in previous_attrs.items():
            if previous_attr is sentinel:
                if hasattr(isaaclab, attr_name):
                    delattr(isaaclab, attr_name)
            else:
                setattr(isaaclab, attr_name, previous_attr)


def _make_robot(
    body_lin_vel_w: torch.Tensor,
    *,
    body_pos_w: torch.Tensor | None = None,
    body_quat_w: torch.Tensor | None = None,
    body_ang_vel_w: torch.Tensor | None = None,
    body_com_pos_b: torch.Tensor | None = None,
    body_com_pos_w: torch.Tensor | None = None,
    body_link_lin_vel_w: torch.Tensor | None = None,
    masses: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    num_envs = body_lin_vel_w.shape[0]
    num_bodies = body_lin_vel_w.shape[1]
    joint_shape = (num_envs, 2)
    joint_limits = torch.stack(
        (
            -torch.ones(joint_shape, dtype=torch.float32),
            torch.ones(joint_shape, dtype=torch.float32),
        ),
        dim=-1,
    )
    if body_pos_w is None:
        body_pos_w = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_quat_w is None:
        body_quat_w = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32)
        body_quat_w[..., 0] = 1.0
    if body_ang_vel_w is None:
        body_ang_vel_w = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_com_pos_b is None:
        body_com_pos_b = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_com_pos_w is None:
        body_com_pos_w = body_pos_w + body_com_pos_b
    if body_link_lin_vel_w is None:
        body_link_lin_vel_w = body_lin_vel_w.clone()
    if masses is None:
        masses = torch.ones((num_envs, num_bodies), dtype=torch.float32)

    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_acc=torch.zeros(joint_shape, dtype=torch.float32),
            applied_torque=torch.zeros(joint_shape, dtype=torch.float32),
            joint_pos=torch.zeros(joint_shape, dtype=torch.float32),
            joint_vel=torch.zeros(joint_shape, dtype=torch.float32),
            soft_joint_pos_limits=joint_limits,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            body_link_lin_vel_w=body_link_lin_vel_w,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_com_pos_b=body_com_pos_b,
            body_com_pos_w=body_com_pos_w,
        )
        ,
        root_physx_view=types.SimpleNamespace(get_masses=lambda: masses.clone()),
    )


def _make_contact_sensor(net_forces_w_history: torch.Tensor | None, net_forces_w: torch.Tensor | None = None):
    data = types.SimpleNamespace(
        net_forces_w_history=net_forces_w_history,
        net_forces_w=net_forces_w,
        force_matrix_w_history=None,
    )
    return types.SimpleNamespace(data=data, device=torch.device("cpu"))


def _make_action_model(num_envs: int = 1, action_dim: int = 2) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        applied_action=torch.zeros((num_envs, action_dim), dtype=torch.float32),
        previous_applied_action=torch.zeros((num_envs, action_dim), dtype=torch.float32),
    )


def _make_reference_motion(
    num_envs: int = 1,
    num_bodies: int = 3,
    num_joints: int = 2,
    *,
    body_positions: torch.Tensor | None = None,
    body_quaternions: torch.Tensor | None = None,
    body_pos_relative: torch.Tensor | None = None,
    body_quat_relative: torch.Tensor | None = None,
    body_linear_velocities: torch.Tensor | None = None,
    body_angular_velocities: torch.Tensor | None = None,
) -> types.SimpleNamespace:
    if body_quaternions is None:
        body_quaternions = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32)
        body_quaternions[..., 0] = 1.0
    if body_positions is None:
        body_positions = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_pos_relative is None:
        body_pos_relative = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_quat_relative is None:
        body_quat_relative = body_quaternions.clone()
    if body_linear_velocities is None:
        body_linear_velocities = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)
    if body_angular_velocities is None:
        body_angular_velocities = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32)

    return types.SimpleNamespace(
        body_positions=body_positions,
        body_quaternions=body_quaternions,
        body_pos_relative=body_pos_relative,
        body_quat_relative=body_quat_relative,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
        joint_pos=torch.zeros((num_envs, num_joints), dtype=torch.float32),
        joint_vel=torch.zeros((num_envs, num_joints), dtype=torch.float32),
    )


def test_foot_slip_term_only_counts_contacting_feet() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="vector",
            terms=(
                rewards_mod.FootSlipPenaltyTermCfg(
                    foot_body_indices=(0, 1),
                    foot_contact_body_indices=(0, 1),
                    force_threshold=1.0,
                    weight=-0.1,
                ),
            ),
        )
    )

    robot = _make_robot(
        torch.tensor(
            [[[0.3, 0.4, 0.0], [0.6, 0.8, 0.0], [5.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    )
    contact_sensor = _make_contact_sensor(
        torch.tensor(
            [[[[2.0, 0.0, 0.0], [0.2, 0.0, 0.0], [10.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
    )

    reward_vector = rewards.get_task_reward(robot, _make_reference_motion(), contact_sensor, _make_action_model())

    assert reward_vector.shape == (1, 1)
    assert torch.allclose(reward_vector[:, 0], torch.tensor([-0.05], dtype=torch.float32))


def test_default_reward_vector_keeps_existing_term_order() -> None:
    rewards_mod = _load_rewards_module()
    terms = (
        rewards_mod.JointAccPenaltyTermCfg(weight=0.0),
        rewards_mod.JointTorquePenaltyTermCfg(weight=0.0),
        rewards_mod.JointLimitPenaltyTermCfg(weight=0.0),
        rewards_mod.SelfCollisionPenaltyTermCfg(weight=0.0, body_indices=()),
        rewards_mod.FootSlipPenaltyTermCfg(
            weight=-0.1,
            foot_body_indices=(0, 1),
            foot_contact_body_indices=(0, 1),
        ),
        rewards_mod.ActionRatePenaltyTermCfg(weight=0.0),
        rewards_mod.AnchorPositionRewardTermCfg(weight=0.0, anchor_body_index=0),
        rewards_mod.AnchorQuaternionRewardTermCfg(weight=0.0, anchor_body_index=0),
        rewards_mod.KeyPositionRewardTermCfg(weight=0.0, key_body_indices=(0, 1)),
        rewards_mod.KeyQuaternionRewardTermCfg(weight=0.0, key_body_indices=(0, 1)),
        rewards_mod.KeyLinearVelocityRewardTermCfg(weight=0.0, anchor_body_index=0, key_body_indices=(0, 1)),
        rewards_mod.KeyAngularVelocityRewardTermCfg(weight=0.0, anchor_body_index=0, key_body_indices=(0, 1)),
    )
    rewards = rewards_mod.Rewards(rewards_mod.RewardSpec(dt=1.0, output_mode="vector", terms=terms))

    robot = _make_robot(
        torch.tensor(
            [[[0.3, 0.4, 0.0], [0.6, 0.8, 0.0], [2.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    )
    contact_sensor = _make_contact_sensor(
        torch.tensor(
            [[[[2.0, 0.0, 0.0], [0.2, 0.0, 0.0], [10.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
    )

    reward_vector = rewards.get_task_reward(robot, _make_reference_motion(), contact_sensor, _make_action_model())

    assert reward_vector.shape == (1, 12)
    assert torch.allclose(reward_vector[0, 4], torch.tensor(-0.05, dtype=torch.float32))
    assert torch.allclose(reward_vector[0, :4], torch.zeros(4, dtype=torch.float32))
    assert torch.allclose(reward_vector[0, 5:], torch.zeros(7, dtype=torch.float32))


def test_com_position_reward_tracks_mass_weighted_xy_error() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="sum",
            terms=(rewards_mod.CoMPositionRewardTermCfg(weight=1.0, std=1.0),),
        )
    )

    body_pos_w = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    masses = torch.tensor([[1.0, 2.0, 1.0]], dtype=torch.float32)
    reference_motion = _make_reference_motion(body_positions=body_pos_w, body_pos_relative=body_pos_w)
    contact_sensor = _make_contact_sensor(torch.zeros((1, 1, 3, 3), dtype=torch.float32))
    action_model = _make_action_model()

    matched_robot = _make_robot(
        torch.zeros((1, 3, 3), dtype=torch.float32),
        body_pos_w=body_pos_w,
        body_com_pos_w=body_pos_w,
        masses=masses,
    )
    matched_reward = rewards.get_task_reward(matched_robot, reference_motion, contact_sensor, action_model)
    assert torch.allclose(matched_reward, torch.tensor([1.0], dtype=torch.float32))

    shifted_body_com_pos_w = body_pos_w.clone()
    shifted_body_com_pos_w[:, 1, 0] += 1.0
    shifted_robot = _make_robot(
        torch.zeros((1, 3, 3), dtype=torch.float32),
        body_pos_w=body_pos_w,
        body_com_pos_w=shifted_body_com_pos_w,
        masses=masses,
    )
    shifted_reward = rewards.get_task_reward(shifted_robot, reference_motion, contact_sensor, action_model)

    assert torch.allclose(shifted_reward, torch.exp(torch.tensor([-0.25], dtype=torch.float32)))


def test_com_velocity_reward_uses_reference_angular_velocity_com_offset_correction() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="sum",
            terms=(rewards_mod.CoMVelocityRewardTermCfg(weight=1.0, std=1.0, anchor_body_index=0),),
        )
    )

    body_com_pos_b = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    robot = _make_robot(
        torch.zeros((1, 2, 3), dtype=torch.float32),
        body_link_lin_vel_w=torch.tensor(
            [[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_com_pos_b=body_com_pos_b,
    )
    reference_motion = _make_reference_motion(
        num_bodies=2,
        body_linear_velocities=torch.zeros((1, 2, 3), dtype=torch.float32),
        body_angular_velocities=torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    )

    reward = rewards.get_task_reward(
        robot,
        reference_motion,
        _make_contact_sensor(torch.zeros((1, 1, 2, 3), dtype=torch.float32)),
        _make_action_model(),
    )

    assert torch.allclose(reward, torch.tensor([1.0], dtype=torch.float32))


def test_com_velocity_reward_uses_current_angular_velocity_com_offset_correction() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="sum",
            terms=(rewards_mod.CoMVelocityRewardTermCfg(weight=1.0, std=1.0, anchor_body_index=0),),
        )
    )

    body_com_pos_b = torch.tensor(
        [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    robot = _make_robot(
        torch.zeros((1, 2, 3), dtype=torch.float32),
        body_ang_vel_w=torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_com_pos_b=body_com_pos_b,
    )
    reference_motion = _make_reference_motion(
        num_bodies=2,
        body_linear_velocities=torch.tensor(
            [[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_angular_velocities=torch.zeros((1, 2, 3), dtype=torch.float32),
    )

    reward = rewards.get_task_reward(
        robot,
        reference_motion,
        _make_contact_sensor(torch.zeros((1, 1, 2, 3), dtype=torch.float32)),
        _make_action_model(),
    )

    assert torch.allclose(reward, torch.tensor([1.0], dtype=torch.float32))


def test_com_support_reward_handles_no_contact_single_and_double_stance() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="sum",
            terms=(
                rewards_mod.CoMSupportRewardTermCfg(
                    weight=1.0,
                    foot_body_indices=(0, 1),
                    foot_contact_body_indices=(0, 1),
                    force_threshold=10.0,
                    support_margin=0.05,
                    std=0.01,
                ),
            ),
        )
    )

    body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    body_com_pos_w = torch.tensor(
        [
            [[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.2, 0.0], [0.0, 0.2, 0.0]],
        ],
        dtype=torch.float32,
    )
    robot = _make_robot(
        torch.zeros((3, 2, 3), dtype=torch.float32),
        body_pos_w=body_pos_w,
        body_com_pos_w=body_com_pos_w,
    )
    contact_sensor = _make_contact_sensor(
        torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                [[[20.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                [[[20.0, 0.0, 0.0], [20.0, 0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
    )

    reward = rewards.get_task_reward(
        robot,
        _make_reference_motion(num_envs=3, num_bodies=2),
        contact_sensor,
        _make_action_model(num_envs=3),
    )

    expected_single = torch.exp(torch.tensor(-0.25, dtype=torch.float32))
    expected_double = torch.exp(torch.tensor(-2.25, dtype=torch.float32))
    assert torch.allclose(reward, torch.tensor([0.0, expected_single, expected_double], dtype=torch.float32))

    metrics = rewards.last_result.metrics["com_support_reward"]
    assert torch.allclose(metrics["distance"], torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32))
    assert torch.allclose(metrics["contact_count"], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32))


def test_end_effector_rewards_ignore_non_end_effector_body_errors() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="vector",
            terms=(
                rewards_mod.EndEffectorPositionRewardTermCfg(weight=1.0, std=1.0, end_effector_body_indices=(0, 1)),
                rewards_mod.EndEffectorVelocityRewardTermCfg(
                    weight=1.0,
                    std=1.0,
                    anchor_body_index=0,
                    end_effector_body_indices=(0, 1),
                ),
            ),
        )
    )

    robot = _make_robot(
        torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_pos_w=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    )
    reference_motion = _make_reference_motion(num_bodies=3)

    reward_vector = rewards.get_task_reward(
        robot,
        reference_motion,
        _make_contact_sensor(torch.zeros((1, 1, 3, 3), dtype=torch.float32)),
        _make_action_model(),
    )

    assert reward_vector.shape == (1, 2)
    assert torch.allclose(reward_vector[0], torch.ones(2, dtype=torch.float32))


def test_register_reward_term_extends_composer_without_changing_rewards_class() -> None:
    rewards_mod = _load_rewards_module()

    class ConstantBonusReward:
        type_name = "constant_bonus"

        def compute(self, context, spec):
            raw = torch.ones(context.robot.data.joint_pos.shape[0], dtype=torch.float32)
            return rewards_mod.RewardTermResult(
                value=raw * spec.weight,
                metrics={"raw": raw, "weighted": raw * spec.weight},
            )

    rewards_mod.register_reward_term(ConstantBonusReward())
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardSpec(
            dt=1.0,
            output_mode="sum",
            terms=(rewards_mod.RewardTermCfg(id="bonus", type="constant_bonus", weight=2.0),),
        )
    )

    reward = rewards.get_task_reward(
        _make_robot(torch.zeros((1, 3, 3), dtype=torch.float32)),
        _make_reference_motion(),
        _make_contact_sensor(torch.zeros((1, 1, 3, 3), dtype=torch.float32)),
        _make_action_model(),
    )

    assert torch.allclose(reward, torch.tensor([2.0], dtype=torch.float32))
    assert rewards.last_metrics["bonus"]["weighted"] == 2.0

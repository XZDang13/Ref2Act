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


def _make_robot(body_lin_vel_w: torch.Tensor) -> types.SimpleNamespace:
    num_envs = body_lin_vel_w.shape[0]
    joint_shape = (num_envs, 2)
    joint_limits = torch.stack(
        (
            -torch.ones(joint_shape, dtype=torch.float32),
            torch.ones(joint_shape, dtype=torch.float32),
        ),
        dim=-1,
    )
    body_pos_w = torch.zeros((num_envs, 3, 3), dtype=torch.float32)
    body_quat_w = torch.zeros((num_envs, 3, 4), dtype=torch.float32)
    body_quat_w[..., 0] = 1.0
    body_ang_vel_w = torch.zeros((num_envs, 3, 3), dtype=torch.float32)
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_acc=torch.zeros(joint_shape, dtype=torch.float32),
            applied_torque=torch.zeros(joint_shape, dtype=torch.float32),
            joint_pos=torch.zeros(joint_shape, dtype=torch.float32),
            joint_vel=torch.zeros(joint_shape, dtype=torch.float32),
            soft_joint_pos_limits=joint_limits,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
        )
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


def _make_reference_motion(num_envs: int = 1, num_bodies: int = 3, num_joints: int = 2) -> types.SimpleNamespace:
    quats = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32)
    quats[..., 0] = 1.0
    return types.SimpleNamespace(
        body_positions=torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32),
        body_quaternions=quats.clone(),
        body_pos_relative=torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32),
        body_quat_relative=quats.clone(),
        body_linear_velocities=torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32),
        body_angular_velocities=torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32),
        joint_pos=torch.zeros((num_envs, num_joints), dtype=torch.float32),
        joint_vel=torch.zeros((num_envs, num_joints), dtype=torch.float32),
    )


def test_foot_slip_penalty_only_counts_contacting_feet() -> None:
    rewards_mod = _load_rewards_module()
    penalty = rewards_mod.RegulationReward(
        rewards_mod.PenaltyRewardCfg(
            collision_track_body_indices=[],
            foot_body_indices=[0, 1],
            foot_contact_body_indices=[0, 1],
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

    penalty_value = penalty.foot_slip_penalty(robot, contact_sensor, [0, 1], [0, 1], threshold=1.0)

    assert torch.allclose(penalty_value, torch.tensor([0.5], dtype=torch.float32))


def test_foot_slip_penalty_is_zero_without_contact_or_planar_speed() -> None:
    rewards_mod = _load_rewards_module()
    penalty = rewards_mod.RegulationReward(
        rewards_mod.PenaltyRewardCfg(
            collision_track_body_indices=[],
            foot_body_indices=[0, 1],
            foot_contact_body_indices=[0, 1],
        )
    )

    robot = _make_robot(
        torch.tensor(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
    )
    no_contact_sensor = _make_contact_sensor(
        torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )
    zero_planar_vel_sensor = _make_contact_sensor(
        torch.tensor(
            [[[[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
    )

    assert torch.allclose(
        penalty.foot_slip_penalty(robot, no_contact_sensor, [0, 1], [0, 1], threshold=1.0),
        torch.zeros(1, dtype=torch.float32),
    )
    assert torch.allclose(
        penalty.foot_slip_penalty(robot, zero_planar_vel_sensor, [0, 1], [0, 1], threshold=1.0),
        torch.zeros(1, dtype=torch.float32),
    )


def test_reward_vector_includes_foot_slip_as_fifth_penalty_term() -> None:
    rewards_mod = _load_rewards_module()
    rewards = rewards_mod.Rewards(
        rewards_mod.RewardsCfg(
            anchor_body_index=0,
            key_body_indices=[0, 1],
            collision_track_body_indices=[],
            foot_body_indices=[0, 1],
            foot_contact_body_indices=[0, 1],
            dt=1.0,
            return_vector=True,
            joint_acc_weight=0.0,
            joint_torque_wegiht=0.0,
            joint_limit_weight=0.0,
            self_collision_weight=0.0,
            action_rate_weight=0.0,
            foot_slip_weight=-0.1,
            mimic_anchor_position_weight=0.0,
            mimic_anchor_quaternion_weight=0.0,
            mimic_key_position_wegiht=0.0,
            mimic_key_quaternion_weight=0.0,
            mimic_key_linear_vel_weight=0.0,
            mimic_key_ang_vel_weight=0.0,
        )
    )

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

    reward_vector = rewards.get_task_reward(
        robot,
        _make_reference_motion(),
        contact_sensor,
        _make_action_model(),
    )

    assert reward_vector.shape == (1, 12)
    assert torch.allclose(reward_vector[0, 4], torch.tensor(-0.05, dtype=torch.float32))
    assert torch.allclose(reward_vector[0, :4], torch.zeros(4, dtype=torch.float32))
    assert torch.allclose(reward_vector[0, 5:], torch.zeros(7, dtype=torch.float32))

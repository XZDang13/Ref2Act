import importlib
import sys
import types

import torch

def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1:
        vec = vec.expand(quat.shape[:-1] + (3,))
    elif vec.shape[:-1] != quat.shape[:-1]:
        vec = torch.broadcast_to(vec, quat.shape[:-1] + (3,))
    return vec.clone()


def _relative_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key_pos.dim() == 3 and anchor_pos.dim() == 2:
        anchor_pos = anchor_pos[:, None, :].expand_as(key_pos)
    quat = torch.zeros(key_quat.shape, dtype=key_quat.dtype, device=key_quat.device)
    quat[..., 0] = 1.0
    return key_pos - anchor_pos, quat


def _quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    return torch.zeros(q.shape[:-1] + (6,), dtype=q.dtype, device=q.device)


def _load_observation_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.scene": sys.modules.get("isaaclab.scene"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
        "scene": getattr(isaaclab, "scene", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveScene = type("InteractiveScene", (), {})

    math_utils_mod = types.ModuleType("isaaclab.utils.math")
    math_utils_mod.quat_apply_inverse = _quat_apply_inverse
    math_utils_mod.quat_mul = lambda q1, q2: q1
    math_utils_mod.quat_inv = lambda q: q

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_utils_mod

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.scene"] = scene_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_utils_mod
    isaaclab.assets = assets_mod
    isaaclab.scene = scene_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.observation", None)
        return importlib.import_module("ref2act.envs.motion_tracking.observation")
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


def test_policy_observation_noise_does_not_mutate_motion_state() -> None:
    observation_mod = _load_observation_module()
    observation = observation_mod.Observation(anchor_body_index=0, key_body_indices=[0, 1], add_noise=True)

    robot_state = observation_mod.MotionState(
        joint_pos=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        joint_vel=torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        anchor_pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        anchor_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        anchor_lin_vel=torch.tensor([[0.7, 0.8, 0.9]], dtype=torch.float32),
        anchor_ang_vel=torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32),
        key_pos=torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]], dtype=torch.float32),
        key_quat=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        key_lin_vel=torch.zeros((1, 2, 3), dtype=torch.float32),
        key_ang_vel=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    reference_state = observation_mod.MotionState(
        joint_pos=torch.zeros((1, 2), dtype=torch.float32),
        joint_vel=torch.zeros((1, 2), dtype=torch.float32),
        anchor_pos=torch.zeros((1, 3), dtype=torch.float32),
        anchor_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        anchor_lin_vel=torch.zeros((1, 3), dtype=torch.float32),
        anchor_ang_vel=torch.zeros((1, 3), dtype=torch.float32),
        key_pos=torch.zeros((1, 2, 3), dtype=torch.float32),
        key_quat=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        key_lin_vel=torch.zeros((1, 2, 3), dtype=torch.float32),
        key_ang_vel=torch.zeros((1, 2, 3), dtype=torch.float32),
    )

    joint_pos_before = robot_state.joint_pos.clone()
    joint_vel_before = robot_state.joint_vel.clone()
    anchor_ang_vel_before = robot_state.anchor_ang_vel.clone()

    torch.manual_seed(0)
    observation.get_policy_observation(
        robot_state,
        reference_state,
        gravity_vector=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        last_applied_action=torch.zeros((1, 2), dtype=torch.float32),
    )

    assert torch.allclose(robot_state.joint_pos, joint_pos_before)
    assert torch.allclose(robot_state.joint_vel, joint_vel_before)
    assert torch.allclose(robot_state.anchor_ang_vel, anchor_ang_vel_before)


def test_default_observation_keeps_privileged_observation_clean() -> None:
    observation_mod = _load_observation_module()
    observation = observation_mod.Observation(anchor_body_index=0, key_body_indices=[0, 1], add_noise=True)

    joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    joint_vel = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    anchor_ang_vel = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)
    last_action = torch.tensor([[0.7, 0.8]], dtype=torch.float32)

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=joint_pos.clone(),
            joint_vel=joint_vel.clone(),
            body_pos_w=torch.tensor(
                [[[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]],
                dtype=torch.float32,
            ),
            body_quat_w=torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            body_lin_vel_w=torch.tensor(
                [[[0.9, 1.0, 1.1], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            body_ang_vel_w=torch.tensor(
                [[[0.4, 0.5, 0.6], [0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        joint_pos=torch.zeros((1, 2), dtype=torch.float32),
        joint_vel=torch.zeros((1, 2), dtype=torch.float32),
        body_positions=torch.tensor(
            [[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]],
            dtype=torch.float32,
        ),
        body_quaternions=torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        body_linear_velocities=torch.zeros((1, 2, 3), dtype=torch.float32),
        body_angular_velocities=torch.zeros((1, 2, 3), dtype=torch.float32),
    )
    scene = types.SimpleNamespace(env_origins=torch.zeros((1, 3), dtype=torch.float32))

    torch.manual_seed(0)
    obs = observation.get_default_observation(robot, reference_motion, scene, last_action)

    assert torch.allclose(robot.data.joint_pos, joint_pos)
    assert torch.allclose(robot.data.joint_vel, joint_vel)
    assert torch.allclose(robot.data.body_ang_vel_w[:, 0], anchor_ang_vel)

    robot_obs = obs["robot"]
    privilege_obs = obs["privilege"]
    assert not torch.allclose(robot_obs[0, 3:6], anchor_ang_vel[0])
    assert not torch.allclose(robot_obs[0, 6:8], joint_pos[0])
    assert not torch.allclose(robot_obs[0, 8:10], joint_vel[0])

    num_joints = joint_pos.shape[1]
    num_keys = 2
    anchor_lin_start = 2 * num_joints + 3 + 6 + num_keys * 3 + num_keys * 6
    anchor_ang_start = anchor_lin_start + 3
    joint_pos_start = anchor_ang_start + 3
    joint_vel_start = joint_pos_start + num_joints
    last_action_start = joint_vel_start + num_joints

    assert torch.allclose(privilege_obs[0, anchor_ang_start:anchor_ang_start + 3], anchor_ang_vel[0])
    assert torch.allclose(privilege_obs[0, joint_pos_start:joint_pos_start + num_joints], joint_pos[0])
    assert torch.allclose(privilege_obs[0, joint_vel_start:joint_vel_start + num_joints], joint_vel[0])
    assert torch.allclose(privilege_obs[0, last_action_start:last_action_start + num_joints], last_action[0])

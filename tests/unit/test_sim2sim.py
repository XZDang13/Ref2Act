import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_sim2sim_module():
    sentinel = object()
    previous_modules = {
        "mujoco": sys.modules.get("mujoco"),
        "mujoco_viewer": sys.modules.get("mujoco_viewer"),
        "mujoco_viewer.mujoco_viewer": sys.modules.get("mujoco_viewer.mujoco_viewer"),
    }

    mujoco_mod = types.ModuleType("mujoco")
    mujoco_mod.MjModel = type("MjModel", (), {"from_xml_path": staticmethod(lambda path: object())})
    mujoco_mod.MjData = type("MjData", (), {})
    mujoco_mod.mj_resetData = lambda model, data: None
    mujoco_mod.mj_forward = lambda model, data: None
    mujoco_mod.mj_step = lambda model, data: None
    mujoco_mod.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1, mjOBJ_JOINT=2)
    mujoco_mod.mjtJoint = types.SimpleNamespace(mjJNT_FREE=0)
    mujoco_mod.mj_name2id = (
        lambda model, objtype, name: model.body_name_to_id.get(name, -1)
        if objtype == mujoco_mod.mjtObj.mjOBJ_BODY
        else model.joint_name_to_id.get(name, -1)
    )
    mujoco_mod.mj_id2name = (
        lambda model, objtype, objid: model.id_to_body_name.get(objid)
        if objtype == mujoco_mod.mjtObj.mjOBJ_BODY
        else model.id_to_joint_name.get(objid)
    )
    mujoco_mod.mj_objectVelocity = lambda model, data, objtype, objid, res, flg_local: res.fill(0.0)

    viewer_pkg = types.ModuleType("mujoco_viewer")
    viewer_mod = types.ModuleType("mujoco_viewer.mujoco_viewer")

    class _Viewer:
        def __init__(self, *args, **kwargs):
            self.is_alive = True

        def render(self):
            return None

        def close(self):
            self.is_alive = False

    viewer_mod.MujocoViewer = _Viewer
    viewer_pkg.mujoco_viewer = viewer_mod

    sys.modules["mujoco"] = mujoco_mod
    sys.modules["mujoco_viewer"] = viewer_pkg
    sys.modules["mujoco_viewer.mujoco_viewer"] = viewer_mod

    try:
        sys.modules.pop("ref2act.bridges.mujoco.env", None)
        return importlib.import_module("ref2act.bridges.mujoco.env")
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def _make_observation_env(sim2sim_mod):
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.policy_dt = 0.1
    env.times = torch.zeros(1, dtype=torch.float32)
    env.motion_id = torch.zeros(1, dtype=torch.long)
    env.previous_action = torch.tensor([18.0, 19.0], dtype=torch.float32)
    env.motion_lib = types.SimpleNamespace(
        get_duration=lambda motion_id: torch.tensor([1.0], dtype=torch.float32),
    )
    env.get_motion_command = lambda times: (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32),
    )
    env.get_projected_gravity = lambda: torch.tensor([8.0, 9.0, 10.0], dtype=torch.float32)
    env.get_anchor_ang_vel_b = lambda: torch.tensor([11.0, 12.0, 13.0], dtype=torch.float32)
    env.get_joint_pos = lambda: torch.tensor([14.0, 15.0], dtype=torch.float32)
    env.get_joint_vel = lambda: torch.tensor([16.0, 17.0], dtype=torch.float32)
    return env


def _make_action_env(sim2sim_mod):
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.times = torch.tensor([0.3], dtype=torch.float32)
    env.previous_action = torch.zeros(2, dtype=torch.float32)
    env.action_scale = torch.tensor([2.0, 3.0], dtype=torch.float32)
    env.action_offset = torch.tensor([1.0, -1.0], dtype=torch.float32)
    env.joint_pos_limits_lower = torch.tensor([0.0, -2.0], dtype=torch.float32)
    env.joint_pos_limits_upper = torch.tensor([2.0, 2.0], dtype=torch.float32)
    env.action_mode = "absolute"
    env.get_joint_pos = lambda: (_ for _ in ()).throw(AssertionError("current joints should not be queried"))
    env.get_motion_command = lambda times: (_ for _ in ()).throw(AssertionError("reference motion should not be queried"))
    return env


def _make_fake_root_alignment_system(sim2sim_mod):
    model = types.SimpleNamespace(
        body_name_to_id={"pelvis": 1, "torso_link": 2},
        id_to_body_name={1: "pelvis", 2: "torso_link"},
        joint_name_to_id={"floating_base_joint": 0},
        id_to_joint_name={0: "floating_base_joint"},
        jnt_type=np.array([sim2sim_mod.mujoco.mjtJoint.mjJNT_FREE], dtype=np.int32),
        jnt_bodyid=np.array([1], dtype=np.int32),
        root_rel_pos=np.array([-0.1, 0.05, 0.2], dtype=np.float64),
        root_rel_quat=sim2sim_mod.normalize_quat_np(
            np.array([0.96592583, 0.0, 0.0, 0.25881905], dtype=np.float64)
        ),
        root_relative_ang_vel=np.array([0.5, -0.25, 0.1], dtype=np.float64),
        root_relative_lin_vel=np.array([0.2, 0.3, -0.4], dtype=np.float64),
    )
    data = types.SimpleNamespace(
        qpos=np.zeros(9, dtype=np.float32),
        qvel=np.zeros(8, dtype=np.float32),
        ctrl=np.zeros(2, dtype=np.float32),
        xpos=np.zeros((3, 3), dtype=np.float64),
        xquat=np.zeros((3, 4), dtype=np.float64),
    )
    data.xquat[:, 0] = 1.0

    def _fake_forward(model, data):
        free_pos = np.asarray(data.qpos[:3], dtype=np.float64)
        free_quat = sim2sim_mod.normalize_quat_np(np.asarray(data.qpos[3:7], dtype=np.float64))
        data.xpos[1] = free_pos
        data.xquat[1] = free_quat
        data.xpos[2] = free_pos + sim2sim_mod.quat_apply_np(free_quat, model.root_rel_pos)
        data.xquat[2] = sim2sim_mod.normalize_quat_np(sim2sim_mod.quat_mul_np(free_quat, model.root_rel_quat))

    def _fake_object_velocity(model, data, objtype, objid, res, flg_local):
        free_lin_vel = np.asarray(data.qvel[:3], dtype=np.float64)
        free_ang_vel = np.asarray(data.qvel[3:6], dtype=np.float64)
        if objid == 1:
            res[:] = np.concatenate([free_ang_vel, free_lin_vel])
            return

        if objid != 2:
            res.fill(0.0)
            return

        root_offset_world = data.xpos[2] - data.xpos[1]
        root_ang_vel = free_ang_vel + model.root_relative_ang_vel
        root_lin_vel = free_lin_vel + np.cross(free_ang_vel, root_offset_world) + model.root_relative_lin_vel
        res[:] = np.concatenate([root_ang_vel, root_lin_vel])

    return model, data, _fake_forward, _fake_object_velocity


def _assert_quat_matches(actual: np.ndarray, expected: np.ndarray, atol: float = 1.0e-6) -> None:
    assert np.allclose(actual, expected, atol=atol) or np.allclose(actual, -expected, atol=atol)


def test_get_obs_matches_policy_observation_layout() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = _make_observation_env(sim2sim_mod)

    obs = env.get_obs(advance_time=False)

    assert torch.allclose(
        obs,
        torch.tensor(
            [5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(env.times, torch.zeros(1, dtype=torch.float32))


def test_get_obs_dict_matches_isaac_style_motion_and_robot_split() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = _make_observation_env(sim2sim_mod)

    obs = env.get_obs_dict(advance_time=False)

    assert set(obs) == {"motion", "robot"}
    assert torch.allclose(obs["motion"], torch.tensor([5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32))
    assert torch.allclose(
        obs["robot"],
        torch.tensor([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0], dtype=torch.float32),
    )
    assert torch.allclose(env.times, torch.zeros(1, dtype=torch.float32))


def test_custom_observation_builder_can_override_policy_and_default_outputs() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = _make_observation_env(sim2sim_mod)

    class _CustomObservation(sim2sim_mod.IsaacLabMujocoObservation):
        def get_default_observation(self, env, context):
            obs = super().get_default_observation(env, context)
            obs["custom"] = torch.cat([context.joint_pos, context.previous_action])
            return obs

        def get_policy_observation(self, env, context):
            return torch.tensor([42.0, 24.0], dtype=torch.float32)

    env.observation_builder = _CustomObservation()

    obs = env.get_obs(advance_time=False)
    obs_dict = env.get_obs_dict(advance_time=False)

    assert torch.allclose(obs, torch.tensor([42.0, 24.0], dtype=torch.float32))
    assert set(obs_dict) == {"motion", "robot", "custom"}
    assert torch.allclose(obs_dict["custom"], torch.tensor([14.0, 15.0, 18.0, 19.0], dtype=torch.float32))
    assert torch.allclose(env.times, torch.zeros(1, dtype=torch.float32))


def test_process_action_matches_default_absolute_action_layout() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = _make_action_env(sim2sim_mod)

    output = env.process_action(torch.tensor([1.0, -1.0], dtype=torch.float64))

    assert output.applied_action.dtype == torch.float32
    assert torch.allclose(output.applied_action, torch.tensor([1.0, -1.0], dtype=torch.float32))
    assert torch.allclose(output.target_joint_pos, torch.tensor([2.0, -2.0], dtype=torch.float32))


def test_custom_action_builder_can_override_applied_action_and_target_position() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = _make_action_env(sim2sim_mod)
    env.mj_model = object()
    env.mj_data = types.SimpleNamespace(ctrl=np.zeros(2, dtype=np.float32))
    env.mj_viewer = None
    env.policy_dt = 0.0
    env.decimation = 2
    env.n_steps = 0

    class _CustomAction(sim2sim_mod.IsaacLabMujocoAction):
        def process_action(self, env, context):
            applied_action = context.raw_action + 10.0
            target_joint_pos = torch.tensor([0.25, -0.25], dtype=torch.float32)
            return sim2sim_mod.MujocoActionOutput(
                applied_action=applied_action,
                target_joint_pos=target_joint_pos,
            )

    env.action_builder = _CustomAction()

    apply_calls: list[str] = []
    env._apply_actions = lambda: apply_calls.append("apply")

    mj_step_calls: list[str] = []
    sim2sim_mod.mujoco.mj_step = lambda model, data: mj_step_calls.append("step")

    get_obs_calls: list[bool] = []

    def _get_obs(advance_time: bool = True) -> torch.Tensor:
        get_obs_calls.append(advance_time)
        return torch.tensor([55.0], dtype=torch.float32)

    env.get_obs = _get_obs

    obs = env.step(torch.tensor([1.0, -1.0], dtype=torch.float32))

    assert torch.allclose(obs, torch.tensor([55.0], dtype=torch.float32))
    assert torch.allclose(env.previous_action, torch.tensor([11.0, 9.0], dtype=torch.float32))
    assert torch.allclose(env.target_pos, torch.tensor([0.25, -0.25], dtype=torch.float32))
    assert apply_calls == ["apply", "apply"]
    assert mj_step_calls == ["step", "step"]
    assert get_obs_calls == [True]
    assert env.n_steps == 1


def test_reset_uses_current_reference_time_and_restores_velocities() -> None:
    sim2sim_mod = _load_sim2sim_module()

    reset_calls: list[str] = []
    forward_calls: list[str] = []

    sim2sim_mod.mujoco.mj_resetData = lambda model, data: reset_calls.append("reset")
    sim2sim_mod.mujoco.mj_forward = lambda model, data: forward_calls.append("forward")

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = object()
    env.mj_data = types.SimpleNamespace(
        qpos=np.zeros(9, dtype=np.float32),
        qvel=np.zeros(8, dtype=np.float32),
        ctrl=np.zeros(2, dtype=np.float32),
    )
    env.mj_viewer = None
    env.previous_action = torch.ones(2, dtype=torch.float32)
    env.motion_id = torch.ones(1, dtype=torch.long)
    env.isaac2mujoco = [1, 0]
    env.root_body_index = 0

    reference_motion = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_positions": torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float32),
        "body_quaternions": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        "body_linear_velocities": torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
        "body_angular_velocities": torch.tensor([[[4.0, 5.0, 6.0]]], dtype=torch.float32),
    }
    env.motion_lib = types.SimpleNamespace(sample_motion=lambda motion_ids, times: reference_motion)
    env._solve_free_joint_state_from_root_reference = lambda **kwargs: (
        np.asarray([0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    )

    get_obs_calls: list[bool] = []

    def _get_obs(advance_time: bool = True) -> torch.Tensor:
        get_obs_calls.append(advance_time)
        return torch.tensor([42.0], dtype=torch.float32)

    env.get_obs = _get_obs

    obs = env.reset()

    assert reset_calls == ["reset"]
    assert forward_calls == ["forward"]
    assert get_obs_calls == [False]
    assert torch.allclose(obs, torch.tensor([42.0], dtype=torch.float32))
    assert torch.allclose(env.previous_action, torch.zeros(2, dtype=torch.float32))
    assert torch.equal(env.motion_id, torch.zeros(1, dtype=torch.long))
    assert torch.allclose(env.times, torch.zeros(1, dtype=torch.float32))
    assert np.allclose(env.mj_data.qpos[:7], np.asarray([0.0, 0.0, 0.55, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(env.mj_data.qpos[7:], np.asarray([0.2, 0.1], dtype=np.float32))
    assert np.allclose(env.mj_data.qvel[:3], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(env.mj_data.qvel[3:6], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))
    assert np.allclose(env.mj_data.qvel[6:], np.asarray([0.4, 0.3], dtype=np.float32))
    assert torch.allclose(env.target_pos, torch.tensor([0.1, 0.2], dtype=torch.float32))


def test_get_motion_command_uses_anchor_body_for_target_projected_gravity() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.motion_id = torch.zeros(1, dtype=torch.long)
    env.gravity_vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    env.root_body_index = 0
    env.anchor_body_index = 1

    reference_motion = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_quaternions": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.70710677, 0.70710677, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    }
    env.motion_lib = types.SimpleNamespace(sample_motion=lambda motion_ids, times: reference_motion)

    _, _, projected_gravity = env.get_motion_command(torch.zeros(1, dtype=torch.float32))

    expected_gravity = sim2sim_mod.quat_rotate_inverse(reference_motion["body_quaternions"][0, 1], env.gravity_vector)
    assert torch.allclose(projected_gravity, expected_gravity.float())


def test_current_anchor_state_uses_anchor_body_quaternion_and_body_frame_ang_vel() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.anchor_body_id = 2
    env.gravity_vector = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    env._get_body_world_quat = lambda body_id: np.asarray([0.70710677, 0.70710677, 0.0, 0.0], dtype=np.float32)
    env._get_body_world_twist = lambda body_id: (
        np.asarray([9.0, 8.0, 7.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )

    projected_gravity = env.get_projected_gravity()
    anchor_ang_vel_b = env.get_anchor_ang_vel_b()

    anchor_quat_w = torch.tensor([0.70710677, 0.70710677, 0.0, 0.0], dtype=torch.float32)
    expected_gravity = sim2sim_mod.quat_rotate_inverse(
        anchor_quat_w,
        env.gravity_vector,
    )
    expected_anchor_ang_vel_b = sim2sim_mod.quat_rotate_inverse(
        anchor_quat_w,
        torch.tensor([9.0, 8.0, 7.0], dtype=torch.float32),
    )
    assert torch.allclose(projected_gravity, expected_gravity.float())
    assert torch.allclose(anchor_ang_vel_b, expected_anchor_ang_vel_b.float())


def test_reset_with_pelvis_root_keeps_free_joint_aligned_to_pelvis_reference() -> None:
    sim2sim_mod = _load_sim2sim_module()
    model, data, fake_forward, fake_object_velocity = _make_fake_root_alignment_system(sim2sim_mod)
    sim2sim_mod.mujoco.mj_forward = fake_forward
    sim2sim_mod.mujoco.mj_objectVelocity = fake_object_velocity

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = model
    env.mj_data = data
    env.mj_viewer = None
    env.previous_action = torch.ones(2, dtype=torch.float32)
    env.motion_id = torch.ones(1, dtype=torch.long)
    env.isaac2mujoco = [1, 0]
    env.free_root_body_id = 1
    env.root_body_id = 1
    env.root_body_index = 0

    reference_motion = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_positions": torch.tensor(
            [[[0.4, -0.2, 0.9], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "body_quaternions": torch.tensor(
            [[[0.9238795, 0.0, 0.0, 0.3826834], [1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "body_linear_velocities": torch.tensor(
            [[[1.1, -0.3, 0.7], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "body_angular_velocities": torch.tensor(
            [[[0.2, -0.4, 0.8], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    }
    env.motion_lib = types.SimpleNamespace(sample_motion=lambda motion_ids, times: reference_motion)
    env.get_obs = lambda advance_time=False: torch.tensor([13.0], dtype=torch.float32)

    obs = env.reset()

    desired_root_pos = reference_motion["body_positions"][0, 0].numpy().copy()
    desired_root_pos[2] += 0.05
    desired_root_quat = reference_motion["body_quaternions"][0, 0].numpy()
    desired_root_lin_vel = reference_motion["body_linear_velocities"][0, 0].numpy()
    desired_root_ang_vel = reference_motion["body_angular_velocities"][0, 0].numpy()
    root_velocity = np.zeros(6, dtype=np.float64)
    sim2sim_mod.mujoco.mj_objectVelocity(model, data, sim2sim_mod.mujoco.mjtObj.mjOBJ_BODY, 1, root_velocity, 0)

    assert torch.allclose(obs, torch.tensor([13.0], dtype=torch.float32))
    assert np.allclose(data.qpos[:3], desired_root_pos, atol=1.0e-6)
    _assert_quat_matches(data.qpos[3:7], desired_root_quat)
    assert np.allclose(root_velocity[:3], desired_root_ang_vel, atol=1.0e-6)
    assert np.allclose(root_velocity[3:], desired_root_lin_vel, atol=1.0e-6)


def test_reset_with_torso_root_aligns_selected_root_body_world_pose_and_twist() -> None:
    sim2sim_mod = _load_sim2sim_module()
    model, data, fake_forward, fake_object_velocity = _make_fake_root_alignment_system(sim2sim_mod)
    sim2sim_mod.mujoco.mj_forward = fake_forward
    sim2sim_mod.mujoco.mj_objectVelocity = fake_object_velocity

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = model
    env.mj_data = data
    env.mj_viewer = None
    env.previous_action = torch.ones(2, dtype=torch.float32)
    env.motion_id = torch.ones(1, dtype=torch.long)
    env.isaac2mujoco = [1, 0]
    env.free_root_body_id = 1
    env.root_body_id = 2
    env.root_body_index = 1

    reference_motion = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_positions": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.3, -0.7, 1.4]]],
            dtype=torch.float32,
        ),
        "body_quaternions": torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.8660254, 0.0, 0.0, 0.5]]],
            dtype=torch.float32,
        ),
        "body_linear_velocities": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.5, -0.2, 0.9]]],
            dtype=torch.float32,
        ),
        "body_angular_velocities": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.7, -0.1, 0.3]]],
            dtype=torch.float32,
        ),
    }
    env.motion_lib = types.SimpleNamespace(sample_motion=lambda motion_ids, times: reference_motion)
    env.get_obs = lambda advance_time=False: torch.tensor([21.0], dtype=torch.float32)

    obs = env.reset()

    desired_root_pos = reference_motion["body_positions"][0, 1].numpy().copy()
    desired_root_pos[2] += 0.05
    desired_root_quat = reference_motion["body_quaternions"][0, 1].numpy()
    desired_root_lin_vel = reference_motion["body_linear_velocities"][0, 1].numpy()
    desired_root_ang_vel = reference_motion["body_angular_velocities"][0, 1].numpy()

    expected_free_quat = sim2sim_mod.normalize_quat_np(
        sim2sim_mod.quat_mul_np(desired_root_quat, sim2sim_mod.quat_inv_np(model.root_rel_quat))
    )
    expected_free_pos = desired_root_pos - sim2sim_mod.quat_apply_np(expected_free_quat, model.root_rel_pos)
    expected_root_offset_world = sim2sim_mod.quat_apply_np(expected_free_quat, model.root_rel_pos)
    expected_free_ang_vel = desired_root_ang_vel - model.root_relative_ang_vel
    expected_free_lin_vel = desired_root_lin_vel - model.root_relative_lin_vel - np.cross(
        expected_free_ang_vel,
        expected_root_offset_world,
    )

    root_velocity = np.zeros(6, dtype=np.float64)
    sim2sim_mod.mujoco.mj_objectVelocity(model, data, sim2sim_mod.mujoco.mjtObj.mjOBJ_BODY, 2, root_velocity, 0)

    assert torch.allclose(obs, torch.tensor([21.0], dtype=torch.float32))
    assert np.allclose(data.qpos[:3], expected_free_pos, atol=1.0e-6)
    _assert_quat_matches(data.qpos[3:7], expected_free_quat)
    assert np.allclose(data.qvel[:3], expected_free_lin_vel, atol=1.0e-6)
    assert np.allclose(data.qvel[3:6], expected_free_ang_vel, atol=1.0e-6)
    assert np.allclose(data.xpos[2], desired_root_pos, atol=1.0e-6)
    _assert_quat_matches(data.xquat[2], desired_root_quat)
    assert np.allclose(root_velocity[:3], desired_root_ang_vel, atol=1.0e-6)
    assert np.allclose(root_velocity[3:], desired_root_lin_vel, atol=1.0e-6)


def test_missing_motion_root_body_raises_explicit_error() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.motion_lib = types.SimpleNamespace(body_names=["pelvis"])

    with pytest.raises(ValueError, match="root body 'torso_link' was not found in motion_lib.body_names."):
        env._resolve_motion_body_index("torso_link", role="root")


def test_missing_model_anchor_body_raises_explicit_error() -> None:
    sim2sim_mod = _load_sim2sim_module()
    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = types.SimpleNamespace(
        body_name_to_id={"pelvis": 1},
        id_to_body_name={1: "pelvis"},
        joint_name_to_id={},
        id_to_joint_name={},
    )

    with pytest.raises(ValueError, match="anchor body 'torso_link' was not found in MuJoCo model body names."):
        env._resolve_model_body_id("torso_link", role="anchor")


def test_step_normalizes_actions_and_advances_observation() -> None:
    sim2sim_mod = _load_sim2sim_module()

    mj_step_calls: list[str] = []
    sim2sim_mod.mujoco.mj_step = lambda model, data: mj_step_calls.append("step")

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = object()
    env.mj_data = types.SimpleNamespace(ctrl=np.zeros(2, dtype=np.float32))
    env.mj_viewer = None
    env.policy_dt = 0.0
    env.decimation = 3
    env.n_steps = 0
    env.previous_action = torch.zeros(2, dtype=torch.float32)
    env.action_scale = torch.tensor([2.0, 3.0], dtype=torch.float32)
    env.action_offset = torch.tensor([1.0, -1.0], dtype=torch.float32)
    env.joint_pos_limits_lower = torch.tensor([0.0, -2.0], dtype=torch.float32)
    env.joint_pos_limits_upper = torch.tensor([2.0, 2.0], dtype=torch.float32)

    apply_calls: list[str] = []
    env._apply_actions = lambda: apply_calls.append("apply")

    get_obs_calls: list[bool] = []

    def _get_obs(advance_time: bool = True) -> torch.Tensor:
        get_obs_calls.append(advance_time)
        return torch.tensor([24.0], dtype=torch.float32)

    env.get_obs = _get_obs

    obs = env.step(torch.tensor([1.0, -1.0], dtype=torch.float64))

    assert torch.allclose(obs, torch.tensor([24.0], dtype=torch.float32))
    assert env.previous_action.dtype == torch.float32
    assert torch.allclose(env.previous_action, torch.tensor([1.0, -1.0], dtype=torch.float32))
    assert torch.allclose(env.target_pos, torch.tensor([2.0, -2.0], dtype=torch.float32))
    assert apply_calls == ["apply", "apply", "apply"]
    assert mj_step_calls == ["step", "step", "step"]
    assert get_obs_calls == [True]
    assert env.n_steps == 1


def test_step_residual_mode_uses_current_reference_joint_positions() -> None:
    sim2sim_mod = _load_sim2sim_module()

    mj_step_calls: list[str] = []
    sim2sim_mod.mujoco.mj_step = lambda model, data: mj_step_calls.append("step")

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = object()
    env.mj_data = types.SimpleNamespace(ctrl=np.zeros(2, dtype=np.float32))
    env.mj_viewer = None
    env.policy_dt = 0.0
    env.decimation = 2
    env.n_steps = 0
    env.times = torch.tensor([0.3], dtype=torch.float32)
    env.previous_action = torch.zeros(2, dtype=torch.float32)
    env.action_scale = torch.tensor([0.5, 0.75], dtype=torch.float32)
    env.action_offset = torch.tensor([9.0, 9.0], dtype=torch.float32)
    env.action_mode = "residual"
    env.joint_pos_limits_lower = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    env.joint_pos_limits_upper = torch.tensor([0.8, 1.0], dtype=torch.float32)

    queried_times: list[torch.Tensor] = []

    def _get_motion_command(times: torch.Tensor):
        queried_times.append(times.clone())
        return (
            torch.tensor([0.4, 0.1], dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        )

    env.get_motion_command = _get_motion_command

    apply_calls: list[str] = []
    env._apply_actions = lambda: apply_calls.append("apply")

    get_obs_calls: list[bool] = []

    def _get_obs(advance_time: bool = True) -> torch.Tensor:
        get_obs_calls.append(advance_time)
        return torch.tensor([99.0], dtype=torch.float32)

    env.get_obs = _get_obs

    obs = env.step(torch.tensor([1.0, 2.0], dtype=torch.float32))

    assert torch.allclose(obs, torch.tensor([99.0], dtype=torch.float32))
    assert queried_times == [torch.tensor([0.3], dtype=torch.float32)]
    assert torch.allclose(env.target_pos, torch.tensor([0.8, 1.0], dtype=torch.float32))
    assert torch.allclose(env.previous_action, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert apply_calls == ["apply", "apply"]
    assert mj_step_calls == ["step", "step"]
    assert get_obs_calls == [True]
    assert env.n_steps == 1


def test_step_current_residual_mode_uses_current_joint_positions() -> None:
    sim2sim_mod = _load_sim2sim_module()

    mj_step_calls: list[str] = []
    sim2sim_mod.mujoco.mj_step = lambda model, data: mj_step_calls.append("step")

    env = object.__new__(sim2sim_mod.MujocoEnv)
    env.mj_model = object()
    env.mj_data = types.SimpleNamespace(ctrl=np.zeros(2, dtype=np.float32))
    env.mj_viewer = None
    env.policy_dt = 0.0
    env.decimation = 2
    env.n_steps = 0
    env.previous_action = torch.zeros(2, dtype=torch.float32)
    env.action_scale = torch.tensor([0.5, 0.75], dtype=torch.float32)
    env.action_offset = torch.tensor([9.0, 9.0], dtype=torch.float32)
    env.action_mode = "current_residual"
    env.joint_pos_limits_lower = torch.tensor([-1.0, -1.0], dtype=torch.float32)
    env.joint_pos_limits_upper = torch.tensor([0.8, 1.0], dtype=torch.float32)

    env.get_motion_command = lambda times: (_ for _ in ()).throw(AssertionError("reference motion should not be queried"))

    joint_pos_calls: list[str] = []

    def _get_joint_pos() -> torch.Tensor:
        joint_pos_calls.append("joint_pos")
        return torch.tensor([0.4, 0.1], dtype=torch.float32)

    env.get_joint_pos = _get_joint_pos

    apply_calls: list[str] = []
    env._apply_actions = lambda: apply_calls.append("apply")

    get_obs_calls: list[bool] = []

    def _get_obs(advance_time: bool = True) -> torch.Tensor:
        get_obs_calls.append(advance_time)
        return torch.tensor([77.0], dtype=torch.float32)

    env.get_obs = _get_obs

    obs = env.step(torch.tensor([1.0, 2.0], dtype=torch.float32))

    assert torch.allclose(obs, torch.tensor([77.0], dtype=torch.float32))
    assert joint_pos_calls == ["joint_pos"]
    assert torch.allclose(env.target_pos, torch.tensor([0.8, 1.0], dtype=torch.float32))
    assert torch.allclose(env.previous_action, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert apply_calls == ["apply", "apply"]
    assert mj_step_calls == ["step", "step"]
    assert get_obs_calls == [True]
    assert env.n_steps == 1

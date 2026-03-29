from pathlib import Path
import importlib
import sys
import types

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
        sys.modules.pop("Ref2Act.sim2sim", None)
        return importlib.import_module("Ref2Act.sim2sim")
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def test_get_obs_matches_policy_observation_layout() -> None:
    sim2sim_mod = _load_sim2sim_module()
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
    env.get_base_ang_vel = lambda: torch.tensor([11.0, 12.0, 13.0], dtype=torch.float32)
    env.get_joint_pos = lambda: torch.tensor([14.0, 15.0], dtype=torch.float32)
    env.get_joint_vel = lambda: torch.tensor([16.0, 17.0], dtype=torch.float32)

    obs = env.get_obs(advance_time=False)

    assert torch.allclose(
        obs,
        torch.tensor(
            [5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(env.times, torch.zeros(1, dtype=torch.float32))


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
    env.root_index = 0

    reference_motion = {
        "joint_pos": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        "joint_vel": torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        "body_positions": torch.tensor([[[0.0, 0.0, 0.5]]], dtype=torch.float32),
        "body_quaternions": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        "body_linear_velocities": torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
        "body_angular_velocities": torch.tensor([[[4.0, 5.0, 6.0]]], dtype=torch.float32),
    }
    env.motion_lib = types.SimpleNamespace(sample_motion=lambda motion_ids, times: reference_motion)

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

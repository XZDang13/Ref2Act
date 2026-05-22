import importlib
import sys
import types

import torch


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim == 1:
        vec = vec.expand(quat.shape[:-1] + (3,))
    elif vec.shape[:-1] != quat.shape[:-1]:
        vec = torch.broadcast_to(vec, quat.shape[:-1] + (3,))

    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + quat[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_conj = quat.clone()
    quat_conj[..., 1:] = -quat_conj[..., 1:]
    return _quat_apply(quat_conj, vec)


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


def _load_modules():
    sentinel = object()
    previous_root = sys.modules.get("isaaclab", sentinel)
    if previous_root is sentinel:
        isaaclab = types.ModuleType("isaaclab")
        sys.modules["isaaclab"] = isaaclab
    else:
        import isaaclab

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
        sys.modules.pop("ref2act.common.observation_spec", None)
        sys.modules.pop("ref2act.envs.motion_tracking.observation", None)
        return (
            importlib.import_module("ref2act.common.observation_spec"),
            importlib.import_module("ref2act.envs.motion_tracking.observation"),
        )
    finally:
        for module_name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        if previous_root is sentinel:
            sys.modules.pop("isaaclab", None)
        else:
            sys.modules["isaaclab"] = previous_root
        for attr_name, previous_attr in previous_attrs.items():
            if previous_attr is sentinel:
                if hasattr(isaaclab, attr_name):
                    delattr(isaaclab, attr_name)
            else:
                setattr(isaaclab, attr_name, previous_attr)


def _make_observation(observation_mod, *, num_envs: int = 1, num_joints: int = 2, num_keys: int = 2, add_noise: bool = False):
    return observation_mod.Observation(
        spec=observation_mod.default_training_observation_spec(add_noise=add_noise),
        layout=observation_mod.ObservationLayout(
            joint_dim=num_joints,
            action_dim=num_joints,
            key_body_count=num_keys,
        ),
        num_envs=num_envs,
        device=torch.device("cpu"),
        anchor_body_index=0,
        key_body_indices=list(range(num_keys)),
    )


def test_default_observation_keeps_privileged_observation_clean() -> None:
    _, observation_mod = _load_modules()
    observation = _make_observation(observation_mod, add_noise=True)

    joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    joint_vel = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    anchor_ang_vel = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)
    target_anchor_lin_vel = torch.tensor([[1.2, 1.3, 1.4]], dtype=torch.float32)
    target_anchor_ang_vel = torch.tensor([[1.5, 1.6, 1.7]], dtype=torch.float32)
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
        body_linear_velocities=torch.stack(
            (
                target_anchor_lin_vel,
                torch.zeros((1, 3), dtype=torch.float32),
            ),
            dim=1,
        ),
        body_angular_velocities=torch.stack(
            (
                target_anchor_ang_vel,
                torch.zeros((1, 3), dtype=torch.float32),
            ),
            dim=1,
        ),
    )
    scene = types.SimpleNamespace(env_origins=torch.zeros((1, 3), dtype=torch.float32))

    torch.manual_seed(0)
    obs = observation.get_default_observation(robot, reference_motion, scene, last_action)

    assert torch.allclose(robot.data.joint_pos, joint_pos)
    assert torch.allclose(robot.data.joint_vel, joint_vel)
    assert torch.allclose(robot.data.body_ang_vel_w[:, 0], anchor_ang_vel)

    robot_obs = obs["robot"]
    privilege_obs = obs["privilege"]
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    assert not torch.allclose(robot_obs[0, 0:6], identity_6d)
    assert not torch.allclose(robot_obs[0, 6:9], anchor_ang_vel[0])
    assert not torch.allclose(robot_obs[0, 9:11], joint_pos[0])
    assert not torch.allclose(robot_obs[0, 11:13], joint_vel[0])

    num_joints = joint_pos.shape[1]
    num_keys = 2
    motion_anchor_pos_start = 2 * num_joints
    motion_anchor_ori_start = motion_anchor_pos_start + 3
    body_pos_start = motion_anchor_ori_start + 6
    body_ori_start = body_pos_start + num_keys * 3
    anchor_lin_start = body_ori_start + num_keys * 6
    anchor_ang_start = anchor_lin_start + 3
    joint_pos_start = anchor_ang_start + 3
    joint_vel_start = joint_pos_start + num_joints
    last_action_start = joint_vel_start + num_joints

    assert torch.allclose(privilege_obs[0, :num_joints], torch.zeros(num_joints))
    assert torch.allclose(privilege_obs[0, num_joints:2 * num_joints], torch.zeros(num_joints))
    assert torch.allclose(
        privilege_obs[0, motion_anchor_pos_start:motion_anchor_pos_start + 3],
        torch.zeros(3),
    )
    assert torch.allclose(
        privilege_obs[0, motion_anchor_ori_start:motion_anchor_ori_start + 6],
        identity_6d,
    )
    assert torch.allclose(
        privilege_obs[0, body_pos_start:body_pos_start + num_keys * 3],
        torch.tensor([0.0, 0.0, 0.0, 0.2, 0.3, 0.4], dtype=torch.float32),
    )
    assert torch.allclose(
        privilege_obs[0, body_ori_start:body_ori_start + num_keys * 6],
        identity_6d.repeat(num_keys),
    )
    assert torch.allclose(privilege_obs[0, anchor_lin_start:anchor_lin_start + 3], robot.data.body_lin_vel_w[0, 0])
    assert torch.allclose(privilege_obs[0, anchor_ang_start:anchor_ang_start + 3], anchor_ang_vel[0])
    assert torch.allclose(privilege_obs[0, joint_pos_start:joint_pos_start + num_joints], joint_pos[0])
    assert torch.allclose(privilege_obs[0, joint_vel_start:joint_vel_start + num_joints], joint_vel[0])
    assert torch.allclose(privilege_obs[0, last_action_start:last_action_start + num_joints], last_action[0])


def test_window_observation_reset_fills_history_oldest_to_newest() -> None:
    common_obs_mod, observation_mod = _load_modules()
    observation = observation_mod.Observation(
        spec=common_obs_mod.ObservationSpec(
            groups=(
                common_obs_mod.ObservationGroupSpec(
                    name="robot",
                    terms=(
                        common_obs_mod.ObservationTermSpec(
                            id="joint_pos_history",
                            type="joint_pos",
                            window_length=3,
                        ),
                    ),
                ),
            )
        ),
        layout=common_obs_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=1),
        num_envs=1,
        device=torch.device("cpu"),
        anchor_body_index=0,
        key_body_indices=[0],
    )

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            joint_vel=torch.zeros((1, 2), dtype=torch.float32),
            body_pos_w=torch.zeros((1, 1, 3), dtype=torch.float32),
            body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            body_lin_vel_w=torch.zeros((1, 1, 3), dtype=torch.float32),
            body_ang_vel_w=torch.zeros((1, 1, 3), dtype=torch.float32),
            GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        )
    )
    reference_motion = types.SimpleNamespace(
        joint_pos=torch.zeros((1, 2), dtype=torch.float32),
        joint_vel=torch.zeros((1, 2), dtype=torch.float32),
        body_positions=torch.zeros((1, 1, 3), dtype=torch.float32),
        body_quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
        body_linear_velocities=torch.zeros((1, 1, 3), dtype=torch.float32),
        body_angular_velocities=torch.zeros((1, 1, 3), dtype=torch.float32),
    )
    scene = types.SimpleNamespace(env_origins=torch.zeros((1, 3), dtype=torch.float32))
    last_action = torch.zeros((1, 2), dtype=torch.float32)

    observation.reset(torch.tensor([0]), robot, reference_motion, scene, last_action)
    initial_obs = observation.get_default_observation(robot, reference_motion, scene, last_action)["robot"]
    assert torch.allclose(initial_obs, torch.tensor([[0.1, 0.2, 0.1, 0.2, 0.1, 0.2]], dtype=torch.float32))

    robot.data.joint_pos[:] = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
    next_obs = observation.get_default_observation(robot, reference_motion, scene, last_action)["robot"]
    assert torch.allclose(next_obs, torch.tensor([[0.1, 0.2, 0.1, 0.2, 0.3, 0.4]], dtype=torch.float32))


def test_custom_observation_term_registration_supports_reordered_term_sets_and_auto_dims() -> None:
    common_obs_mod, _ = _load_modules()

    class BonusObservationTerm:
        type_name = "bonus"

        def compute(self, context, spec):
            return context.extras["bonus"]

        def dimension(self, layout, spec):
            return 2

    common_obs_mod.register_observation_term(BonusObservationTerm())

    spec = common_obs_mod.ObservationSpec(
        groups=(
            common_obs_mod.ObservationGroupSpec(
                name="robot",
                terms=(
                    common_obs_mod.ObservationTermSpec(id="bonus", type="bonus"),
                    common_obs_mod.ObservationTermSpec(id="joint_pos", type="joint_pos"),
                ),
            ),
        )
    )
    layout = common_obs_mod.ObservationLayout(joint_dim=2, action_dim=2, key_body_count=0)
    composer = common_obs_mod.ObservationComposer(spec=spec, layout=layout, num_envs=1, device="cpu")
    outputs = composer.compose(
        common_obs_mod.ObservationContext(
            joint_pos=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            extras={"bonus": torch.tensor([[3.0, 4.0]], dtype=torch.float32)},
        )
    )

    assert torch.allclose(outputs["robot"], torch.tensor([[3.0, 4.0, 1.0, 2.0]], dtype=torch.float32))
    assert spec.describe(layout).group_dims == {"robot": 4}

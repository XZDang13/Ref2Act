import types

import numpy as np
import torch

from ref2act.bridges.mujoco.env import wxyz_to_xyzw_np, xyzw_to_wxyz_np
from ref2act.bridges.mujoco.observation import IsaacLabMujocoObservation, default_mujoco_observation_spec
from ref2act.common.observation_spec import ObservationComposer, ObservationLayout
from ref2act.envs.motion_tracking.observation import build_observation_context, default_training_observation_spec
from ref2act.envs.motion_tracking.types import MotionState


def _state(*, joint_offset: float, anchor_pos: tuple[float, float, float], yaw_quat: torch.Tensor) -> MotionState:
    joint_pos = torch.arange(23, dtype=torch.float32).unsqueeze(0) + joint_offset
    empty3 = torch.empty((1, 0, 3), dtype=torch.float32)
    return MotionState(
        joint_pos=joint_pos,
        joint_vel=joint_pos * 0.1,
        anchor_pos=torch.tensor([anchor_pos], dtype=torch.float32),
        anchor_quat=yaw_quat.reshape(1, 4),
        anchor_lin_vel=torch.tensor([[0.1, 0.2, 0.3]]),
        anchor_ang_vel=torch.tensor([[0.4, 0.5, 0.6]]),
        key_pos=empty3,
        key_quat=torch.empty((1, 0, 4), dtype=torch.float32),
        key_lin_vel=empty3,
        key_ang_vel=empty3,
    )


def test_mujoco_quaternion_boundary_round_trip() -> None:
    xyzw = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
    assert np.array_equal(wxyz_to_xyzw_np(xyzw_to_wxyz_np(xyzw)), xyzw)
    assert np.array_equal(xyzw_to_wxyz_np(np.asarray([0.0, 0.0, 0.0, 1.0])), [1.0, 0.0, 0.0, 0.0])


def test_mujoco_policy_spec_is_current_107_dim_without_privilege() -> None:
    spec = default_mujoco_observation_spec()
    description = spec.describe(ObservationLayout(joint_dim=23, action_dim=23, key_body_count=0))
    assert description.group_dims == {"motion": 29, "robot": 78}
    assert description.total_dim == 107
    assert all(group.name != "privilege" for group in spec.groups)


def test_mujoco_policy_observation_matches_noise_free_training_groups() -> None:
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    robot_state = _state(joint_offset=0.0, anchor_pos=(0.0, 0.0, 1.0), yaw_quat=identity)
    reference_state = _state(joint_offset=1.0, anchor_pos=(0.2, -0.1, 1.1), yaw_quat=identity)
    context = build_observation_context(
        robot_state,
        reference_state,
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.zeros((1, 23)),
    )
    layout = ObservationLayout(joint_dim=23, action_dim=23, key_body_count=0)

    fake_env = types.SimpleNamespace(action_offset=torch.zeros(23))
    mujoco_builder = IsaacLabMujocoObservation()
    mujoco_policy = mujoco_builder.get_policy_observation(fake_env, context)

    training_spec = default_training_observation_spec(add_noise=False)
    training_policy_spec = type(training_spec)(
        groups=tuple(group for group in training_spec.groups if group.name in {"motion", "robot"})
    )
    composer = ObservationComposer(spec=training_policy_spec, layout=layout, num_envs=1, device="cpu")
    training_groups = composer.compose(context)
    expected = torch.cat((training_groups["motion"], training_groups["robot"]), dim=-1).squeeze(0)

    assert mujoco_policy.shape == (107,)
    assert torch.allclose(mujoco_policy, expected)
    assert torch.isfinite(mujoco_policy).all()

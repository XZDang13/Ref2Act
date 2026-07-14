import time
import numpy as np
import torch

import mujoco

from ref2act.assets import scene_asset_path
from ref2act.common.math import quat_apply_inverse
from ref2act.bridges.mujoco.action import (
    IsaacLabMujocoAction,
    MujocoActionBuilder,
    MujocoActionContext,
    MujocoActionOutput,
)
from ref2act.motion.library import MotionLib
from ref2act.bridges.mujoco.observation import (
    IsaacLabMujocoObservation,
    MujocoObservationBuilder,
    MujocoObservationContext,
)
from ref2act.envs.motion_tracking.observation import build_observation_context
from ref2act.envs.motion_tracking.types import MotionState

mujoco_env_xml = str(scene_asset_path("g1", "scene.xml"))


def wxyz_to_xyzw_np(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[[1, 2, 3, 0]]


def xyzw_to_wxyz_np(q: np.ndarray) -> np.ndarray:
    return np.asarray(q)[[3, 0, 1, 2]]


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_inv_np(q: np.ndarray) -> np.ndarray:
    return quat_conjugate_np(q) / max(float(np.dot(q, q)), 1.0e-8)


def quat_apply_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q[:3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + q[3] * t + np.cross(q_xyz, t)


def normalize_quat_np(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-8:
        raise ValueError("Encountered a zero-norm quaternion while solving MuJoCo root state.")
    return q / norm


class MujocoEnv:
    def __init__(
        self,
        simulation_dt: float,
        decimation: float,
        kp: torch.Tensor,
        kd: torch.Tensor,
        effort_limits: torch.Tensor,
        joint_pos_limits: torch.Tensor,
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        expert_motion_file: str,
        root_link_name: str = "torso_link",
        anchor_body_name: str = "torso_link",
        render: bool = False,
        action_mode: str = "absolute",
        observation_builder: MujocoObservationBuilder | None = None,
        action_builder: MujocoActionBuilder | None = None,
    ):
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_env_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = simulation_dt
        self.mj_viewer = None
        self.render = render
        if self.render:
            import mujoco_viewer.mujoco_viewer as mjv

            self.mj_viewer = mjv.MujocoViewer(
                self.mj_model,
                self.mj_data,
                width=1400,
                height=1200,
                hide_menus=True,
            )

        self.motion_lib = MotionLib(expert_motion_file)
        (
            self.free_root_body_name,
            self.free_root_body_id,
            self.root_body_index,
            self.root_body_id,
            self.anchor_body_index,
            self.anchor_body_id,
        ) = self._resolve_body_ids(root_link_name, anchor_body_name)
        self._configure_follow_camera()
        self.motion_id = torch.zeros(1, dtype=torch.long)

        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0]).float()
        self.mujoco2isaac = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
        self.isaac2mujoco = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

        self.kp = torch.as_tensor(kp, dtype=torch.float32, device="cpu").clone()
        self.kd = torch.as_tensor(kd, dtype=torch.float32, device="cpu").clone()
        self.effort_limits = torch.as_tensor(effort_limits, dtype=torch.float32, device="cpu").clone()
        joint_pos_limits = torch.as_tensor(joint_pos_limits, dtype=torch.float32, device="cpu").clone()
        self.joint_pos_limits_lower = joint_pos_limits[:, 0]
        self.joint_pos_limits_upper = joint_pos_limits[:, 1]

        self.action_offset = torch.as_tensor(action_offset, dtype=torch.float32, device="cpu").clone()
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device="cpu").clone()
        self.action_mode = self._normalize_action_mode(action_mode)
        self.previous_action = torch.zeros_like(self.action_offset)
        self.observation_builder = observation_builder or IsaacLabMujocoObservation()
        self.action_builder = action_builder or IsaacLabMujocoAction()

        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.policy_dt = simulation_dt * decimation

        self.n_steps = 0

    def _resolve_motion_body_index(self, body_name: str, *, role: str) -> int:
        try:
            return self.motion_lib.body_names.index(body_name)
        except ValueError as exc:
            raise ValueError(f"{role} body '{body_name}' was not found in motion_lib.body_names.") from exc

    def _resolve_model_body_id(self, body_name: str, *, role: str) -> int:
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"{role} body '{body_name}' was not found in MuJoCo model body names.")
        return int(body_id)

    def _resolve_free_root_body(self) -> tuple[str, int]:
        free_joint_ids = np.flatnonzero(self.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(
                f"MuJoCo bridge expects exactly one free joint body, found {len(free_joint_ids)} free joints."
            )

        free_joint_id = int(free_joint_ids[0])
        free_root_body_id = int(self.mj_model.jnt_bodyid[free_joint_id])
        free_root_body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, free_root_body_id)
        if free_root_body_name is None:
            raise ValueError(f"MuJoCo body id {free_root_body_id} has no registered name.")
        return free_root_body_name, free_root_body_id

    def _resolve_body_ids(
        self,
        root_link_name: str,
        anchor_body_name: str,
    ) -> tuple[str, int, int, int, int, int]:
        free_root_body_name, free_root_body_id = self._resolve_free_root_body()
        root_body_index = self._resolve_motion_body_index(root_link_name, role="root")
        root_body_id = self._resolve_model_body_id(root_link_name, role="root")
        anchor_body_index = self._resolve_motion_body_index(anchor_body_name, role="anchor")
        anchor_body_id = self._resolve_model_body_id(anchor_body_name, role="anchor")
        return (
            free_root_body_name,
            free_root_body_id,
            root_body_index,
            root_body_id,
            anchor_body_index,
            anchor_body_id,
        )

    def _configure_follow_camera(self) -> None:
        if self.mj_viewer is None:
            return

        camera = self.mj_viewer.cam
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self.root_body_id
        camera.fixedcamid = -1
        camera.distance = 4.0
        camera.azimuth = -140.0
        camera.elevation = -20.0

    def _normalize_action_mode(self, action_mode: object | str) -> str:
        if isinstance(action_mode, str):
            normalized = action_mode
        elif hasattr(action_mode, "name"):
            normalized = str(action_mode.name)
        else:
            normalized = str(action_mode).split(".")[-1]

        normalized = normalized.replace("-", "_").lower()
        if normalized in {"currentresidual", "current_residual"}:
            return "current_residual"
        return normalized

    def _get_body_world_pose(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.mj_data.xpos[body_id], dtype=np.float64).copy(),
            wxyz_to_xyzw_np(np.asarray(self.mj_data.xquat[body_id], dtype=np.float64)).copy(),
        )

    def _get_body_world_quat(self, body_id: int) -> np.ndarray:
        return wxyz_to_xyzw_np(np.asarray(self.mj_data.xquat[body_id], dtype=np.float64)).copy()

    def _get_body_world_twist(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
        angular_velocity = velocity[:3].astype(np.float32, copy=False)
        linear_velocity = velocity[3:].astype(np.float32, copy=False)
        return angular_velocity, linear_velocity

    def get_projected_gravity(self):
        anchor_quat_w = torch.from_numpy(
            self._get_body_world_quat(self.anchor_body_id).astype(np.float32, copy=False)
        ).float()
        projected_gravity = quat_apply_inverse(anchor_quat_w, self.gravity_vector).float()

        return projected_gravity

    def get_anchor_ang_vel_b(self):
        anchor_quat_w = torch.from_numpy(
            self._get_body_world_quat(self.anchor_body_id).astype(np.float32, copy=False)
        ).float()
        anchor_ang_vel_w = torch.from_numpy(self._get_body_world_twist(self.anchor_body_id)[0]).float()
        return quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w).float()

    def get_joint_pos(self):
        joint_pos = torch.from_numpy(self.mj_data.qpos[7:]).float()[self.mujoco2isaac]
        return joint_pos

    def get_joint_vel(self):
        joint_vel = torch.from_numpy(self.mj_data.qvel[6:]).float()[self.mujoco2isaac]

        return joint_vel

    def get_motion_command(self, times):
        motion_ids = torch.full(times.shape, int(self.motion_id.item()), dtype=torch.long)
        reference_motion = self.motion_lib.sample_motion(motion_ids=motion_ids, times=times)

        joint_pos = reference_motion["joint_pos"].squeeze(0)
        joint_vel = reference_motion["joint_vel"].squeeze(0)
        body_quat = reference_motion["body_quaternions"].squeeze(0)
        body_linear_velocities = reference_motion["body_linear_velocities"].squeeze(0)
        body_angular_velocities = reference_motion["body_angular_velocities"].squeeze(0)
        anchor_quat = body_quat[self.anchor_body_index]
        anchor_lin_vel = body_linear_velocities[self.anchor_body_index]
        anchor_ang_vel_b = quat_apply_inverse(anchor_quat, body_angular_velocities[self.anchor_body_index]).float()

        projected_gravity = quat_apply_inverse(anchor_quat, self.gravity_vector).float()

        return joint_pos, joint_vel, projected_gravity, anchor_lin_vel.float(), anchor_ang_vel_b

    def _get_observation_builder(self) -> MujocoObservationBuilder:
        builder = getattr(self, "observation_builder", None)
        if builder is None:
            builder = IsaacLabMujocoObservation()
            self.observation_builder = builder
        return builder

    def _get_action_builder(self) -> MujocoActionBuilder:
        builder = getattr(self, "action_builder", None)
        if builder is None:
            builder = IsaacLabMujocoAction()
            self.action_builder = builder
        return builder

    def _build_observation_context(self, advance_time: bool = True) -> MujocoObservationContext:
        if advance_time:
            self.times += self.policy_dt

        current_duration = self.motion_lib.get_duration(self.motion_id).squeeze(0)
        if self.times.item() > current_duration.item():
            self.times = torch.zeros(1)
            self.previous_action[:] = 0.0

        reference = self.motion_lib.sample_motion(motion_ids=self.motion_id, times=self.times)
        anchor_pos_np, anchor_quat_np = self._get_body_world_pose(self.anchor_body_id)
        anchor_ang_vel_np, anchor_lin_vel_np = self._get_body_world_twist(self.anchor_body_id)
        empty_pos = torch.empty((1, 0, 3), dtype=torch.float32)
        empty_quat = torch.empty((1, 0, 4), dtype=torch.float32)
        robot_state = MotionState(
            joint_pos=self.get_joint_pos().unsqueeze(0),
            joint_vel=self.get_joint_vel().unsqueeze(0),
            anchor_pos=torch.from_numpy(anchor_pos_np.astype(np.float32)).unsqueeze(0),
            anchor_quat=torch.from_numpy(anchor_quat_np.astype(np.float32)).unsqueeze(0),
            anchor_lin_vel=torch.from_numpy(anchor_lin_vel_np).unsqueeze(0),
            anchor_ang_vel=torch.from_numpy(anchor_ang_vel_np).unsqueeze(0),
            key_pos=empty_pos,
            key_quat=empty_quat,
            key_lin_vel=empty_pos,
            key_ang_vel=empty_pos,
        )
        reference_state = MotionState(
            joint_pos=reference["joint_pos"],
            joint_vel=reference["joint_vel"],
            anchor_pos=reference["body_positions"][:, self.anchor_body_index],
            anchor_quat=reference["body_quaternions"][:, self.anchor_body_index],
            anchor_lin_vel=reference["body_linear_velocities"][:, self.anchor_body_index],
            anchor_ang_vel=reference["body_angular_velocities"][:, self.anchor_body_index],
            key_pos=empty_pos,
            key_quat=empty_quat,
            key_lin_vel=empty_pos,
            key_ang_vel=empty_pos,
        )
        return build_observation_context(
            robot_state,
            reference_state,
            self.gravity_vector.unsqueeze(0),
            self.previous_action.unsqueeze(0),
        )

    def get_obs_dict(self, advance_time: bool = True) -> dict[str, torch.Tensor]:
        context = self._build_observation_context(advance_time=advance_time)
        return self._get_observation_builder().get_default_observation(self, context)

    def get_obs(self, advance_time: bool = True) -> torch.Tensor:
        context = self._build_observation_context(advance_time=advance_time)
        return self._get_observation_builder().get_policy_observation(self, context)

    def _solve_free_joint_state_from_root_reference(
        self,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        root_linear_vel: np.ndarray,
        root_angular_vel: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.mj_data.qpos[:7] = 0.0
        self.mj_data.qpos[3] = 1.0
        self.mj_data.qpos[7:] = joint_positions
        self.mj_data.qvel[:6] = 0.0
        self.mj_data.qvel[6:] = joint_velocities
        mujoco.mj_forward(self.mj_model, self.mj_data)

        free_root_pos, free_root_quat = self._get_body_world_pose(self.free_root_body_id)
        current_root_pos, current_root_quat = self._get_body_world_pose(self.root_body_id)
        free_to_root_pos = quat_apply_np(quat_inv_np(free_root_quat), current_root_pos - free_root_pos)
        free_to_root_quat = quat_mul_np(quat_inv_np(free_root_quat), current_root_quat)

        solved_free_quat = normalize_quat_np(quat_mul_np(root_quat, quat_inv_np(free_to_root_quat)))
        solved_free_pos = root_pos - quat_apply_np(solved_free_quat, free_to_root_pos)

        self.mj_data.qpos[:3] = solved_free_pos
        self.mj_data.qpos[3:7] = xyzw_to_wxyz_np(solved_free_quat)
        self.mj_data.qvel[:6] = 0.0
        self.mj_data.qvel[6:] = joint_velocities
        mujoco.mj_forward(self.mj_model, self.mj_data)

        relative_root_ang_vel, relative_root_lin_vel = self._get_body_world_twist(self.root_body_id)
        free_root_world_pos, _ = self._get_body_world_pose(self.free_root_body_id)
        current_root_world_pos, _ = self._get_body_world_pose(self.root_body_id)
        root_offset_world = current_root_world_pos - free_root_world_pos

        solved_free_ang_vel = np.asarray(root_angular_vel, dtype=np.float64) - np.asarray(
            relative_root_ang_vel,
            dtype=np.float64,
        )
        solved_free_lin_vel = np.asarray(root_linear_vel, dtype=np.float64) - np.asarray(
            relative_root_lin_vel,
            dtype=np.float64,
        ) - np.cross(solved_free_ang_vel, root_offset_world)

        free_joint_pose = np.concatenate([solved_free_pos, xyzw_to_wxyz_np(solved_free_quat)]).astype(
            np.float32, copy=False
        )
        free_joint_velocity = np.concatenate([solved_free_lin_vel, solved_free_ang_vel]).astype(
            np.float32,
            copy=False,
        )
        return free_joint_pose, free_joint_velocity

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.previous_action[:] = 0.0
        self.motion_id.zero_()
        self.times = torch.zeros(1)

        reference_motion = self.motion_lib.sample_motion(motion_ids=self.motion_id, times=self.times)

        joint_positions = reference_motion["joint_pos"].squeeze(0).numpy()[self.isaac2mujoco]
        joint_velocities = reference_motion["joint_vel"].squeeze(0).numpy()[self.isaac2mujoco]
        body_positions = reference_motion["body_positions"].squeeze(0).numpy()
        body_rotations = reference_motion["body_quaternions"].squeeze(0).numpy()
        body_linear_velocities = reference_motion["body_linear_velocities"].squeeze(0).numpy()
        body_angular_velocities = reference_motion["body_angular_velocities"].squeeze(0).numpy()

        root_pos = body_positions[self.root_body_index].copy()
        root_pos[2] += 0.05
        root_quat = body_rotations[self.root_body_index].copy()
        root_linear_vel = body_linear_velocities[self.root_body_index].copy()
        root_ang_vel = body_angular_velocities[self.root_body_index].copy()

        free_joint_pose, free_joint_velocity = self._solve_free_joint_state_from_root_reference(
            root_pos=root_pos,
            root_quat=root_quat,
            root_linear_vel=root_linear_vel,
            root_angular_vel=root_ang_vel,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )

        self.mj_data.qpos[:7] = free_joint_pose
        self.mj_data.qpos[7:] = joint_positions
        self.mj_data.qvel[:6] = free_joint_velocity
        self.mj_data.qvel[6:] = joint_velocities

        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.mj_viewer is not None and self.mj_viewer.is_alive:
            self.mj_viewer.render()
        else:
            # viewer was closed manually -> stop touching it
            self.mj_viewer = None

        observation_builder = self._get_observation_builder()
        if hasattr(self.motion_lib, "get_duration"):
            context = self._build_observation_context(advance_time=False)
            if hasattr(observation_builder, "reset"):
                observation_builder.reset(self, context)
            obs = observation_builder.get_policy_observation(self, context)
        else:
            obs = self.get_obs(advance_time=False)

        self.target_pos = reference_motion["joint_pos"].squeeze(0).clone()

        return obs

    def _apply_actions(self):
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()

        # PD control
        tau = self.kp * (self.target_pos - joint_pos) - self.kd * joint_vel

        tau_clipped = torch.clip(tau, -self.effort_limits, self.effort_limits)
        tau_clipped = tau_clipped[self.isaac2mujoco]

        self.mj_data.ctrl[:] = tau_clipped.numpy()

    def _build_action_context(self, actions: torch.Tensor) -> MujocoActionContext:
        normalized_action = torch.as_tensor(actions, dtype=torch.float32, device="cpu")
        return MujocoActionContext(
            raw_action=normalized_action,
            action_mode=getattr(self, "action_mode", "absolute"),
            action_scale=self.action_scale,
            action_offset=self.action_offset,
            joint_pos_limits_lower=self.joint_pos_limits_lower,
            joint_pos_limits_upper=self.joint_pos_limits_upper,
            current_joint_pos_loader=self.get_joint_pos,
            reference_joint_pos_loader=lambda: self.get_motion_command(self.times)[0],
        )

    def process_action(self, actions: torch.Tensor) -> MujocoActionOutput:
        context = self._build_action_context(actions)
        return self._get_action_builder().process_action(self, context)

    def _compute_target_pos(self, actions: torch.Tensor) -> torch.Tensor:
        return self.process_action(actions).target_joint_pos

    def step(self, actions):
        step_start_time = time.perf_counter()
        action_output = self.process_action(actions)
        self.previous_action.copy_(action_output.applied_action)
        self.target_pos = action_output.target_joint_pos

        for _ in range(self.decimation):
            self._apply_actions()
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None and self.mj_viewer.is_alive:
            self.mj_viewer.render()
        else:
            # viewer was closed manually -> stop touching it
            self.mj_viewer = None

        obs = self.get_obs(advance_time=True)

        time_until_next_step = self.policy_dt - (time.perf_counter() - step_start_time)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        self.n_steps += 1

        return obs

    def close(self):
        if self.mj_viewer is not None:
            try:
                if self.mj_viewer.is_alive:
                    self.mj_viewer.close()
            finally:
                self.mj_viewer = None

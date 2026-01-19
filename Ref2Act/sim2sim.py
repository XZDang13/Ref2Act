from importlib import resources as importlib_resources
from pathlib import Path
import time
import numpy as np
import torch

import mujoco
import mujoco_viewer.mujoco_viewer as mjv

from .motion_lib import MotionLib

def _default_assets_root() -> Path:
    try:
        assets_root = Path(importlib_resources.files("Ref2Act") / "assets")
        if assets_root.exists():
            return assets_root
    except Exception:
        pass
    return Path(__file__).resolve().parent.parent / "assets"

_assets_root = _default_assets_root()
mujoco_env_xml = str(
    _assets_root
    / "G1"
    / "scene.xml"
)

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion [w, x, y, z]
        v (torch.Tensor): Vector to rotate

    Returns:
        torch.Tensor: Rotated vector
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c

class MujocoEnv:
    def __init__(self, simulation_dt:float, decimation:float,
                 kp:torch.Tensor, kd:torch.Tensor, effort_limits: torch.Tensor,
                 action_offset: torch.Tensor, action_scale: torch.Tensor,  expert_motion_file:str,
                 root_name:str="pelvis", render:bool=False):
        
        self.mj_model = mujoco.MjModel.from_xml_path(mujoco_env_xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = simulation_dt
        self.mj_viewer = None
        self.render = render
        if self.render:
            self.mj_viewer = mjv.MujocoViewer(
                                self.mj_model,
                                self.mj_data,
                                width=1400,
                                height=1200,
                                hide_menus=True,
                            )

        self.motion_lib = MotionLib(expert_motion_file)
        self.root_index = self.motion_lib.body_names.index(root_name)

        self.gravity_vector = torch.tensor([0.0, 0.0, -1.0]).float()
        self.previous_action = torch.zeros(23).float()

        self.mujoco2isaac = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
        self.isaac2mujoco = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

        self.kp = kp
        self.kd = kd
        self.effort_limits = effort_limits

        self.action_offset = action_offset
        self.action_scale = action_scale

        self.simulation_dt = simulation_dt
        self.decimation = decimation
        self.policy_dt = simulation_dt * decimation

        self.n_steps = 0

        for i in range(self.mj_model.njnt):
            print(i, mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i))

    def get_projected_gravity(self):
        base_quat = torch.from_numpy(self.mj_data.qpos[3:7]).float()
        projected_gravity = quat_rotate_inverse(base_quat, self.gravity_vector).float()

        return projected_gravity
    
    def get_base_ang_vel(self):
        base_ang_vel = torch.from_numpy(self.mj_data.qvel[3:6]).float()
        return base_ang_vel
    
    def get_joint_pos(self):
        joint_pos = torch.from_numpy(self.mj_data.qpos[7:]).float()[self.mujoco2isaac]
        return joint_pos
    
    def get_joint_vel(self):
        joint_vel = torch.from_numpy(self.mj_data.qvel[6:]).float()[self.mujoco2isaac]

        return joint_vel
    
    def get_motion_command(self, times):
        reference_motion = self.motion_lib.sample_motion(times)

        joint_pos = reference_motion["joint_pos"].squeeze(0)
        joint_vel = reference_motion["joint_vel"].squeeze(0)
        body_quat = reference_motion["body_quaternions"].squeeze(0)
        root_quat = body_quat[self.root_index]

        projected_gravity = quat_rotate_inverse(root_quat, self.gravity_vector).float()

        return joint_pos, joint_vel, projected_gravity

    def get_obs(self):
        
        self.times += self.policy_dt

        if self.times > self.motion_lib.duration:
            self.times = torch.zeros(1)
            self.previous_action[:] = 0.0

        target_joint_pos, target_joint_vel, target_projected_gravity = self.get_motion_command(self.times)

        projected_gravity = self.get_projected_gravity()
        base_ang_vel = self.get_base_ang_vel()
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()

        #print(target_joint_pos)
        #print(target_joint_vel)
        #print(target_projected_gravity)
        #print(projected_gravity)
        #print(base_ang_vel)
        #print(joint_pos)
        #print(joint_vel)
        #print(self.previous_action)
        #print("-----------")

        return torch.cat([
            target_joint_pos,
            target_joint_vel,
            target_projected_gravity,
            projected_gravity,
            base_ang_vel,
            joint_pos,
            joint_vel,
            self.previous_action,
        ])

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.previous_action[:] = 0.0
        self.times = torch.zeros(1)
        
        reference_motion = self.motion_lib.sample_motion(self.times)

        joint_positions = reference_motion["joint_pos"].squeeze(0).numpy()[self.isaac2mujoco]
        joint_velocities = reference_motion["joint_vel"].squeeze(0).numpy()[self.isaac2mujoco]
        body_positions = reference_motion["body_positions"].squeeze(0).numpy()
        body_rotations = reference_motion["body_quaternions"].squeeze(0).numpy()
        body_linear_velocities = reference_motion["body_linear_velocities"].squeeze(0).numpy()
        body_angular_velocities = reference_motion["body_angular_velocities"].squeeze(0).numpy()

        root_pos = body_positions[self.root_index]
        #root_pos[2] += 0.50
        root_quat = body_rotations[self.root_index]
        root_linear_vel = body_linear_velocities[self.root_index]
        root_ang_vel = body_angular_velocities[self.root_index]

        self.mj_data.qpos[0] = 0.0
        self.mj_data.qpos[1] = 0.0
        self.mj_data.qpos[2] = root_pos[2]
        self.mj_data.qpos[3:7] = root_quat
        self.mj_data.qpos[7:] = joint_positions

        self.mj_data.qvel[:3] = root_linear_vel
        self.mj_data.qvel[3:6] = root_ang_vel
        self.mj_data.qvel[6:] = joint_velocities

        mujoco.mj_forward(self.mj_model, self.mj_data)

        if self.mj_viewer is not None and self.mj_viewer.is_alive:
            self.mj_viewer.render()
        else:
            # viewer was closed manually -> stop touching it
            self.mj_viewer = None

        obs = self.get_obs()

        self.target_pos = reference_motion["joint_pos"].squeeze(0)

        return obs
    
    def _apply_actions(self):
        
        joint_pos = self.get_joint_pos()
        joint_vel = self.get_joint_vel()

        # PD control
        tau = self.kp * (self.target_pos - joint_pos) - self.kd * joint_vel
        #print(tau)
        tau_clipped = torch.clip(tau, -self.effort_limits, self.effort_limits)
        tau_clipped = tau_clipped[self.isaac2mujoco]

        self.mj_data.ctrl[:] = tau_clipped.numpy()

    def step(self, actions):
        step_start_time = time.perf_counter()
        self.previous_action = actions.clone()
        self.target_pos = actions * self.action_scale + self.action_offset

        print(self.target_pos)

        for _ in range(self.decimation):
            self._apply_actions()
            mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None and self.mj_viewer.is_alive:
            self.mj_viewer.render()
        else:
            # viewer was closed manually -> stop touching it
            self.mj_viewer = None

        print(self.get_joint_pos())
        obs = self.get_obs()

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

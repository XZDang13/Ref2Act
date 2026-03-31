from __future__ import annotations

import pickle
from pathlib import Path

import torch

from ref2act.motion.processing.resample import _linear_derivative, _so3_derivative

from ..smoothing import DEFAULT_SMOOTHING_PROFILE, smooth_motion_trajectory


class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


class GMRMotionData:
    def __init__(
        self,
        file: str,
        device: torch.device,
        joint_order: list[str],
        height_offset: float = 0.0,
        smooth_motion: bool = False,
        smoothing_profile: str = DEFAULT_SMOOTHING_PROFILE,
    ):
        with open(file, "rb") as handle:
            motion_data = NumpyCompatUnpickler(handle).load()

        self.device = device
        self.joint_order = joint_order
        self.fps = round(motion_data["fps"])
        self.offset = torch.zeros(3, device=self.device)
        self.offset[-1] += height_offset
        self.root_pos = torch.from_numpy(motion_data["root_pos"]).float().to(self.device) + self.offset
        self.root_rot = torch.from_numpy(motion_data["root_rot"]).float().to(self.device)
        self.root_rot = self.root_rot[:, [3, 0, 1, 2]]
        self.joint_pos = torch.from_numpy(motion_data["dof_pos"]).float().to(self.device)
        self.num_frames = self.joint_pos.size(0)
        if smooth_motion:
            self.root_pos, self.root_rot, self.joint_pos = smooth_motion_trajectory(
                self.root_pos,
                self.root_rot,
                self.joint_pos,
                fps=float(self.fps),
                profile=smoothing_profile,
            )
            print(f"[INFO]: Enabled motion smoothing with profile '{smoothing_profile}'.")
        self.render_interval = 1
        self.physic_dt = 1 / (self.render_interval * self.fps)
        self.current_step = 0

        self.root_lin_vel = _linear_derivative(self.root_pos, self.physic_dt)
        self.root_ang_vel = _so3_derivative(self.root_rot, self.physic_dt)
        self.joint_vel = _linear_derivative(self.joint_pos, self.physic_dt)

    def get_init_state(self):
        return (
            self.root_pos[0:1],
            self.root_rot[0:1],
            self.root_lin_vel[0:1],
            self.root_ang_vel[0:1],
            self.joint_pos[0:1],
            self.joint_vel[0:1],
        )

    def set_root_height(self, height_offset: float):
        offset = torch.zeros(3, device=self.device)
        offset[-1] += height_offset
        self.root_pos += offset

    def get_next_state(self):
        motion = (
            self.root_pos[self.current_step : self.current_step + 1],
            self.root_rot[self.current_step : self.current_step + 1],
            self.root_lin_vel[self.current_step : self.current_step + 1],
            self.root_ang_vel[self.current_step : self.current_step + 1],
            self.joint_pos[self.current_step : self.current_step + 1],
            self.joint_vel[self.current_step : self.current_step + 1],
        )
        self.current_step += 1

        reset_flag = False
        if self.current_step >= self.num_frames:
            self.current_step = 0
            reset_flag = True
        return motion, reset_flag


def peek_motion_fps(file: str | Path) -> int:
    with open(file, "rb") as handle:
        motion_data = NumpyCompatUnpickler(handle).load()
    return round(motion_data["fps"])

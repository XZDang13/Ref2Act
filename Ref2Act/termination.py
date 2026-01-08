from __future__ import annotations

import torch

from .motion_lib import SamplerMod


class Termination:
    def __init__(
        self,
        *,
        termination_height: float,
        max_episode_length: int,
        motion_duration: float,
        motion_dt: float,
        sampler_mod: SamplerMod,
        early_termination: bool = True,
    ) -> None:
        self.termination_height = termination_height
        self.max_episode_length = max_episode_length
        self.motion_duration = motion_duration
        self.motion_dt = motion_dt
        self.sampler_mod = sampler_mod
        self.early_termination = early_termination

    def time_out(self, episode_length_buf: torch.Tensor) -> torch.Tensor:
        return episode_length_buf >= self.max_episode_length - 1

    def height_terminate(self, base_height: torch.Tensor) -> torch.Tensor:
        return base_height < self.termination_height

    def end_of_motion(self, motion_times: torch.Tensor) -> torch.Tensor:
        if self.sampler_mod != SamplerMod.Clamp:
            return torch.zeros_like(motion_times, dtype=torch.bool)
        end_time = torch.as_tensor(
            self.motion_duration - self.motion_dt * 0.5, device=motion_times.device
        )
        return motion_times >= end_time

    def get_dones(
        self,
        episode_length_buf: torch.Tensor,
        base_height: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = base_height.device
        time_out = self.time_out(episode_length_buf.to(device))
        terminate = self.end_of_motion(motion_times.to(device))
        if self.early_termination:
            terminate = terminate | self.height_terminate(base_height)
        return terminate, time_out

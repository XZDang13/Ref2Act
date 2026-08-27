from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VelocityCommandCfg:
    linear_x_range: tuple[float, float] = (-1.0, 1.5)
    linear_y_range: tuple[float, float] = (-0.6, 0.6)
    yaw_rate_range: tuple[float, float] = (-1.0, 1.0)
    resampling_time_range_s: tuple[float, float] = (3.0, 8.0)
    standing_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name, value_range in (
            ("linear_x_range", self.linear_x_range),
            ("linear_y_range", self.linear_y_range),
            ("yaw_rate_range", self.yaw_rate_range),
            ("resampling_time_range_s", self.resampling_time_range_s),
        ):
            if value_range[1] < value_range[0]:
                raise ValueError(f"{name} must be ordered, got {value_range}.")
        if self.resampling_time_range_s[0] <= 0.0:
            raise ValueError("Command resampling time must be positive.")
        if not 0.0 <= self.standing_fraction <= 1.0:
            raise ValueError("standing_fraction must be in [0, 1].")


class UniformVelocityCommandGenerator:
    """Independent task command process with no reference-motion dependency."""

    def __init__(
        self,
        *,
        cfg: VelocityCommandCfg,
        num_envs: int,
        step_dt: float,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.device = torch.device(device)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device)
        self.steps_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        ranges = (
            self.cfg.linear_x_range,
            self.cfg.linear_y_range,
            self.cfg.yaw_rate_range,
        )
        for index, value_range in enumerate(ranges):
            self.commands[env_ids, index] = torch.empty(
                env_ids.numel(),
                device=self.device,
            ).uniform_(*value_range)

        if self.cfg.standing_fraction > 0.0:
            standing = torch.rand(env_ids.numel(), device=self.device) < self.cfg.standing_fraction
            self.commands[env_ids[standing]] = 0.0

        min_steps = max(1, round(self.cfg.resampling_time_range_s[0] / self.step_dt))
        max_steps = max(min_steps, round(self.cfg.resampling_time_range_s[1] / self.step_dt))
        self.steps_left[env_ids] = torch.randint(
            min_steps,
            max_steps + 1,
            (env_ids.numel(),),
            device=self.device,
        )

    def step(self) -> None:
        self.steps_left -= 1
        due = torch.nonzero(self.steps_left <= 0, as_tuple=False).flatten()
        if due.numel() > 0:
            self.reset(due)


__all__ = ["UniformVelocityCommandGenerator", "VelocityCommandCfg"]

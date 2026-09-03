from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch


@dataclass
class VelocityCommandCfg:
    """Velocity-command ranges with Unitree-style progressive expansion."""

    linear_x_range: tuple[float, float] = (-0.1, 0.1)
    linear_y_range: tuple[float, float] = (-0.1, 0.1)
    yaw_rate_range: tuple[float, float] = (-0.1, 0.1)
    linear_x_limit: tuple[float, float] = (-0.5, 1.0)
    linear_y_limit: tuple[float, float] = (-0.3, 0.3)
    yaw_rate_limit: tuple[float, float] = (-0.2, 0.2)
    curriculum_enabled: bool = True
    curriculum_threshold: float = 0.8
    curriculum_step: float = 0.1
    resampling_time_range_s: tuple[float, float] = (10.0, 10.0)
    standing_fraction: float = 0.02

    def __post_init__(self) -> None:
        for name, value_range in (
            ("linear_x_range", self.linear_x_range),
            ("linear_y_range", self.linear_y_range),
            ("yaw_rate_range", self.yaw_rate_range),
            ("linear_x_limit", self.linear_x_limit),
            ("linear_y_limit", self.linear_y_limit),
            ("yaw_rate_limit", self.yaw_rate_limit),
            ("resampling_time_range_s", self.resampling_time_range_s),
        ):
            if value_range[1] < value_range[0]:
                raise ValueError(f"{name} must be ordered, got {value_range}.")
        if self.resampling_time_range_s[0] <= 0.0:
            raise ValueError("Command resampling time must be positive.")
        if not 0.0 <= self.standing_fraction <= 1.0:
            raise ValueError("standing_fraction must be in [0, 1].")
        if not 0.0 < self.curriculum_threshold <= 1.0:
            raise ValueError("curriculum_threshold must be in (0, 1].")
        if self.curriculum_step <= 0.0:
            raise ValueError("curriculum_step must be positive.")
        for name, initial, limit in (
            ("linear_x", self.linear_x_range, self.linear_x_limit),
            ("linear_y", self.linear_y_range, self.linear_y_limit),
            ("yaw_rate", self.yaw_rate_range, self.yaw_rate_limit),
        ):
            if initial[0] < limit[0] or initial[1] > limit[1]:
                raise ValueError(f"{name}_range must lie inside {name}_limit.")


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
        self.current_linear_x_range = tuple(float(value) for value in cfg.linear_x_range)
        self.current_linear_y_range = tuple(float(value) for value in cfg.linear_y_range)
        self.current_yaw_rate_range = tuple(float(value) for value in cfg.yaw_rate_range)

    @staticmethod
    def _expand_range(
        current: tuple[float, float],
        limit: tuple[float, float],
        step: float,
    ) -> tuple[float, float]:
        return (
            max(float(limit[0]), float(current[0]) - float(step)),
            min(float(limit[1]), float(current[1]) + float(step)),
        )

    def update_curriculum(self, *, linear_score: float, yaw_score: float) -> bool:
        """Expand command ranges after a full evaluation window succeeds."""

        if not self.cfg.curriculum_enabled:
            return False
        changed = False
        if float(linear_score) > float(self.cfg.curriculum_threshold):
            previous_x = self.current_linear_x_range
            previous_y = self.current_linear_y_range
            self.current_linear_x_range = self._expand_range(
                previous_x, self.cfg.linear_x_limit, self.cfg.curriculum_step
            )
            self.current_linear_y_range = self._expand_range(
                previous_y, self.cfg.linear_y_limit, self.cfg.curriculum_step
            )
            changed |= (
                self.current_linear_x_range != previous_x
                or self.current_linear_y_range != previous_y
            )
        if float(yaw_score) > float(self.cfg.curriculum_threshold):
            previous_yaw = self.current_yaw_rate_range
            self.current_yaw_rate_range = self._expand_range(
                previous_yaw, self.cfg.yaw_rate_limit, self.cfg.curriculum_step
            )
            changed |= self.current_yaw_rate_range != previous_yaw
        return changed

    def _normalize_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        ranges = (
            self.current_linear_x_range,
            self.current_linear_y_range,
            self.current_yaw_rate_range,
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


LOCOMOTION_COMMAND_CATEGORIES = (
    "stand",
    "sagittal",
    "lateral",
    "turn",
    "mixed",
)


@dataclass
class StratifiedVelocityCommandCfg(VelocityCommandCfg):
    """Discrete, balanced command distribution for flat locomotion.

    Every moving sample has a clearly identifiable target.  This avoids the
    near-zero commands produced by continuous uniform sampling and makes the
    stationary policy a measurable zero-reward baseline.
    """

    linear_x_range: tuple[float, float] = (-0.4, 1.0)
    linear_y_range: tuple[float, float] = (-0.3, 0.3)
    yaw_rate_range: tuple[float, float] = (-0.5, 0.5)
    linear_x_limit: tuple[float, float] = (-0.4, 1.0)
    linear_y_limit: tuple[float, float] = (-0.3, 0.3)
    yaw_rate_limit: tuple[float, float] = (-0.5, 0.5)
    curriculum_enabled: bool = False
    standing_fraction: float = 0.0
    category_fractions: tuple[float, ...] = (0.10, 0.35, 0.15, 0.20, 0.20)
    sagittal_levels: tuple[float, ...] = (-0.4, 0.3, 0.6, 1.0)
    lateral_levels: tuple[float, ...] = (-0.3, -0.15, 0.15, 0.3)
    yaw_levels: tuple[float, ...] = (-0.5, -0.25, 0.25, 0.5)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.category_fractions) != len(LOCOMOTION_COMMAND_CATEGORIES):
            raise ValueError(
                "category_fractions must contain stand, sagittal, lateral, turn, mixed."
            )
        if any(value < 0.0 for value in self.category_fractions):
            raise ValueError("Command category fractions must be non-negative.")
        if abs(sum(self.category_fractions) - 1.0) > 1.0e-6:
            raise ValueError("Command category fractions must sum to one.")
        for name, values, limit in (
            ("sagittal_levels", self.sagittal_levels, self.linear_x_limit),
            ("lateral_levels", self.lateral_levels, self.linear_y_limit),
            ("yaw_levels", self.yaw_levels, self.yaw_rate_limit),
        ):
            if not values or any(value == 0.0 for value in values):
                raise ValueError(f"{name} must contain non-zero values.")
            if min(values) < limit[0] or max(values) > limit[1]:
                raise ValueError(f"{name} lies outside {limit}.")


class StratifiedVelocityCommandGenerator(UniformVelocityCommandGenerator):
    """Sample stand/single-axis/mixed commands with fixed proportions."""

    cfg: StratifiedVelocityCommandCfg

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category_names = LOCOMOTION_COMMAND_CATEGORIES
        self.category_ids = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._probabilities = torch.tensor(
            self.cfg.category_fractions, device=self.device, dtype=torch.float32
        )

    def _sample_levels(self, values: tuple[float, ...], count: int) -> torch.Tensor:
        levels = torch.as_tensor(values, device=self.device, dtype=self.commands.dtype)
        return levels[torch.randint(levels.numel(), (count,), device=self.device)]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        categories = torch.multinomial(
            self._probabilities, env_ids.numel(), replacement=True
        )
        self.category_ids[env_ids] = categories
        self.commands[env_ids] = 0.0
        for category, name in enumerate(self.category_names):
            selected = env_ids[categories == category]
            count = int(selected.numel())
            if count == 0 or name == "stand":
                continue
            if name in {"sagittal", "mixed"}:
                self.commands[selected, 0] = self._sample_levels(
                    self.cfg.sagittal_levels, count
                )
            if name in {"lateral", "mixed"}:
                self.commands[selected, 1] = self._sample_levels(
                    self.cfg.lateral_levels, count
                )
            if name in {"turn", "mixed"}:
                self.commands[selected, 2] = self._sample_levels(
                    self.cfg.yaw_levels, count
                )


VelocityCommandConfig: TypeAlias = VelocityCommandCfg | StratifiedVelocityCommandCfg


def make_velocity_command_generator(
    *,
    cfg: VelocityCommandConfig,
    num_envs: int,
    step_dt: float,
    device: torch.device | str,
) -> UniformVelocityCommandGenerator:
    generator_type = (
        StratifiedVelocityCommandGenerator
        if isinstance(cfg, StratifiedVelocityCommandCfg)
        else UniformVelocityCommandGenerator
    )
    return generator_type(
        cfg=cfg, num_envs=num_envs, step_dt=step_dt, device=device
    )


__all__ = [
    "LOCOMOTION_COMMAND_CATEGORIES",
    "StratifiedVelocityCommandCfg",
    "StratifiedVelocityCommandGenerator",
    "UniformVelocityCommandGenerator",
    "VelocityCommandCfg",
    "make_velocity_command_generator",
]

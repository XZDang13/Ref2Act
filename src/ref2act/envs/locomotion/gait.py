"""Command-only gait clock shared by training and deployment (no simulator API)."""
from dataclasses import dataclass
import math
import torch


@dataclass
class CommandGaitCfg:
    # Hz is full cycles per foot; alternating steps/s is twice this value.
    min_frequency: float = 1.0
    max_frequency: float = 1.25
    forward_slow: float = 0.3
    forward_fast: float = 0.6
    lateral_scale: float = 0.3
    yaw_scale: float = 0.5
    smoothing_time: float = 0.2
    stand_threshold: float = 0.01

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Gait schedule parameters must be finite.')
        if not 0 < self.min_frequency <= self.max_frequency:
            raise ValueError('Require 0 < min_frequency <= max_frequency.')
        if not 0 <= self.forward_slow < self.forward_fast:
            raise ValueError('Require 0 <= forward_slow < forward_fast.')
        if min(self.lateral_scale, self.yaw_scale, self.smoothing_time) <= 0:
            raise ValueError('Gait scales and smoothing_time must be positive.')
        if self.stand_threshold < 0:
            raise ValueError('stand_threshold must be non-negative.')

    def target_frequency(self, commands: torch.Tensor) -> torch.Tensor:
        if commands.ndim != 2 or commands.shape[1] != 3:
            raise ValueError('commands must have shape [env, 3].')
        x = (commands[:, 0].abs() - self.forward_slow) / (self.forward_fast - self.forward_slow)
        y = commands[:, 1].abs() / self.lateral_scale
        yaw = commands[:, 2].abs() / self.yaw_scale
        demand = torch.stack((x, y, yaw), -1).amax(-1).clamp(0, 1)
        return self.min_frequency + (self.max_frequency - self.min_frequency) * demand


class CommandGaitClock:
    """Advance exactly once per control step; phase() is an idempotent read.

    Zero commands finish the current swing at the next double-support boundary,
    then hold phase. Frequency is filtered, never phase. Reset only touches ids.
    """
    def __init__(self, cfg: CommandGaitCfg, num_envs: int, device, *, step_dt: float,
                 stance_ratio: float = .55, offsets=(0., .5)):
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError('step_dt must be finite and positive.')
        if not .5 < stance_ratio < 1 or tuple(offsets) != (0., .5):
            raise ValueError('Adaptive clock requires alternating feet and double support.')
        if step_dt * cfg.max_frequency >= .5:
            raise ValueError('Control step must resolve each half cycle.')
        self.cfg, self.dt = cfg, step_dt
        self.swing_half_width = math.pi * (1 - stance_ratio)
        self.angle = torch.zeros(num_envs, device=device)
        self.frequency = torch.full_like(self.angle, cfg.min_frequency)
        self.target = self.frequency.clone()
        self.advancing = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids, commands, initial_phase):
        target = self.cfg.target_frequency(commands[ids])
        self.frequency[ids] = target
        self.target[ids] = target
        standing = commands[ids].norm(dim=-1) < self.cfg.stand_threshold
        # Both feet grounded at +/- pi/2; moving resets retain randomized phase.
        self.angle[ids] = torch.where(standing, math.pi / 2, initial_phase[ids]).remainder(2 * math.pi)
        self.advancing[ids] = ~standing

    def advance(self, commands):
        self.target.copy_(self.cfg.target_frequency(commands))
        alpha = -math.expm1(-self.dt / self.cfg.smoothing_time)
        self.frequency.lerp_(self.target, alpha)
        increment = 2 * math.pi * self.dt * self.frequency
        half_phase = self.angle.remainder(math.pi)
        grounded = (half_phase >= self.swing_half_width) & (half_phase <= math.pi - self.swing_half_width)
        distance = (self.swing_half_width + 1e-5 - half_phase).remainder(math.pi)
        stopping_increment = torch.where(grounded, 0., torch.minimum(increment, distance))
        standing = commands.norm(dim=-1) < self.cfg.stand_threshold
        increment = torch.where(standing, stopping_increment, increment)
        self.advancing.copy_(increment > 0)
        self.angle.add_(increment).remainder_(2 * math.pi)

    def phase(self):
        phases = self.angle[:, None] + self.angle.new_tensor((0., math.pi))
        return (phases + math.pi).remainder(2 * math.pi) - math.pi

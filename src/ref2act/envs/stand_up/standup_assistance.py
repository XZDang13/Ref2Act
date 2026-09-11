"""Performance-driven stand-up assistance; independent of optimizer settings."""
from __future__ import annotations

import math
import torch


def assistance_settings(config: dict) -> dict:
    settings = {
        'schedule': config.get('schedule', 'linear'),
        'orientation_gate_start': float(config.get('orientation_gate_start', -1.)),
        'orientation_gate_full': float(config.get('orientation_gate_full', -1.)),
        'minimum_draw': float(config.get('minimum_draw', 0.)),
        'probe_fraction': float(config.get('probe_fraction', 0.)),
        'decrement': float(config.get('decrement', .05)),
        'minimum_outcomes': int(config.get('minimum_outcomes', 256)),
        'required_windows': int(config.get('required_windows', 3)),
        'stable_rate': float(config.get('stable_rate', .6)),
        'minimum_probe_outcomes': int(config.get('minimum_probe_outcomes', 32)),
        'probe_stable_rate': float(config.get('probe_stable_rate', .3)),
    }
    if settings['schedule'] not in ('linear', 'performance'):
        raise ValueError('assistance.schedule must be linear or performance.')
    if not all(math.isfinite(v) for v in settings.values() if isinstance(v, (int, float))):
        raise ValueError('Assistance settings must be finite.')
    start, full = settings['orientation_gate_start'], settings['orientation_gate_full']
    if (start, full) != (-1., -1.) and not -1 <= start < full <= 1:
        raise ValueError('Assistance orientation gate requires -1 <= start < full <= 1.')
    if not 0 <= settings['minimum_draw'] <= 1 or not 0 <= settings['probe_fraction'] < 1:
        raise ValueError('Invalid assistance draw range or probe fraction.')
    if not 0 < settings['decrement'] <= 1 or min(settings['minimum_outcomes'], settings['required_windows'], settings['minimum_probe_outcomes']) < 1:
        raise ValueError('Assistance curriculum step and sample counts must be positive.')
    if not all(0 < settings[k] <= 1 for k in ('stable_rate', 'probe_stable_rate')):
        raise ValueError('Assistance curriculum rates must be in (0, 1].')
    if settings['schedule'] == 'performance' and settings['probe_fraction'] <= 0:
        raise ValueError('Performance assistance requires unassisted probe episodes.')
    return settings


def draw_assistance(count: int, device, settings: dict) -> torch.Tensor:
    draw = settings['minimum_draw'] + (1 - settings['minimum_draw']) * torch.rand(count, device=device)
    if settings['probe_fraction'] > 0:
        draw[torch.rand(count, device=device) < settings['probe_fraction']] = 0
    return draw


def orientation_multiplier(upright: torch.Tensor, settings: dict) -> torch.Tensor:
    start, full = settings['orientation_gate_start'], settings['orientation_gate_full']
    if (start, full) == (-1., -1.):
        return torch.ones_like(upright)
    t = ((torch.nan_to_num(upright, nan=-1.) - start) / (full - start)).clamp(0, 1)
    return t.square() * (3 - 2 * t)


class PerformanceAssistance:
    """Reduce force only on disjoint windows of completed, same-level episodes.

    The caller excludes episodes spanning force changes. Stable outcomes mean
    standing at episode end, not merely having touched a height once. The final
    decrement additionally requires successful entirely unassisted episodes.
    """
    def __init__(self, maximum: float, settings: dict):
        if not math.isfinite(maximum) or maximum < 0:
            raise ValueError('Assistance maximum must be finite and nonnegative.')
        self.maximum = maximum
        self.settings = dict(settings)
        self.ratio = maximum
        self.streak = 0
        self.last_stable_rate = 0.
        self.last_probe_rate = 0.
        self.outcomes = self.stable = self.probes = self.probe_stable = 0.

    def observe(self, outcomes: float, stable: float, probes: float, probe_stable: float) -> bool:
        values = (outcomes, stable, probes, probe_stable)
        if not all(math.isfinite(v) and v >= 0 for v in values) or not (probe_stable <= probes <= outcomes and probe_stable <= stable <= outcomes):
            raise ValueError('Invalid assistance episode counts.')
        if self.ratio <= 0:
            return False
        self.outcomes += outcomes
        self.stable += stable
        self.probes += probes
        self.probe_stable += probe_stable
        final_step = self.ratio <= self.settings['decrement'] + 1.e-8
        if self.outcomes < self.settings['minimum_outcomes'] or (final_step and self.probes < self.settings['minimum_probe_outcomes']):
            return False
        self.last_stable_rate = self.stable / self.outcomes
        self.last_probe_rate = self.probe_stable / max(self.probes, 1)
        passed = self.last_stable_rate >= self.settings['stable_rate']
        if final_step:
            passed &= self.last_probe_rate >= self.settings['probe_stable_rate']
        self.streak = self.streak + 1 if passed else 0
        self.outcomes = self.stable = self.probes = self.probe_stable = 0.
        if self.streak < self.settings['required_windows']:
            return False
        self.ratio = max(0., round(self.ratio - self.settings['decrement'], 10))
        self.streak = 0
        return True

    def state_dict(self) -> dict:
        return {key: value for key, value in vars(self).items()}

    def load_state_dict(self, state: dict) -> None:
        if state['settings'] != self.settings or state['maximum'] != self.maximum:
            raise ValueError('Assistance curriculum changed; use a policy warm start instead of exact resume.')
        if not math.isfinite(state['ratio']) or not 0 <= state['ratio'] <= self.maximum:
            raise ValueError('Invalid saved assistance ratio.')
        for key in vars(self):
            if key not in ('settings', 'maximum'):
                setattr(self, key, state[key])

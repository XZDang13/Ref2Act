from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from ref2act.common.utils import IndexLike

from .library import MotionClip, MotionLib
from .segments import build_legacy_time_segments


class SamplerMod(Enum):
    Cycle = 0
    Clamp = 1


class SamplingStrategy(Enum):
    Start = 0
    Random = 1
    FailureWeighted = 2


class SegmentSource(Enum):
    Time = 0
    Anchor = 1


@dataclass(frozen=True)
class ResetSample:
    env_ids: torch.Tensor
    motion_ids: torch.Tensor
    times: torch.Tensor
    target_bin_indices: torch.Tensor


@dataclass(frozen=True)
class AdaptiveSamplerCfg:
    enabled: bool = False
    warmup_samples: int = 96
    anchor_drop_fail_rate: float = 0.97
    anchor_reenable_fail_rate: float = 0.75
    anchor_cooldown_resets: int = 500
    motion_min_samples: int = 256
    motion_drop_fail_rate: float = 0.98
    motion_drop_anchor_fraction: float = 0.8
    motion_cooldown_resets: int = 1000
    probe_probability: float = 0.01
    mastered_fail_rate: float = 0.05
    mastered_probability_scale: float = 0.25
    min_live_motion_fraction: float = 0.6
    min_live_anchor_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.warmup_samples < 1:
            raise ValueError("warmup_samples must be at least 1.")
        if self.motion_min_samples < 1:
            raise ValueError("motion_min_samples must be at least 1.")
        if self.anchor_cooldown_resets < 0:
            raise ValueError("anchor_cooldown_resets must be non-negative.")
        if self.motion_cooldown_resets < 0:
            raise ValueError("motion_cooldown_resets must be non-negative.")
        for name in (
            "anchor_drop_fail_rate",
            "anchor_reenable_fail_rate",
            "motion_drop_fail_rate",
            "motion_drop_anchor_fraction",
            "probe_probability",
            "mastered_fail_rate",
            "min_live_motion_fraction",
            "min_live_anchor_fraction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1].")
        if self.anchor_reenable_fail_rate > self.anchor_drop_fail_rate:
            raise ValueError("anchor_reenable_fail_rate must be <= anchor_drop_fail_rate.")
        if self.mastered_fail_rate > self.anchor_drop_fail_rate:
            raise ValueError("mastered_fail_rate must be <= anchor_drop_fail_rate.")
        probability_scale = float(self.mastered_probability_scale)
        if not math.isfinite(probability_scale) or probability_scale <= 0.0 or probability_scale > 1.0:
            raise ValueError("mastered_probability_scale must be a finite value in (0, 1].")


class MotionSampler:
    def __init__(
        self,
        num_envs: int,
        motion_lib: MotionLib,
        dt: float,
        *,
        anchor_body_index: int | None = None,
        bin_size: float | None = None,
        failure_decay: float = 1.0,
        failure_weight_uniform_mix: float = 0.1,
        failure_weight_max_uniform_ratio: float | None = 2.5,
        failure_weight_exploration_bonus: float = 0.5,
        segment_source: SegmentSource = SegmentSource.Time,
        adaptive_sampler: AdaptiveSamplerCfg | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.motion_lib = motion_lib
        self.anchor_body_index = anchor_body_index
        self.segment_source = segment_source
        self.adaptive_sampler = adaptive_sampler if adaptive_sampler is not None else AdaptiveSamplerCfg()
        if not (0.0 < failure_decay <= 1.0):
            raise ValueError("failure_decay must be in (0, 1].")
        self.failure_decay = failure_decay
        if not (0.0 <= failure_weight_uniform_mix <= 1.0):
            raise ValueError("failure_weight_uniform_mix must be in [0, 1].")
        self.failure_weight_uniform_mix = float(failure_weight_uniform_mix)
        if failure_weight_max_uniform_ratio is None:
            self.failure_weight_max_uniform_ratio = None
        else:
            resolved_max_uniform_ratio = float(failure_weight_max_uniform_ratio)
            if not math.isfinite(resolved_max_uniform_ratio) or resolved_max_uniform_ratio < 1.0:
                raise ValueError("failure_weight_max_uniform_ratio must be None or a finite float >= 1.")
            self.failure_weight_max_uniform_ratio = resolved_max_uniform_ratio
        resolved_exploration_bonus = float(failure_weight_exploration_bonus)
        if not math.isfinite(resolved_exploration_bonus) or resolved_exploration_bonus < 0.0:
            raise ValueError("failure_weight_exploration_bonus must be a finite float >= 0.")
        self.failure_weight_exploration_bonus = resolved_exploration_bonus

        self.current_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_start_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_bin_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.bin_size: float | None = float(bin_size) if bin_size is not None else None
        self.num_bins = 0
        self.num_bins_per_motion: torch.Tensor | None = None
        self.bin_start_times: list[torch.Tensor] | None = None
        self.bin_end_times: list[torch.Tensor] | None = None
        self.bin_types: list[torch.Tensor] | None = None
        self.bin_reset_times: list[torch.Tensor] | None = None
        self.bin_reset_eligible: list[torch.Tensor] | None = None
        self.bin_uses_segment_metadata: list[bool] | None = None
        self.bin_fail_counts: list[torch.Tensor] | None = None
        self.bin_sample_counts: list[torch.Tensor] | None = None
        self.supports_failure_weighted_sampling = False

        self.adaptive_bin_quarantined: list[torch.Tensor] | None = None
        self.adaptive_bin_cooldowns: list[torch.Tensor] | None = None
        self.adaptive_bin_mastered: list[torch.Tensor] | None = None
        self.adaptive_motion_quarantined: torch.Tensor | None = None
        self.adaptive_motion_cooldowns: torch.Tensor | None = None
        self.adaptive_motion_mastered: torch.Tensor | None = None
        self.adaptive_probe_sample_count = 0

        should_init_failure_bins = (
            self.segment_source == SegmentSource.Anchor
            or self.bin_size is not None
            or (self.segment_source == SegmentSource.Time and self.motion_lib.all_clips_have_segments)
        )
        if should_init_failure_bins:
            self.init_failure_bins(self.bin_size)

    @property
    def duration(self) -> float:
        return self.motion_lib.duration

    def _normalize_env_ids(self, env_ids: IndexLike | None = None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def get_current_durations(self, env_ids: IndexLike | None = None) -> torch.Tensor:
        env_ids = self._normalize_env_ids(env_ids)
        return self.motion_lib.get_duration(self.current_motion_ids[env_ids])

    def sample_motion_ids(self, env_ids: IndexLike | None = None) -> torch.Tensor:
        env_ids = self._normalize_env_ids(env_ids)
        return torch.randint(self.motion_lib.num_motions, (env_ids.numel(),), device=self.device)

    def _build_guarded_sampling_probabilities(
        self,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        eligible_mask: torch.Tensor | None = None,
        probability_scale: torch.Tensor | None = None,
        probe_mask: torch.Tensor | None = None,
        probe_probability: float = 0.0,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        resolved_probe_probability = float(probe_probability)
        if (
            not math.isfinite(resolved_probe_probability)
            or resolved_probe_probability < 0.0
            or resolved_probe_probability > 1.0
        ):
            raise ValueError("probe_probability must be a finite value in [0, 1].")

        fail_counts = torch.as_tensor(fail_counts, dtype=torch.float32, device=self.device).reshape(-1)
        sample_counts = torch.as_tensor(sample_counts, dtype=torch.float32, device=self.device).reshape(-1)
        if fail_counts.shape != sample_counts.shape:
            raise ValueError("fail_counts and sample_counts must have the same shape.")

        if eligible_mask is None:
            eligible = torch.ones_like(fail_counts, dtype=torch.bool, device=self.device)
        else:
            eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device).reshape(-1)
            if eligible.shape != fail_counts.shape:
                raise ValueError("eligible_mask must have the same shape as fail_counts.")

        if probability_scale is None:
            scale = torch.ones_like(fail_counts, dtype=torch.float32, device=self.device)
        else:
            scale = torch.as_tensor(probability_scale, dtype=torch.float32, device=self.device).reshape(-1)
            if scale.shape != fail_counts.shape:
                raise ValueError("probability_scale must have the same shape as fail_counts.")
            if not bool(torch.all(torch.isfinite(scale)).item()) or bool(torch.any(scale < 0.0).item()):
                raise ValueError("probability_scale must contain finite non-negative values.")

        if probe_mask is None:
            probe = torch.zeros_like(eligible, dtype=torch.bool, device=self.device)
        else:
            probe = torch.as_tensor(probe_mask, dtype=torch.bool, device=self.device).reshape(-1)
            if probe.shape != fail_counts.shape:
                raise ValueError("probe_mask must have the same shape as fail_counts.")

        probe_probs = probe.to(dtype=torch.float32)
        probe_sum = torch.sum(probe_probs)
        if float(probe_sum.item()) > 0.0:
            probe_probs = probe_probs / probe_sum

        uniform_probs = eligible.to(dtype=torch.float32)
        uniform_sum = torch.sum(uniform_probs)
        if uniform_sum <= 0:
            if float(probe_sum.item()) > 0.0:
                return probe_probs
            raise ValueError("eligible_mask must include at least one entry.")
        uniform_probs = uniform_probs / uniform_sum
        scaled_uniform_probs = torch.where(eligible, uniform_probs * scale, torch.zeros_like(uniform_probs))
        scaled_uniform_sum = torch.sum(scaled_uniform_probs)
        if float(scaled_uniform_sum.item()) > 0.0:
            scaled_uniform_probs = scaled_uniform_probs / scaled_uniform_sum
        else:
            scaled_uniform_probs = uniform_probs

        eligible_sample_counts = torch.where(
            eligible,
            torch.clamp(sample_counts, min=0.0),
            torch.zeros_like(sample_counts),
        )
        eligible_count = torch.sum(eligible.to(dtype=torch.float32))
        total_eligible_samples = torch.sum(eligible_sample_counts)
        exploration_scale = torch.log(total_eligible_samples + eligible_count + 1.0)
        exploration = torch.sqrt(exploration_scale / (eligible_sample_counts + 1.0))

        fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
        score = fail_rate + self.failure_weight_exploration_bonus * exploration
        learned_weights = score.pow(1.0 / temperature)
        learned_weights = torch.where(eligible, learned_weights, torch.zeros_like(learned_weights))

        learned_sum = torch.sum(learned_weights)
        if bool(torch.all(torch.isfinite(learned_weights)).item()) and float(learned_sum.item()) > 0.0:
            learned_probs = learned_weights / learned_sum
        else:
            learned_probs = uniform_probs

        probs = (
            (1.0 - self.failure_weight_uniform_mix) * learned_probs
            + self.failure_weight_uniform_mix * uniform_probs
        )
        probs = torch.where(eligible, probs * scale, torch.zeros_like(probs))
        probs_sum = torch.sum(probs)
        if float(probs_sum.item()) > 0.0:
            probs = probs / probs_sum
        else:
            probs = scaled_uniform_probs

        probs = self._apply_max_uniform_probability_cap(
            probs,
            eligible_mask=eligible,
            uniform_probs=scaled_uniform_probs,
        )
        if resolved_probe_probability > 0.0 and float(probe_sum.item()) > 0.0:
            probs = (1.0 - resolved_probe_probability) * probs + resolved_probe_probability * probe_probs
            probs = probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)
        return probs

    def _apply_max_uniform_probability_cap(
        self,
        probs: torch.Tensor,
        *,
        eligible_mask: torch.Tensor,
        uniform_probs: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.as_tensor(probs, dtype=torch.float32, device=self.device).reshape(-1)
        eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device).reshape(-1)
        uniform_probs = torch.as_tensor(uniform_probs, dtype=torch.float32, device=self.device).reshape(-1)
        if probs.shape != eligible.shape or probs.shape != uniform_probs.shape:
            raise ValueError("Probability cap inputs must have the same shape.")

        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs = probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)

        if self.failure_weight_max_uniform_ratio is None:
            return probs

        eligible_count = int(torch.sum(eligible).item())
        if eligible_count == 0:
            raise ValueError("eligible_mask must include at least one entry.")

        max_prob = float(self.failure_weight_max_uniform_ratio) / float(eligible_count)
        if max_prob >= 1.0:
            return probs

        eps = torch.finfo(probs.dtype).eps
        if not bool(torch.all(torch.isfinite(probs)).item()):
            return uniform_probs
        if float(torch.sum(probs).item()) <= eps:
            return uniform_probs
        if bool(torch.all(probs[eligible] <= max_prob + eps).item()):
            return probs

        # Project onto the simplex with a shared upper bound so no eligible entry exceeds max_prob.
        eligible_probs = probs[eligible]
        lower = torch.min(eligible_probs - max_prob)
        upper = torch.max(eligible_probs)
        for _ in range(64):
            threshold = 0.5 * (lower + upper)
            projected_probs = torch.clamp(eligible_probs - threshold, min=0.0, max=max_prob)
            if float(torch.sum(projected_probs).item()) > 1.0:
                lower = threshold
            else:
                upper = threshold

        eligible_projected_probs = torch.clamp(eligible_probs - upper, min=0.0, max=max_prob)
        projected_sum = torch.sum(eligible_projected_probs)
        if not bool(torch.isfinite(projected_sum).item()) or float(projected_sum.item()) <= eps:
            return uniform_probs

        capped_probs = torch.zeros_like(probs)
        capped_probs[eligible] = eligible_projected_probs
        return capped_probs / projected_sum

    def _has_adaptive_state(self) -> bool:
        return (
            bool(self.adaptive_sampler.enabled)
            and self._has_failure_bins()
            and self.adaptive_bin_quarantined is not None
            and self.adaptive_bin_cooldowns is not None
            and self.adaptive_bin_mastered is not None
            and self.adaptive_motion_quarantined is not None
            and self.adaptive_motion_cooldowns is not None
            and self.adaptive_motion_mastered is not None
        )

    def _init_adaptive_state(self) -> None:
        if not self.adaptive_sampler.enabled or not self._has_failure_bins():
            self.adaptive_bin_quarantined = None
            self.adaptive_bin_cooldowns = None
            self.adaptive_bin_mastered = None
            self.adaptive_motion_quarantined = None
            self.adaptive_motion_cooldowns = None
            self.adaptive_motion_mastered = None
            self.adaptive_probe_sample_count = 0
            return

        self.adaptive_bin_quarantined = [
            torch.zeros_like(sample_counts, dtype=torch.bool, device=self.device)
            for sample_counts in self.bin_sample_counts
        ]
        self.adaptive_bin_cooldowns = [
            torch.zeros_like(sample_counts, dtype=torch.long, device=self.device)
            for sample_counts in self.bin_sample_counts
        ]
        self.adaptive_bin_mastered = [
            torch.zeros_like(sample_counts, dtype=torch.bool, device=self.device)
            for sample_counts in self.bin_sample_counts
        ]
        self.adaptive_motion_quarantined = torch.zeros(
            self.motion_lib.num_motions,
            dtype=torch.bool,
            device=self.device,
        )
        self.adaptive_motion_cooldowns = torch.zeros(
            self.motion_lib.num_motions,
            dtype=torch.long,
            device=self.device,
        )
        self.adaptive_motion_mastered = torch.zeros(
            self.motion_lib.num_motions,
            dtype=torch.bool,
            device=self.device,
        )
        self.adaptive_probe_sample_count = 0

    def _adaptive_base_bin_eligible(self, motion_id: int) -> torch.Tensor:
        if self.segment_source == SegmentSource.Anchor:
            return self.bin_reset_eligible[motion_id]
        return torch.ones_like(self.bin_sample_counts[motion_id], dtype=torch.bool, device=self.device)

    def _adaptive_limited_drop_mask(
        self,
        candidate_mask: torch.Tensor,
        base_eligible: torch.Tensor,
        current_quarantined: torch.Tensor,
        scores: torch.Tensor,
        *,
        min_live_fraction: float,
    ) -> torch.Tensor:
        candidate_mask = torch.as_tensor(candidate_mask, dtype=torch.bool, device=self.device).reshape(-1)
        base_eligible = torch.as_tensor(base_eligible, dtype=torch.bool, device=self.device).reshape(-1)
        current_quarantined = torch.as_tensor(current_quarantined, dtype=torch.bool, device=self.device).reshape(-1)
        scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device).reshape(-1)
        selected = torch.zeros_like(candidate_mask)

        total_eligible = int(torch.sum(base_eligible).item())
        if total_eligible == 0:
            return selected

        min_live = int(math.ceil(float(total_eligible) * float(min_live_fraction)))
        max_quarantined = max(total_eligible - min_live, 0)
        current_count = int(torch.sum(current_quarantined & base_eligible).item())
        allowed_count = max(max_quarantined - current_count, 0)
        if allowed_count == 0:
            return selected

        candidate_indices = torch.nonzero(
            candidate_mask & base_eligible & ~current_quarantined,
            as_tuple=False,
        ).squeeze(-1)
        if candidate_indices.numel() == 0:
            return selected
        if int(candidate_indices.numel()) > allowed_count:
            order = torch.argsort(scores[candidate_indices], descending=True)
            candidate_indices = candidate_indices[order[:allowed_count]]
        selected[candidate_indices] = True
        return selected

    def _adaptive_fail_rate(self, fail_counts: torch.Tensor, sample_counts: torch.Tensor) -> torch.Tensor:
        return fail_counts / torch.clamp(sample_counts, min=1.0)

    def _advance_adaptive_cooldowns(self) -> None:
        if not self._has_adaptive_state():
            return
        for cooldowns in self.adaptive_bin_cooldowns:
            cooldowns[cooldowns > 0] -= 1
        self.adaptive_motion_cooldowns[self.adaptive_motion_cooldowns > 0] -= 1

    def _update_adaptive_state(self) -> None:
        if not self._has_adaptive_state():
            return
        self._update_adaptive_bin_state()
        self._update_adaptive_motion_state()

    def _update_adaptive_bin_state(self) -> None:
        cfg = self.adaptive_sampler
        for motion_id in range(self.motion_lib.num_motions):
            base_eligible = self._adaptive_base_bin_eligible(motion_id)
            quarantined = self.adaptive_bin_quarantined[motion_id]
            cooldowns = self.adaptive_bin_cooldowns[motion_id]
            fail_counts = self.bin_fail_counts[motion_id]
            sample_counts = self.bin_sample_counts[motion_id]
            fail_rate = self._adaptive_fail_rate(fail_counts, sample_counts)
            warmed = sample_counts >= float(cfg.warmup_samples)

            reenable_mask = (
                base_eligible
                & quarantined
                & (cooldowns <= 0)
                & (~warmed | (fail_rate <= float(cfg.anchor_reenable_fail_rate)))
            )
            quarantined[reenable_mask] = False
            cooldowns[reenable_mask] = 0

            drop_candidates = (
                base_eligible
                & ~quarantined
                & warmed
                & (fail_rate >= float(cfg.anchor_drop_fail_rate))
            )
            drop_mask = self._adaptive_limited_drop_mask(
                drop_candidates,
                base_eligible,
                quarantined,
                fail_rate,
                min_live_fraction=float(cfg.min_live_anchor_fraction),
            )
            quarantined[drop_mask] = True
            cooldowns[drop_mask] = int(cfg.anchor_cooldown_resets)

            self.adaptive_bin_mastered[motion_id][:] = (
                base_eligible
                & ~quarantined
                & warmed
                & (fail_rate <= float(cfg.mastered_fail_rate))
            )

    def _adaptive_motion_anchor_quarantine_fraction(self) -> torch.Tensor:
        fractions = torch.zeros(self.motion_lib.num_motions, dtype=torch.float32, device=self.device)
        for motion_id in range(self.motion_lib.num_motions):
            base_eligible = self._adaptive_base_bin_eligible(motion_id)
            total_eligible = torch.sum(base_eligible.to(dtype=torch.float32))
            if float(total_eligible.item()) <= 0.0:
                continue
            quarantined = self.adaptive_bin_quarantined[motion_id] & base_eligible
            fractions[motion_id] = torch.sum(quarantined.to(dtype=torch.float32)) / total_eligible
        return fractions

    def _update_adaptive_motion_state(self) -> None:
        cfg = self.adaptive_sampler
        motion_fail_counts = torch.stack([fail_counts.sum() for fail_counts in self.bin_fail_counts], dim=0)
        motion_sample_counts = torch.stack([sample_counts.sum() for sample_counts in self.bin_sample_counts], dim=0)
        motion_fail_rate = self._adaptive_fail_rate(motion_fail_counts, motion_sample_counts)
        anchor_quarantine_fraction = self._adaptive_motion_anchor_quarantine_fraction()
        warmed = motion_sample_counts >= float(cfg.motion_min_samples)

        reenable_mask = (
            self.adaptive_motion_quarantined
            & (self.adaptive_motion_cooldowns <= 0)
            & (
                ~warmed
                | (motion_fail_rate <= float(cfg.anchor_reenable_fail_rate))
                | (anchor_quarantine_fraction < float(cfg.motion_drop_anchor_fraction))
            )
        )
        self.adaptive_motion_quarantined[reenable_mask] = False
        self.adaptive_motion_cooldowns[reenable_mask] = 0

        drop_candidates = (
            ~self.adaptive_motion_quarantined
            & warmed
            & (motion_fail_rate >= float(cfg.motion_drop_fail_rate))
            & (anchor_quarantine_fraction >= float(cfg.motion_drop_anchor_fraction))
        )
        base_eligible = torch.ones_like(self.adaptive_motion_quarantined, dtype=torch.bool, device=self.device)
        drop_mask = self._adaptive_limited_drop_mask(
            drop_candidates,
            base_eligible,
            self.adaptive_motion_quarantined,
            motion_fail_rate,
            min_live_fraction=float(cfg.min_live_motion_fraction),
        )
        self.adaptive_motion_quarantined[drop_mask] = True
        self.adaptive_motion_cooldowns[drop_mask] = int(cfg.motion_cooldown_resets)

        self.adaptive_motion_mastered[:] = (
            ~self.adaptive_motion_quarantined
            & warmed
            & (motion_fail_rate <= float(cfg.mastered_fail_rate))
        )

    def _record_adaptive_probe_samples(self, sampled_indices: torch.Tensor, probe_mask: torch.Tensor | None) -> None:
        if not self._has_adaptive_state() or probe_mask is None:
            return
        probe_mask = torch.as_tensor(probe_mask, dtype=torch.bool, device=self.device).reshape(-1)
        sampled_indices = torch.as_tensor(sampled_indices, dtype=torch.long, device=self.device).reshape(-1)
        if sampled_indices.numel() == 0 or not bool(torch.any(probe_mask).item()):
            return
        self.adaptive_probe_sample_count += int(torch.sum(probe_mask[sampled_indices]).item())

    def _adaptive_motion_probability_inputs(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float]:
        if not self._has_adaptive_state():
            return None, None, None, 0.0
        eligible = ~self.adaptive_motion_quarantined
        scale = torch.where(
            self.adaptive_motion_mastered,
            torch.full_like(
                self.adaptive_motion_mastered,
                float(self.adaptive_sampler.mastered_probability_scale),
                dtype=torch.float32,
            ),
            torch.ones_like(self.adaptive_motion_mastered, dtype=torch.float32),
        )
        return eligible, scale, self.adaptive_motion_quarantined, float(self.adaptive_sampler.probe_probability)

    def _adaptive_bin_probability_inputs(
        self,
        motion_id: int,
        base_eligible: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, float]:
        if not self._has_adaptive_state():
            return base_eligible, None, None, 0.0
        resolved_base_eligible = (
            torch.ones_like(self.bin_sample_counts[motion_id], dtype=torch.bool, device=self.device)
            if base_eligible is None
            else torch.as_tensor(base_eligible, dtype=torch.bool, device=self.device).reshape(-1)
        )
        quarantined = self.adaptive_bin_quarantined[motion_id] & resolved_base_eligible
        eligible = resolved_base_eligible & ~quarantined
        scale = torch.where(
            self.adaptive_bin_mastered[motion_id],
            torch.full_like(
                self.adaptive_bin_mastered[motion_id],
                float(self.adaptive_sampler.mastered_probability_scale),
                dtype=torch.float32,
            ),
            torch.ones_like(self.adaptive_bin_mastered[motion_id], dtype=torch.float32),
        )
        return eligible, scale, quarantined, float(self.adaptive_sampler.probe_probability)

    def _sample_failure_weighted_motion_ids(
        self,
        env_ids: IndexLike | None = None,
        *,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        resolved_env_ids = self._normalize_env_ids(env_ids)
        num_samples = int(resolved_env_ids.numel())
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        motion_fail_counts = torch.stack([fail_counts.sum() for fail_counts in self.bin_fail_counts], dim=0)
        motion_sample_counts = torch.stack([sample_counts.sum() for sample_counts in self.bin_sample_counts], dim=0)
        eligible_mask, probability_scale, probe_mask, probe_probability = self._adaptive_motion_probability_inputs()
        motion_probs = self._build_guarded_sampling_probabilities(
            motion_fail_counts,
            motion_sample_counts,
            temperature=temperature,
            eligible_mask=eligible_mask,
            probability_scale=probability_scale,
            probe_mask=probe_mask,
            probe_probability=probe_probability,
        )
        motion_ids = torch.multinomial(motion_probs, num_samples, replacement=True)
        self._record_adaptive_probe_samples(motion_ids, probe_mask)
        return motion_ids

    def _sample_start_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.zeros(motion_ids.shape, dtype=torch.float32, device=self.device)
        if self._has_failure_bins():
            target_bin_indices = self._times_to_bins(motion_ids, times)
        else:
            target_bin_indices = torch.full(motion_ids.shape, -1, dtype=torch.long, device=self.device)
        return times, target_bin_indices

    def _sample_time_source_random_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        durations = self.motion_lib.get_duration(motion_ids)
        times = torch.rand_like(durations) * durations

        if self._has_failure_bins():
            target_bin_indices = self._times_to_bins(motion_ids, times)
        else:
            target_bin_indices = torch.full(motion_ids.shape, -1, dtype=torch.long, device=self.device)

        if not torch.any(self.motion_lib.motion_has_segments[motion_ids]):
            return times, target_bin_indices

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            if not bool(self.motion_lib.motion_has_segments[motion_id].item()):
                continue

            mask = motion_ids == motion_id
            segment_start_times = self.motion_lib.motion_segment_start_times[motion_id]
            if segment_start_times is None or segment_start_times.numel() == 0:
                continue

            segment_indices = torch.randint(
                segment_start_times.shape[0],
                (int(mask.sum().item()),),
                device=self.device,
            )
            times[mask] = segment_start_times[segment_indices]
            if self._has_failure_bins():
                target_bin_indices[mask] = segment_indices

        return times, target_bin_indices

    def _sample_anchor_source_random_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_failure_bins()

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.empty(motion_ids.shape, dtype=torch.float32, device=self.device)
        target_bin_indices = torch.empty(motion_ids.shape, dtype=torch.long, device=self.device)

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            mask = motion_ids == motion_id
            num_samples = int(mask.sum().item())
            num_anchor_bins = int(self.num_bins_per_motion[motion_id].item())
            if num_anchor_bins == 0:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one reset anchor for every motion clip."
                )
            sampled_bin_indices = torch.randint(num_anchor_bins, (num_samples,), device=self.device)
            target_bin_indices[mask] = sampled_bin_indices
            times[mask] = self.bin_reset_times[motion_id][sampled_bin_indices]

        return times, target_bin_indices

    def _sample_rand_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.segment_source == SegmentSource.Anchor:
            return self._sample_anchor_source_random_times_for_motion_ids(motion_ids)
        return self._sample_time_source_random_times_for_motion_ids(motion_ids)

    def _sample_failure_weighted_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.empty(motion_ids.shape, dtype=torch.float32, device=self.device)
        target_bin_indices = torch.empty(motion_ids.shape, dtype=torch.long, device=self.device)

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            mask = motion_ids == motion_id
            num_samples = int(mask.sum().item())
            fail_counts = self.bin_fail_counts[motion_id]
            sample_counts = self.bin_sample_counts[motion_id]
            eligible_mask = self.bin_reset_eligible[motion_id] if self.segment_source == SegmentSource.Anchor else None
            eligible_mask, probability_scale, probe_mask, probe_probability = self._adaptive_bin_probability_inputs(
                motion_id,
                eligible_mask,
            )
            probs = self._build_guarded_sampling_probabilities(
                fail_counts,
                sample_counts,
                temperature=temperature,
                eligible_mask=eligible_mask,
                probability_scale=probability_scale,
                probe_mask=probe_mask,
                probe_probability=probe_probability,
            )

            bin_indices = torch.multinomial(probs, num_samples, replacement=True)
            self._record_adaptive_probe_samples(bin_indices, probe_mask)
            target_bin_indices[mask] = bin_indices

            if self.segment_source == SegmentSource.Anchor:
                times[mask] = self.bin_reset_times[motion_id][bin_indices]
            else:
                bin_starts = self.bin_start_times[motion_id][bin_indices]
                bin_ends = self.bin_end_times[motion_id][bin_indices]
                times[mask] = bin_starts + torch.rand(num_samples, device=self.device) * (bin_ends - bin_starts)

        return times, target_bin_indices

    def _sample_times_and_target_bins_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        strategy: SamplingStrategy,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if strategy == SamplingStrategy.Start:
            return self._sample_start_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.Random:
            return self._sample_rand_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.FailureWeighted:
            return self._sample_failure_weighted_times_for_motion_ids(
                motion_ids,
                temperature=temperature,
            )
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    def sample_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        strategy: SamplingStrategy,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        times, _ = self._sample_times_and_target_bins_for_motion_ids(
            motion_ids,
            strategy,
            temperature=temperature,
        )
        return times

    def reset(
        self,
        env_ids: IndexLike | None = None,
        *,
        strategy: SamplingStrategy = SamplingStrategy.Random,
        temperature: float = 1.0,
    ) -> ResetSample:
        resolved_env_ids = self._normalize_env_ids(env_ids)
        if strategy == SamplingStrategy.FailureWeighted:
            motion_ids = self._sample_failure_weighted_motion_ids(resolved_env_ids, temperature=temperature)
        else:
            motion_ids = self.sample_motion_ids(resolved_env_ids)
        times, target_bin_indices = self._sample_times_and_target_bins_for_motion_ids(
            motion_ids,
            strategy,
            temperature=temperature,
        )
        self.current_motion_ids[resolved_env_ids] = motion_ids
        self.current_times[resolved_env_ids] = times
        self.episode_start_motion_ids[resolved_env_ids] = motion_ids
        self.episode_start_times[resolved_env_ids] = times
        self.episode_start_bin_indices[resolved_env_ids] = target_bin_indices
        self._record_sample_bins(motion_ids, times, target_bin_indices=target_bin_indices)
        return ResetSample(
            env_ids=resolved_env_ids,
            motion_ids=motion_ids,
            times=times,
            target_bin_indices=target_bin_indices,
        )

    def advance(self, env_ids: IndexLike | None = None) -> None:
        env_ids = self._normalize_env_ids(env_ids)
        self.current_times[env_ids] += self.dt

    def sample_motion_batch(
        self,
        env_ids: IndexLike | None = None,
        *,
        position_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        env_ids = self._normalize_env_ids(env_ids)
        motion_ids = self.current_motion_ids[env_ids]
        times = self.current_times[env_ids]
        return self.motion_lib.sample_motion(
            motion_ids=motion_ids,
            times=times,
            position_offsets=position_offsets,
        )

    def _build_time_source_bins_for_clip(
        self,
        clip: MotionClip,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        if clip.has_segments:
            return (
                clip.segment_start_times.to(device=self.device),
                clip.segment_end_times.to(device=self.device),
                clip.segment_types.to(device=self.device),
                True,
            )
        if self.bin_size is None:
            raise RuntimeError("Time-segment target bins require either segment metadata or bin_size.")

        start_times_np, end_times_np, segment_types_np = build_legacy_time_segments(
            duration=clip.duration,
            bin_size=self.bin_size,
        )
        return (
            torch.as_tensor(start_times_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(end_times_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(segment_types_np, dtype=torch.long, device=self.device),
            False,
        )

    def _build_anchor_source_bins_for_clip(
        self,
        clip: MotionClip,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not clip.has_anchor_segments:
            raise self._anchor_sampling_error(
                "Anchor segment sampling requires motion clips with anchor metadata."
            )

        anchor_times = clip.anchor_times.to(device=self.device)
        if anchor_times.numel() == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return (
                empty,
                empty,
                torch.empty(0, dtype=torch.long, device=self.device),
                empty,
                torch.empty(0, dtype=torch.bool, device=self.device),
            )

        start_times = anchor_times.clone()
        start_times[0] = 0.0

        end_times = torch.empty_like(anchor_times)
        if anchor_times.numel() > 1:
            end_times[:-1] = anchor_times[1:]
        end_times[-1] = float(clip.duration)

        segment_end_times = clip.anchor_segment_end_times.to(device=self.device)
        segment_labels = clip.anchor_segment_labels.to(device=self.device)
        segment_indices = torch.searchsorted(segment_end_times, anchor_times, right=True)
        segment_indices = torch.clamp(segment_indices, max=max(int(segment_labels.numel()) - 1, 0))
        anchor_labels = segment_labels[segment_indices]

        reset_times = anchor_times.clone()
        eligible = torch.ones(anchor_times.shape, dtype=torch.bool, device=self.device)
        return start_times, end_times, anchor_labels, reset_times, eligible

    def init_failure_bins(self, bin_size: float | None = None) -> None:
        if bin_size is not None and bin_size <= 0.0:
            raise ValueError("bin_size must be > 0")

        self.bin_size = float(bin_size) if bin_size is not None else None
        self.bin_start_times = []
        self.bin_end_times = []
        self.bin_types = []
        self.bin_reset_times = []
        self.bin_reset_eligible = []
        self.bin_uses_segment_metadata = []
        num_bins_per_motion: list[int] = []

        for clip in self.motion_lib.clips:
            if self.segment_source == SegmentSource.Anchor:
                start_times, end_times, segment_types, reset_times, reset_eligible = (
                    self._build_anchor_source_bins_for_clip(clip)
                )
                self.bin_uses_segment_metadata.append(True)
            else:
                start_times, end_times, segment_types, uses_segment_metadata = self._build_time_source_bins_for_clip(
                    clip
                )
                reset_times = start_times.clone()
                reset_eligible = torch.ones(start_times.shape, dtype=torch.bool, device=self.device)
                self.bin_uses_segment_metadata.append(uses_segment_metadata)

            self.bin_start_times.append(start_times)
            self.bin_end_times.append(end_times)
            self.bin_types.append(segment_types)
            self.bin_reset_times.append(reset_times)
            self.bin_reset_eligible.append(reset_eligible)
            num_bins_per_motion.append(int(start_times.shape[0]))

        self.num_bins_per_motion = torch.as_tensor(num_bins_per_motion, dtype=torch.long, device=self.device)
        self.num_bins = int(self.num_bins_per_motion.sum().item())
        self.bin_fail_counts = [
            torch.zeros(int(num_bins), dtype=torch.float32, device=self.device)
            for num_bins in self.num_bins_per_motion.tolist()
        ]
        self.bin_sample_counts = [
            torch.zeros(int(num_bins), dtype=torch.float32, device=self.device)
            for num_bins in self.num_bins_per_motion.tolist()
        ]
        self.supports_failure_weighted_sampling = all(self.bin_uses_segment_metadata)
        self._init_adaptive_state()

        if self.segment_source == SegmentSource.Anchor:
            clips_without_eligible_bins = [
                clip.name
                for clip, reset_eligible in zip(self.motion_lib.clips, self.bin_reset_eligible, strict=False)
                if not bool(torch.any(reset_eligible).item())
            ]
            if clips_without_eligible_bins:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one eligible reset anchor for every motion clip. "
                    f"Clips without eligible reset anchors: {', '.join(clips_without_eligible_bins)}"
                )

    def reset_failure_stats(self) -> None:
        self._check_failure_bins()
        for fail_counts, sample_counts in zip(self.bin_fail_counts, self.bin_sample_counts, strict=False):
            fail_counts.zero_()
            sample_counts.zero_()
        self._init_adaptive_state()

    def record_failures(
        self,
        env_ids: IndexLike | None = None,
        times: torch.Tensor | None = None,
    ) -> None:
        if not self._has_failure_bins():
            return

        env_ids = self._normalize_env_ids(env_ids)
        if self.segment_source == SegmentSource.Anchor:
            if times is None:
                times = self.current_times[env_ids]
            else:
                times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
                if times.shape[0] != env_ids.shape[0]:
                    raise ValueError("times must have the same batch size as env_ids.")
            motion_ids = self.current_motion_ids[env_ids]
            # Anchor bins cover the interval until the next anchor, so binning the
            # failure time attributes the failure to the nearest previous anchor.
            bin_indices = self._times_to_bins(motion_ids, times)
        else:
            if times is None:
                times = self.current_times[env_ids]
            else:
                times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
                if times.shape[0] != env_ids.shape[0]:
                    raise ValueError("times must have the same batch size as env_ids.")
            motion_ids = self.current_motion_ids[env_ids]
            bin_indices = self._times_to_bins(motion_ids, times)

        self._accumulate_bin_counts(self.bin_fail_counts, motion_ids, bin_indices)
        self._update_adaptive_state()

    def _has_failure_bins(self) -> bool:
        return (
            self.num_bins_per_motion is not None
            and self.bin_start_times is not None
            and self.bin_end_times is not None
            and self.bin_types is not None
            and self.bin_reset_times is not None
            and self.bin_reset_eligible is not None
            and self.bin_uses_segment_metadata is not None
            and self.bin_fail_counts is not None
            and self.bin_sample_counts is not None
        )

    def _check_failure_bins(self) -> None:
        if not self._has_failure_bins():
            raise RuntimeError("Failure bins not initialized. Call init_failure_bins(...) first.")

    def _anchor_sampling_error(self, message: str) -> RuntimeError:
        return RuntimeError(
            f"{message} Reconvert the motion .npz files with `ref2act-convert --segment-method anchor`."
        )

    def _check_failure_weighted_support(self) -> None:
        if not self.supports_failure_weighted_sampling:
            if self.segment_source == SegmentSource.Anchor:
                raise self._anchor_sampling_error(
                    "Failure-weighted anchor sampling requires motion clips with anchor metadata."
                )
            raise RuntimeError(
                "Failure-weighted sampling requires motion clips with segment metadata. "
                "Reconvert the motion .npz files with `ref2act-convert --segment-bin-size ...`."
            )

    def _times_to_bins(self, motion_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1)
        bin_indices = torch.empty_like(motion_ids)

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            motion_mask = motion_ids == motion_id
            end_times = self.bin_end_times[motion_id]
            motion_times = torch.clamp(times[motion_mask], min=0.0)
            motion_bin_indices = torch.searchsorted(end_times, motion_times, right=True)
            max_bin_index = int(self.num_bins_per_motion[motion_id].item()) - 1
            bin_indices[motion_mask] = torch.clamp(motion_bin_indices, max=max_bin_index)

        return bin_indices

    def _accumulate_bin_counts(
        self,
        counts_by_motion: list[torch.Tensor],
        motion_ids: torch.Tensor,
        bin_indices: torch.Tensor,
    ) -> None:
        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            motion_mask = motion_ids == motion_id
            counts_by_motion[motion_id] += torch.bincount(
                bin_indices[motion_mask],
                minlength=int(self.num_bins_per_motion[motion_id].item()),
            ).to(device=self.device, dtype=torch.float32)

    def _record_sample_bins(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        *,
        target_bin_indices: torch.Tensor | None = None,
    ) -> None:
        if not self._has_failure_bins():
            return
        if motion_ids.numel() == 0:
            return

        if target_bin_indices is None:
            target_bin_indices = self._times_to_bins(motion_ids, times)
        else:
            target_bin_indices = torch.as_tensor(target_bin_indices, dtype=torch.long, device=self.device).reshape(-1)
            if target_bin_indices.shape != motion_ids.shape:
                raise ValueError("target_bin_indices must have the same shape as motion_ids.")

        if self.failure_decay < 1.0:
            for fail_counts, sample_counts in zip(self.bin_fail_counts, self.bin_sample_counts, strict=False):
                fail_counts.mul_(self.failure_decay)
                sample_counts.mul_(self.failure_decay)
        self._accumulate_bin_counts(self.bin_sample_counts, motion_ids, target_bin_indices)
        self._advance_adaptive_cooldowns()
        self._update_adaptive_state()

    def get_adaptive_stats(self) -> dict[str, torch.Tensor]:
        if not self._has_adaptive_state():
            return {
                "adaptive_enabled": torch.tensor(float(self.adaptive_sampler.enabled), device=self.device),
                "adaptive_live_motion_count": torch.tensor(float(self.motion_lib.num_motions), device=self.device),
                "adaptive_quarantined_motion_count": torch.tensor(0.0, device=self.device),
                "adaptive_mastered_motion_count": torch.tensor(0.0, device=self.device),
                "adaptive_live_anchor_count": torch.tensor(float(self.num_bins), device=self.device),
                "adaptive_quarantined_anchor_count": torch.tensor(0.0, device=self.device),
                "adaptive_mastered_anchor_count": torch.tensor(0.0, device=self.device),
                "adaptive_probe_sample_count": torch.tensor(
                    float(self.adaptive_probe_sample_count),
                    device=self.device,
                ),
            }

        motion_quarantined_count = torch.sum(self.adaptive_motion_quarantined.to(dtype=torch.float32))
        motion_mastered_count = torch.sum(self.adaptive_motion_mastered.to(dtype=torch.float32))
        base_anchor_count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        quarantined_anchor_count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        mastered_anchor_count = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for motion_id in range(self.motion_lib.num_motions):
            base_eligible = self._adaptive_base_bin_eligible(motion_id)
            base_anchor_count += torch.sum(base_eligible.to(dtype=torch.float32))
            quarantined_anchor_count += torch.sum(
                (self.adaptive_bin_quarantined[motion_id] & base_eligible).to(dtype=torch.float32)
            )
            mastered_anchor_count += torch.sum(
                (self.adaptive_bin_mastered[motion_id] & base_eligible).to(dtype=torch.float32)
            )

        return {
            "adaptive_enabled": torch.tensor(1.0, device=self.device),
            "adaptive_live_motion_count": torch.tensor(
                float(self.motion_lib.num_motions),
                dtype=torch.float32,
                device=self.device,
            )
            - motion_quarantined_count,
            "adaptive_quarantined_motion_count": motion_quarantined_count,
            "adaptive_mastered_motion_count": motion_mastered_count,
            "adaptive_live_anchor_count": base_anchor_count - quarantined_anchor_count,
            "adaptive_quarantined_anchor_count": quarantined_anchor_count,
            "adaptive_mastered_anchor_count": mastered_anchor_count,
            "adaptive_probe_sample_count": torch.tensor(
                float(self.adaptive_probe_sample_count),
                dtype=torch.float32,
                device=self.device,
            ),
        }


Sampler = MotionSampler

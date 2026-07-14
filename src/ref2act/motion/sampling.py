from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import torch
import torch.nn.functional as F

from ref2act.common.utils import IndexLike

from .library import MotionClip, MotionLib
from .segments import build_time_bins


DEFAULT_ANCHOR_FAILURE_BIN_SIZE = 0.3
FAILURE_BIN_PROGRESS_CLIPS = 100
FAILURE_BIN_PROGRESS_SECONDS = 5.0
ANCHOR_SKIPPED_CLIP_LOG_LIMIT = 10


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


class MotionSampler:
    def __init__(
        self,
        num_envs: int,
        motion_lib: MotionLib,
        dt: float,
        *,
        anchor_body_index: int | None = None,
        bin_size: float | None = None,
        weight_fail: float = 0.5,
        weight_novel: float = 0.3,
        cap_beta: float = 2.0,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        motion_sampling_warmup_s: float = 0.0,
        motion_sampling_ramp_s: float = 0.0,
        motion_sampling_schedule: str = "cosine",
        segment_source: SegmentSource = SegmentSource.Time,
        enable_failure_bins: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.motion_lib = motion_lib
        self.anchor_body_index = anchor_body_index
        self.segment_source = segment_source
        self.weight_fail = self._validate_nonnegative_float("weight_fail", weight_fail)
        self.weight_novel = self._validate_nonnegative_float("weight_novel", weight_novel)
        self.cap_beta = self._validate_nonnegative_float("cap_beta", cap_beta)
        self.adaptive_uniform_ratio = self._validate_nonnegative_float(
            "adaptive_uniform_ratio",
            adaptive_uniform_ratio,
        )
        self.adaptive_alpha = self._validate_unit_interval_float("adaptive_alpha", adaptive_alpha)
        self.adaptive_kernel_size = int(adaptive_kernel_size)
        if self.adaptive_kernel_size < 1:
            raise ValueError("adaptive_kernel_size must be >= 1.")
        self.adaptive_lambda = self._validate_nonnegative_float("adaptive_lambda", adaptive_lambda)
        self.motion_sampling_warmup_s = self._validate_nonnegative_float(
            "motion_sampling_warmup_s",
            motion_sampling_warmup_s,
        )
        self.motion_sampling_ramp_s = self._validate_nonnegative_float(
            "motion_sampling_ramp_s",
            motion_sampling_ramp_s,
        )
        self.motion_sampling_schedule = str(motion_sampling_schedule).lower()
        if self.motion_sampling_schedule not in {"linear", "cosine"}:
            raise ValueError("motion_sampling_schedule must be 'linear' or 'cosine'.")
        self._global_step = 0
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel = self.kernel / torch.clamp(self.kernel.sum(), min=torch.finfo(self.kernel.dtype).eps)

        self.current_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_start_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_bin_indices = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.episode_start_sampling_strategy_values = torch.full(
            (num_envs,),
            SamplingStrategy.Random.value,
            dtype=torch.long,
            device=self.device,
        )
        self.motion_sample_counts = torch.zeros(self.motion_lib.num_motions, dtype=torch.float32, device=self.device)
        self.motion_assigned_counts = torch.zeros(self.motion_lib.num_motions, dtype=torch.float32, device=self.device)
        self.motion_fail_counts = torch.zeros(self.motion_lib.num_motions, dtype=torch.float32, device=self.device)

        self.bin_size: float | None = float(bin_size) if bin_size is not None else None
        self.num_bins = 0
        self.num_bins_per_motion: torch.Tensor | None = None
        self.bin_start_times: list[torch.Tensor] | None = None
        self.bin_end_times: list[torch.Tensor] | None = None
        self.bin_types: list[torch.Tensor] | None = None
        self.bin_reset_times: list[torch.Tensor] | None = None
        self.bin_reset_eligible: list[torch.Tensor] | None = None
        self.bin_fail_counts: list[torch.Tensor] | None = None
        self.bin_sample_counts: list[torch.Tensor] | None = None
        self.motion_reset_eligible: torch.Tensor | None = None
        self.max_bins_per_motion = 0
        self._bin_start_times_padded: torch.Tensor | None = None
        self._bin_end_times_padded: torch.Tensor | None = None
        self._bin_types_padded: torch.Tensor | None = None
        self._bin_reset_times_padded: torch.Tensor | None = None
        self._bin_reset_eligible_padded: torch.Tensor | None = None
        self._bin_valid_mask_padded: torch.Tensor | None = None
        self._bin_fail_counts_padded: torch.Tensor | None = None
        self._bin_sample_counts_padded: torch.Tensor | None = None
        self.supports_failure_weighted_sampling = False

        should_init_failure_bins = (
            bool(enable_failure_bins)
            and (
                self.segment_source == SegmentSource.Anchor
                or self.bin_size is not None
            )
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
        if self.segment_source == SegmentSource.Anchor and self.motion_reset_eligible is not None:
            eligible_motion_ids = torch.nonzero(self.motion_reset_eligible, as_tuple=False).reshape(-1)
            if eligible_motion_ids.numel() == 0:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one motion clip with an eligible reset anchor."
                )
            sampled_offsets = torch.randint(eligible_motion_ids.numel(), (env_ids.numel(),), device=self.device)
            return eligible_motion_ids[sampled_offsets]
        return torch.randint(self.motion_lib.num_motions, (env_ids.numel(),), device=self.device)

    @staticmethod
    def _validate_nonnegative_float(name: str, value: float) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved < 0.0:
            raise ValueError(f"{name} must be a finite float >= 0.")
        return resolved

    @staticmethod
    def _validate_unit_interval_float(name: str, value: float) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
            raise ValueError(f"{name} must be a finite float in [0, 1].")
        return resolved

    def _motion_sampling_progress(self) -> float:
        elapsed_s = float(self._global_step) * float(self.dt)
        if elapsed_s <= self.motion_sampling_warmup_s:
            return 0.0
        if self.motion_sampling_ramp_s <= 0.0:
            return 1.0

        x = (elapsed_s - self.motion_sampling_warmup_s) / self.motion_sampling_ramp_s
        x = max(0.0, min(1.0, x))
        if self.motion_sampling_schedule == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * x)
        return x

    def _in_motion_sampling_warmup(self) -> bool:
        elapsed_s = float(self._global_step) * float(self.dt)
        return elapsed_s <= self.motion_sampling_warmup_s

    def _mix_sampling_terms(
        self,
        fail_probs: torch.Tensor,
        novel_probs: torch.Tensor,
        uniform_probs: torch.Tensor,
    ) -> torch.Tensor:
        progress = self._motion_sampling_progress()
        w_fail = progress * self.weight_fail
        w_novel = progress * self.weight_novel
        w_sum = w_fail + w_novel
        if w_sum > 1.0:
            w_fail = w_fail / w_sum
            w_novel = w_novel / w_sum
            w_uniform = 0.0
        else:
            w_uniform = max(0.0, 1.0 - w_fail - w_novel)

        probs = w_fail * fail_probs + w_novel * novel_probs + w_uniform * uniform_probs
        eps = torch.finfo(probs.dtype).eps
        if probs.ndim == 1:
            probs_sum = torch.sum(probs)
            if float(probs_sum.item()) <= 0.0 or not bool(torch.isfinite(probs_sum).item()):
                return uniform_probs
            return probs / torch.clamp(probs_sum, min=eps)

        probs_sum = torch.sum(probs, dim=1, keepdim=True)
        valid = (probs_sum > 0.0) & torch.isfinite(probs_sum)
        return torch.where(valid, probs / torch.clamp(probs_sum, min=eps), uniform_probs)

    def _uniform_probabilities(self, eligible: torch.Tensor) -> torch.Tensor:
        eligible_float = eligible.to(dtype=torch.float32)
        if eligible.ndim == 1:
            eligible_count = torch.sum(eligible_float)
            if float(eligible_count.item()) <= 0.0:
                raise ValueError("eligible_mask must include at least one entry.")
            return eligible_float / eligible_count

        eligible_count = torch.sum(eligible_float, dim=1, keepdim=True)
        if bool(torch.any(eligible_count <= 0.0).item()):
            raise ValueError("eligible_mask must include at least one entry in every row.")
        return eligible_float / torch.clamp(eligible_count, min=torch.finfo(eligible_float.dtype).eps)

    def _novel_probabilities(self, assigned_counts: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        assigned_counts = torch.clamp(assigned_counts.to(dtype=torch.float32), min=0.0)
        scores = 1.0 / torch.sqrt(assigned_counts + 1.0)
        scores = torch.where(eligible, scores, torch.zeros_like(scores))

        if assigned_counts.ndim == 1:
            score_sum = torch.sum(scores)
            if float(score_sum.item()) <= 0.0 or not bool(torch.isfinite(score_sum).item()):
                return self._uniform_probabilities(eligible)
            return scores / score_sum

        score_sum = torch.sum(scores, dim=1, keepdim=True)
        uniform_probs = self._uniform_probabilities(eligible)
        valid = (score_sum > 0.0) & torch.isfinite(score_sum)
        return torch.where(valid, scores / torch.clamp(score_sum, min=torch.finfo(scores.dtype).eps), uniform_probs)

    def _build_motion_sampling_probabilities(
        self,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fail_counts = self.motion_fail_counts.to(dtype=torch.float32)
        sample_counts = self.motion_sample_counts.to(dtype=torch.float32)
        assigned_counts = self.motion_assigned_counts.to(dtype=torch.float32)
        if eligible_mask is None:
            eligible = torch.ones_like(fail_counts, dtype=torch.bool, device=self.device)
        else:
            eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device).reshape(-1)
            if eligible.shape != fail_counts.shape:
                raise ValueError("eligible_mask must have the same shape as motion fail counts.")

        uniform_probs = self._uniform_probabilities(eligible)
        fail_rates = fail_counts / torch.clamp(sample_counts, min=1.0)
        fail_rates = torch.where(eligible, fail_rates, torch.zeros_like(fail_rates))
        if bool(torch.any(eligible).item()):
            mean_fail = fail_rates[eligible].mean()
        else:
            mean_fail = fail_rates.new_tensor(0.0)
        beta_cap = self.cap_beta * mean_fail
        if float(beta_cap.item()) > 0.0:
            capped_rates = torch.minimum(fail_rates, beta_cap)
        else:
            capped_rates = torch.zeros_like(fail_rates)
        capped_rates = torch.where(eligible, capped_rates, torch.zeros_like(capped_rates))
        capped_sum = torch.sum(capped_rates)
        if float(capped_sum.item()) > 0.0 and bool(torch.isfinite(capped_sum).item()):
            fail_probs = capped_rates / capped_sum
        else:
            fail_probs = torch.zeros_like(capped_rates)

        novel_probs = self._novel_probabilities(assigned_counts, eligible)
        return self._mix_sampling_terms(fail_probs, novel_probs, uniform_probs)

    def _smooth_bin_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.adaptive_kernel_size <= 1:
            return weights
        if weights.ndim == 1:
            smoothed = F.conv1d(
                F.pad(weights.view(1, 1, -1), (0, self.adaptive_kernel_size - 1), mode="replicate"),
                self.kernel.view(1, 1, -1),
            ).view(-1)
            return smoothed
        return F.conv1d(
            F.pad(weights.unsqueeze(1), (0, self.adaptive_kernel_size - 1), mode="replicate"),
            self.kernel.view(1, 1, -1),
        ).squeeze(1)

    def _build_bin_sampling_probabilities(
        self,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fail_counts = torch.as_tensor(fail_counts, dtype=torch.float32, device=self.device)
        sample_counts = torch.as_tensor(sample_counts, dtype=torch.float32, device=self.device)
        if fail_counts.shape != sample_counts.shape:
            raise ValueError("fail_counts and sample_counts must have the same shape.")
        if eligible_mask is None:
            eligible = torch.ones_like(fail_counts, dtype=torch.bool, device=self.device)
        else:
            eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device)
            if eligible.shape != fail_counts.shape:
                raise ValueError("eligible_mask must have the same shape as fail_counts.")

        # Match MOSAIC: bin-level sampling ignores sample counts and uses failure counts plus a uniform floor.
        del sample_counts
        uniform_probs = self._uniform_probabilities(eligible)
        eligible_float = eligible.to(dtype=torch.float32)
        if fail_counts.ndim == 1:
            eligible_count = torch.sum(eligible_float)
            uniform_floor = self.adaptive_uniform_ratio / torch.clamp(
                eligible_count,
                min=torch.finfo(fail_counts.dtype).eps,
            )
        else:
            eligible_count = torch.sum(eligible_float, dim=1, keepdim=True)
            uniform_floor = self.adaptive_uniform_ratio / torch.clamp(
                eligible_count,
                min=torch.finfo(fail_counts.dtype).eps,
            )

        fail_weights = (torch.clamp(fail_counts, min=0.0) + uniform_floor) * eligible_float
        fail_weights = self._smooth_bin_weights(fail_weights)
        fail_weights = torch.where(eligible, fail_weights, torch.zeros_like(fail_weights))
        if fail_counts.ndim == 1:
            fail_sum = torch.sum(fail_weights)
            if float(fail_sum.item()) > 0.0 and bool(torch.isfinite(fail_sum).item()):
                fail_probs = fail_weights / fail_sum
            else:
                fail_probs = uniform_probs
        else:
            fail_sum = torch.sum(fail_weights, dim=1, keepdim=True)
            valid = (fail_sum > 0.0) & torch.isfinite(fail_sum)
            fail_probs = torch.where(
                valid,
                fail_weights / torch.clamp(fail_sum, min=torch.finfo(fail_weights.dtype).eps),
                uniform_probs,
            )

        return fail_probs

    def _sample_failure_weighted_motion_ids(
        self,
        env_ids: IndexLike | None = None,
    ) -> torch.Tensor:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        resolved_env_ids = self._normalize_env_ids(env_ids)
        num_samples = int(resolved_env_ids.numel())
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        motion_eligible_mask = self.motion_reset_eligible if self.segment_source == SegmentSource.Anchor else None
        motion_probs = self._build_motion_sampling_probabilities(eligible_mask=motion_eligible_mask)
        return torch.multinomial(motion_probs, num_samples, replacement=True)

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

        return times, target_bin_indices

    def _sample_anchor_source_random_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_failure_bins()

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        self._check_anchor_motion_ids_have_eligible_resets(motion_ids)
        times = torch.empty(motion_ids.shape, dtype=torch.float32, device=self.device)
        target_bin_indices = torch.empty(motion_ids.shape, dtype=torch.long, device=self.device)

        if (
            self._has_padded_failure_bins()
            and self._bin_reset_times_padded is not None
            and self._bin_valid_mask_padded is not None
            and self._bin_reset_eligible_padded is not None
        ):
            eligible = self._bin_valid_mask_padded[motion_ids] & self._bin_reset_eligible_padded[motion_ids]
            eligible_count = torch.sum(eligible.to(dtype=torch.float32), dim=1, keepdim=True)
            if bool(torch.any(eligible_count <= 0).item()):
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one eligible reset anchor for every motion clip."
                )
            probs = eligible.to(dtype=torch.float32) / eligible_count
            target_bin_indices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
            times = self._bin_reset_times_padded[motion_ids, target_bin_indices]
            return times, target_bin_indices

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            mask = motion_ids == motion_id
            num_samples = int(mask.sum().item())
            eligible_indices = torch.nonzero(self.bin_reset_eligible[motion_id], as_tuple=False).reshape(-1)
            if eligible_indices.numel() == 0:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one eligible reset anchor for every motion clip."
                )
            sampled_offsets = torch.randint(eligible_indices.numel(), (num_samples,), device=self.device)
            sampled_bin_indices = eligible_indices[sampled_offsets]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if self.segment_source == SegmentSource.Anchor:
            self._check_anchor_motion_ids_have_eligible_resets(motion_ids)
        times = torch.empty(motion_ids.shape, dtype=torch.float32, device=self.device)
        target_bin_indices = torch.empty(motion_ids.shape, dtype=torch.long, device=self.device)

        if (
            self._has_padded_failure_bins()
            and self._bin_fail_counts_padded is not None
            and self._bin_sample_counts_padded is not None
            and self._bin_valid_mask_padded is not None
            and self._bin_start_times_padded is not None
            and self._bin_end_times_padded is not None
            and self._bin_reset_times_padded is not None
            and self._bin_reset_eligible_padded is not None
            and self.max_bins_per_motion > 0
        ):
            fail_counts = self._bin_fail_counts_padded[motion_ids]
            sample_counts = self._bin_sample_counts_padded[motion_ids]
            if self.segment_source == SegmentSource.Anchor:
                base_eligible = self._bin_reset_eligible_padded[motion_ids] & self._bin_valid_mask_padded[motion_ids]
            else:
                base_eligible = self._bin_valid_mask_padded[motion_ids]
            probs = self._build_bin_sampling_probabilities(
                fail_counts,
                sample_counts,
                eligible_mask=base_eligible,
            )
            target_bin_indices = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)

            if self.segment_source == SegmentSource.Anchor:
                times = self._bin_reset_times_padded[motion_ids, target_bin_indices]
            else:
                bin_starts = self._bin_start_times_padded[motion_ids, target_bin_indices]
                bin_ends = self._bin_end_times_padded[motion_ids, target_bin_indices]
                times = bin_starts + torch.rand(motion_ids.shape, device=self.device) * (bin_ends - bin_starts)
            return times, target_bin_indices

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            mask = motion_ids == motion_id
            num_samples = int(mask.sum().item())
            fail_counts = self.bin_fail_counts[motion_id]
            sample_counts = self.bin_sample_counts[motion_id]
            eligible_mask = self.bin_reset_eligible[motion_id] if self.segment_source == SegmentSource.Anchor else None
            probs = self._build_bin_sampling_probabilities(
                fail_counts,
                sample_counts,
                eligible_mask=eligible_mask,
            )

            bin_indices = torch.multinomial(probs, num_samples, replacement=True)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if strategy == SamplingStrategy.Start:
            return self._sample_start_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.Random:
            return self._sample_rand_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.FailureWeighted:
            return self._sample_failure_weighted_times_for_motion_ids(motion_ids)
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    def sample_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        strategy: SamplingStrategy,
    ) -> torch.Tensor:
        times, _ = self._sample_times_and_target_bins_for_motion_ids(
            motion_ids,
            strategy,
        )
        return times

    def reset(
        self,
        env_ids: IndexLike | None = None,
        *,
        strategy: SamplingStrategy = SamplingStrategy.Random,
    ) -> ResetSample:
        resolved_env_ids = self._normalize_env_ids(env_ids)
        if strategy == SamplingStrategy.FailureWeighted:
            motion_ids = self._sample_failure_weighted_motion_ids(resolved_env_ids)
        else:
            motion_ids = self.sample_motion_ids(resolved_env_ids)
        times, target_bin_indices = self._sample_times_and_target_bins_for_motion_ids(
            motion_ids,
            strategy,
        )
        self.current_motion_ids[resolved_env_ids] = motion_ids
        self.current_times[resolved_env_ids] = times
        self.episode_start_motion_ids[resolved_env_ids] = motion_ids
        self.episode_start_times[resolved_env_ids] = times
        self.episode_start_bin_indices[resolved_env_ids] = target_bin_indices
        self.episode_start_sampling_strategy_values[resolved_env_ids] = strategy.value
        self._record_motion_assignments(motion_ids)
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
        self._global_step += 1

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bin_size is None:
            raise RuntimeError("Time-segment target bins require bin_size.")

        start_times_np, end_times_np, segment_types_np = build_time_bins(
            duration=clip.duration,
            bin_size=self.bin_size,
        )
        return (
            torch.as_tensor(start_times_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(end_times_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(segment_types_np, dtype=torch.long, device=self.device),
        )

    def _build_anchor_source_bins_for_clip(
        self,
        clip: MotionClip,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if clip.anchor_frame_indices is None or clip.anchor_times is None:
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

        bin_size = self.bin_size if self.bin_size is not None else DEFAULT_ANCHOR_FAILURE_BIN_SIZE
        clip_duration = float(clip.duration)
        span_start_times = anchor_times.clone()
        span_start_times[0] = 0.0
        if anchor_times.numel() > 1:
            span_end_times = torch.cat(
                (
                    anchor_times[1:],
                    torch.as_tensor([clip_duration], dtype=torch.float32, device=self.device),
                ),
                dim=0,
            )
        else:
            span_end_times = torch.as_tensor([clip_duration], dtype=torch.float32, device=self.device)

        span_start_times = torch.clamp(span_start_times, min=0.0, max=clip_duration)
        span_end_times = torch.clamp(span_end_times, min=0.0, max=clip_duration)
        positive_span_mask = span_end_times > span_start_times
        if not bool(torch.any(positive_span_mask).item()):
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return (
                empty,
                empty,
                torch.empty(0, dtype=torch.long, device=self.device),
                empty,
                torch.empty(0, dtype=torch.bool, device=self.device),
            )

        span_start_times = span_start_times[positive_span_mask]
        span_end_times = span_end_times[positive_span_mask]
        span_anchor_times = anchor_times[positive_span_mask]

        span_lengths = span_end_times - span_start_times
        num_sub_bins = torch.clamp(
            torch.ceil(span_lengths / float(bin_size) - 1.0e-6).to(dtype=torch.long),
            min=1,
        )
        total_bins = int(torch.sum(num_sub_bins).item())
        if total_bins <= 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return (
                empty,
                empty,
                torch.empty(0, dtype=torch.long, device=self.device),
                empty,
                torch.empty(0, dtype=torch.bool, device=self.device),
            )

        span_indices = torch.repeat_interleave(
            torch.arange(num_sub_bins.shape[0], dtype=torch.long, device=self.device),
            num_sub_bins,
        )
        span_first_bin_offsets = torch.repeat_interleave(
            torch.cumsum(num_sub_bins, dim=0) - num_sub_bins,
            num_sub_bins,
        )
        sub_bin_indices = torch.arange(total_bins, dtype=torch.long, device=self.device) - span_first_bin_offsets
        start_times = span_start_times[span_indices] + sub_bin_indices.to(dtype=torch.float32) * float(bin_size)
        end_times = torch.minimum(start_times + float(bin_size), span_end_times[span_indices])
        positive_bin_mask = end_times > start_times
        if not bool(torch.all(positive_bin_mask).item()):
            start_times = start_times[positive_bin_mask]
            end_times = end_times[positive_bin_mask]
            span_indices = span_indices[positive_bin_mask]

        return (
            start_times.to(dtype=torch.float32),
            end_times.to(dtype=torch.float32),
            torch.zeros(start_times.shape, dtype=torch.long, device=self.device),
            span_anchor_times[span_indices].to(dtype=torch.float32),
            torch.ones(start_times.shape, dtype=torch.bool, device=self.device),
        )

    def _build_padded_bin_tensors(self) -> None:
        if self.num_bins_per_motion is None:
            raise RuntimeError("num_bins_per_motion must be initialized before padded bin tensors.")
        if (
            self.bin_start_times is None
            or self.bin_end_times is None
            or self.bin_types is None
            or self.bin_reset_times is None
            or self.bin_reset_eligible is None
        ):
            raise RuntimeError("Bin metadata lists must be initialized before padded bin tensors.")

        num_motions = self.motion_lib.num_motions
        self.max_bins_per_motion = int(self.num_bins_per_motion.max().item()) if num_motions > 0 else 0
        shape = (num_motions, self.max_bins_per_motion)
        self._bin_start_times_padded = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._bin_end_times_padded = torch.full(shape, float("inf"), dtype=torch.float32, device=self.device)
        self._bin_types_padded = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._bin_reset_times_padded = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._bin_reset_eligible_padded = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._bin_valid_mask_padded = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._bin_fail_counts_padded = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._bin_sample_counts_padded = torch.zeros(shape, dtype=torch.float32, device=self.device)

        for motion_id, num_bins in enumerate(self.num_bins_per_motion.tolist()):
            if num_bins == 0:
                continue
            bin_slice = slice(0, int(num_bins))
            self._bin_start_times_padded[motion_id, bin_slice] = self.bin_start_times[motion_id]
            self._bin_end_times_padded[motion_id, bin_slice] = self.bin_end_times[motion_id]
            self._bin_types_padded[motion_id, bin_slice] = self.bin_types[motion_id]
            self._bin_reset_times_padded[motion_id, bin_slice] = self.bin_reset_times[motion_id]
            self._bin_reset_eligible_padded[motion_id, bin_slice] = self.bin_reset_eligible[motion_id]
            self._bin_valid_mask_padded[motion_id, bin_slice] = True

        self.bin_fail_counts = [
            self._bin_fail_counts_padded[motion_id, : int(num_bins)]
            for motion_id, num_bins in enumerate(self.num_bins_per_motion.tolist())
        ]
        self.bin_sample_counts = [
            self._bin_sample_counts_padded[motion_id, : int(num_bins)]
            for motion_id, num_bins in enumerate(self.num_bins_per_motion.tolist())
        ]

    def _has_padded_failure_bins(self) -> bool:
        return (
            self._bin_start_times_padded is not None
            and self._bin_end_times_padded is not None
            and self._bin_reset_times_padded is not None
            and self._bin_reset_eligible_padded is not None
            and self._bin_valid_mask_padded is not None
            and self._bin_fail_counts_padded is not None
            and self._bin_sample_counts_padded is not None
        )

    def init_failure_bins(self, bin_size: float | None = None) -> None:
        if bin_size is not None and bin_size <= 0.0:
            raise ValueError("bin_size must be > 0")

        self.bin_size = float(bin_size) if bin_size is not None else None
        self.bin_start_times = []
        self.bin_end_times = []
        self.bin_types = []
        self.bin_reset_times = []
        self.bin_reset_eligible = []
        num_bins_per_motion: list[int] = []
        init_start_time = perf_counter()
        last_progress_time = init_start_time
        total_clips = len(self.motion_lib.clips)
        print(
            "initializing motion failure bins: "
            f"source={self.segment_source}, clips={total_clips}, bin_size={self.bin_size}",
            flush=True,
        )

        for clip_index, clip in enumerate(self.motion_lib.clips, start=1):
            if self.segment_source == SegmentSource.Anchor:
                start_times, end_times, segment_types, reset_times, reset_eligible = (
                    self._build_anchor_source_bins_for_clip(clip)
                )
            else:
                start_times, end_times, segment_types = self._build_time_source_bins_for_clip(clip)
                reset_times = start_times.clone()
                reset_eligible = torch.ones(start_times.shape, dtype=torch.bool, device=self.device)

            self.bin_start_times.append(start_times)
            self.bin_end_times.append(end_times)
            self.bin_types.append(segment_types)
            self.bin_reset_times.append(reset_times)
            self.bin_reset_eligible.append(reset_eligible)
            num_bins_per_motion.append(int(start_times.shape[0]))
            now = perf_counter()
            should_log_progress = (
                clip_index == total_clips
                or (FAILURE_BIN_PROGRESS_CLIPS > 0 and clip_index % FAILURE_BIN_PROGRESS_CLIPS == 0)
                or now - last_progress_time >= FAILURE_BIN_PROGRESS_SECONDS
            )
            if should_log_progress:
                elapsed = now - init_start_time
                print(
                    "initialized motion failure bins: "
                    f"{clip_index}/{total_clips} clip(s), bins={sum(num_bins_per_motion)}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_progress_time = now

        self.num_bins_per_motion = torch.as_tensor(num_bins_per_motion, dtype=torch.long, device=self.device)
        self.num_bins = int(self.num_bins_per_motion.sum().item())
        padded_start_time = perf_counter()
        print(
            "packing motion failure bin tensors: "
            f"motions={self.motion_lib.num_motions}, bins={self.num_bins}, "
            "max_bins_per_motion="
            f"{int(self.num_bins_per_motion.max().item()) if self.motion_lib.num_motions > 0 else 0}",
            flush=True,
        )
        self._build_padded_bin_tensors()
        print(
            "packed motion failure bin tensors: "
            f"elapsed={perf_counter() - padded_start_time:.1f}s, "
            f"total_elapsed={perf_counter() - init_start_time:.1f}s",
            flush=True,
        )
        self.supports_failure_weighted_sampling = True

        if self.segment_source == SegmentSource.Anchor:
            motion_reset_eligible_values = [
                bool(torch.any(reset_eligible).item())
                for reset_eligible in self.bin_reset_eligible
            ]
            self.motion_reset_eligible = torch.as_tensor(
                motion_reset_eligible_values,
                dtype=torch.bool,
                device=self.device,
            )
            clips_without_eligible_bins = [
                clip.name
                for clip, motion_is_eligible in zip(self.motion_lib.clips, motion_reset_eligible_values, strict=False)
                if not motion_is_eligible
            ]
            if len(clips_without_eligible_bins) == self.motion_lib.num_motions:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one motion clip with an eligible reset anchor. "
                    "All loaded clips have no reset anchors."
                )
            if clips_without_eligible_bins:
                preview = ", ".join(clips_without_eligible_bins[:ANCHOR_SKIPPED_CLIP_LOG_LIMIT])
                if len(clips_without_eligible_bins) > ANCHOR_SKIPPED_CLIP_LOG_LIMIT:
                    preview = f"{preview}, ..."
                print(
                    "anchor reset sampling will skip clips without eligible reset anchors: "
                    f"count={len(clips_without_eligible_bins)}/{self.motion_lib.num_motions}, "
                    f"clips={preview}",
                    flush=True,
                )
        else:
            self.motion_reset_eligible = None

    def reset_failure_stats(self) -> None:
        self._check_failure_bins()
        self.motion_sample_counts.zero_()
        self.motion_assigned_counts.zero_()
        self.motion_fail_counts.zero_()
        for fail_counts, sample_counts in zip(self.bin_fail_counts, self.bin_sample_counts, strict=False):
            fail_counts.zero_()
            sample_counts.zero_()

    def record_failures(
        self,
        env_ids: IndexLike | None = None,
        times: torch.Tensor | None = None,
    ) -> None:
        if not self._has_failure_bins():
            return
        if self._in_motion_sampling_warmup():
            return

        env_ids = self._normalize_env_ids(env_ids)
        if env_ids.numel() == 0:
            return

        if times is None:
            times = self.current_times[env_ids]
        else:
            times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
            if times.shape[0] != env_ids.shape[0]:
                raise ValueError("times must have the same batch size as env_ids.")

        initialized_mask = self.episode_start_bin_indices[env_ids] >= 0
        if not bool(torch.any(initialized_mask).item()):
            return

        env_ids = env_ids[initialized_mask]
        times = times[initialized_mask]
        motion_ids = self.current_motion_ids[env_ids]
        # For anchor bins, this attributes the failure to the nearest previous anchor.
        bin_indices = self._times_to_bins(motion_ids, times)

        self.motion_fail_counts.index_add_(
            0,
            motion_ids,
            torch.ones(motion_ids.shape, dtype=self.motion_fail_counts.dtype, device=self.device),
        )
        batch_fail_counts = self._build_padded_bin_count_updates(motion_ids, bin_indices)
        if self._bin_fail_counts_padded is None or self._bin_valid_mask_padded is None:
            raise RuntimeError("Padded failure count tensors are not initialized.")
        self._bin_fail_counts_padded.mul_(1.0 - self.adaptive_alpha).add_(
            batch_fail_counts,
            alpha=self.adaptive_alpha,
        )
        self._bin_fail_counts_padded.mul_(self._bin_valid_mask_padded.to(dtype=torch.float32))

    def _has_failure_bins(self) -> bool:
        return (
            self.num_bins_per_motion is not None
            and self.bin_start_times is not None
            and self.bin_end_times is not None
            and self.bin_types is not None
            and self.bin_reset_times is not None
            and self.bin_reset_eligible is not None
            and self.bin_fail_counts is not None
            and self.bin_sample_counts is not None
            and self._has_padded_failure_bins()
        )

    def _check_failure_bins(self) -> None:
        if not self._has_failure_bins():
            raise RuntimeError("Failure bins not initialized. Call init_failure_bins(...) first.")

    def _anchor_sampling_error(self, message: str) -> RuntimeError:
        return RuntimeError(
            f"{message} Provide an enabled reset_anchors.json sidecar next to final_motion.npz."
        )

    def _check_anchor_motion_ids_have_eligible_resets(self, motion_ids: torch.Tensor) -> None:
        if self.segment_source != SegmentSource.Anchor or self.motion_reset_eligible is None:
            return
        if motion_ids.numel() == 0:
            return

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        eligible = self.motion_reset_eligible[motion_ids]
        if bool(torch.all(eligible).item()):
            return

        skipped_motion_ids = torch.unique(motion_ids[~eligible], sorted=True).tolist()
        skipped_clip_names = [
            self.motion_lib.clips[int(motion_id)].name
            for motion_id in skipped_motion_ids[:ANCHOR_SKIPPED_CLIP_LOG_LIMIT]
        ]
        preview = ", ".join(skipped_clip_names)
        if len(skipped_motion_ids) > ANCHOR_SKIPPED_CLIP_LOG_LIMIT:
            preview = f"{preview}, ..."
        raise self._anchor_sampling_error(
            "Anchor segment sampling requested motion clips without eligible reset anchors: "
            f"{preview}"
        )

    def _check_failure_weighted_support(self) -> None:
        if not self.supports_failure_weighted_sampling:
            if self.segment_source == SegmentSource.Anchor:
                raise self._anchor_sampling_error(
                    "Failure-weighted anchor sampling requires motion clips with anchor metadata."
                )
            raise RuntimeError(
                "Failure-weighted sampling requires initialized time bins. "
                "Set sampler bin_size."
            )

    def _times_to_bins(self, motion_ids: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.as_tensor(times, dtype=torch.float32, device=self.device).reshape(-1)
        if (
            self._has_padded_failure_bins()
            and self._bin_end_times_padded is not None
            and self.num_bins_per_motion is not None
            and self.max_bins_per_motion > 0
        ):
            motion_times = torch.clamp(times, min=0.0).unsqueeze(1)
            end_times = self._bin_end_times_padded[motion_ids]
            bin_indices = torch.sum(motion_times >= end_times, dim=1).to(dtype=torch.long)
            return torch.minimum(bin_indices, self.num_bins_per_motion[motion_ids] - 1)

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
        if (
            self._has_padded_failure_bins()
            and self.num_bins_per_motion is not None
            and self.max_bins_per_motion > 0
            and (counts_by_motion is self.bin_fail_counts or counts_by_motion is self.bin_sample_counts)
        ):
            target_counts = (
                self._bin_fail_counts_padded
                if counts_by_motion is self.bin_fail_counts
                else self._bin_sample_counts_padded
            )
            if target_counts is None:
                raise RuntimeError("Padded count tensor is not initialized.")
            motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
            bin_indices = torch.as_tensor(bin_indices, dtype=torch.long, device=self.device).reshape(-1)
            flat_indices = motion_ids * int(self.max_bins_per_motion) + bin_indices
            increments = torch.bincount(
                flat_indices,
                minlength=self.motion_lib.num_motions * int(self.max_bins_per_motion),
            ).to(device=self.device, dtype=torch.float32)
            target_counts.add_(increments.view(self.motion_lib.num_motions, int(self.max_bins_per_motion)))
            return

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            motion_mask = motion_ids == motion_id
            counts_by_motion[motion_id] += torch.bincount(
                bin_indices[motion_mask],
                minlength=int(self.num_bins_per_motion[motion_id].item()),
            ).to(device=self.device, dtype=torch.float32)

    def _build_padded_bin_count_updates(
        self,
        motion_ids: torch.Tensor,
        bin_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self._has_padded_failure_bins() or self.max_bins_per_motion <= 0:
            raise RuntimeError("Padded failure bin tensors are not initialized.")

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        bin_indices = torch.as_tensor(bin_indices, dtype=torch.long, device=self.device).reshape(-1)
        if motion_ids.shape != bin_indices.shape:
            raise ValueError("motion_ids and bin_indices must have the same shape.")

        flat_indices = motion_ids * int(self.max_bins_per_motion) + bin_indices
        counts = torch.bincount(
            flat_indices,
            minlength=self.motion_lib.num_motions * int(self.max_bins_per_motion),
        ).to(device=self.device, dtype=torch.float32)
        return counts.view(self.motion_lib.num_motions, int(self.max_bins_per_motion))

    def _record_motion_assignments(self, motion_ids: torch.Tensor) -> None:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if motion_ids.numel() == 0:
            return

        self.motion_assigned_counts.index_add_(
            0,
            motion_ids,
            torch.ones(motion_ids.shape, dtype=self.motion_assigned_counts.dtype, device=self.device),
        )

    def _record_sample_bins(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        *,
        target_bin_indices: torch.Tensor | None = None,
    ) -> None:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if motion_ids.numel() == 0:
            return

        self.motion_sample_counts.index_add_(
            0,
            motion_ids,
            torch.ones(motion_ids.shape, dtype=self.motion_sample_counts.dtype, device=self.device),
        )

        if not self._has_failure_bins():
            return

        if target_bin_indices is None:
            target_bin_indices = self._times_to_bins(motion_ids, times)
        else:
            target_bin_indices = torch.as_tensor(target_bin_indices, dtype=torch.long, device=self.device).reshape(-1)
            if target_bin_indices.shape != motion_ids.shape:
                raise ValueError("target_bin_indices must have the same shape as motion_ids.")

        self._accumulate_bin_counts(self.bin_sample_counts, motion_ids, target_bin_indices)


Sampler = MotionSampler

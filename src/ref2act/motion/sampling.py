from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch

from ref2act.common.utils import IndexLike

from .library import MotionClip, MotionLib
from .segments import ANCHOR_FRAME_LABEL_RED, build_legacy_time_segments


DEFAULT_ANCHOR_FAILURE_BIN_SIZE = 0.3


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
        failure_decay: float = 1.0,
        failure_weight_uniform_mix: float = 0.1,
        failure_weight_max_uniform_ratio: float | None = 2.5,
        failure_weight_exploration_bonus: float = 0.5,
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
        self.episode_start_sampling_strategy_values = torch.full(
            (num_envs,),
            SamplingStrategy.Random.value,
            dtype=torch.long,
            device=self.device,
        )

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
                or (self.segment_source == SegmentSource.Time and self.motion_lib.all_clips_have_segments)
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
        return torch.randint(self.motion_lib.num_motions, (env_ids.numel(),), device=self.device)

    def _build_guarded_sampling_probabilities(
        self,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

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

        uniform_probs = eligible.to(dtype=torch.float32)
        uniform_sum = torch.sum(uniform_probs)
        if float(uniform_sum.item()) <= 0.0:
            raise ValueError("eligible_mask must include at least one entry.")
        uniform_probs = uniform_probs / uniform_sum

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
        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs_sum = torch.sum(probs)
        if float(probs_sum.item()) > 0.0:
            probs = probs / probs_sum
        else:
            probs = uniform_probs

        probs = self._apply_max_uniform_probability_cap(
            probs,
            eligible_mask=eligible,
            uniform_probs=uniform_probs,
        )
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

    def _apply_max_uniform_probability_cap_batched(
        self,
        probs: torch.Tensor,
        *,
        eligible_mask: torch.Tensor,
        uniform_probs: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
        eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device)
        uniform_probs = torch.as_tensor(uniform_probs, dtype=torch.float32, device=self.device)
        if probs.shape != eligible.shape or probs.shape != uniform_probs.shape:
            raise ValueError("Probability cap inputs must have the same shape.")
        if probs.ndim != 2:
            raise ValueError("Batched probability cap inputs must be 2-D.")

        eps = torch.finfo(probs.dtype).eps
        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs = probs / torch.clamp(torch.sum(probs, dim=1, keepdim=True), min=eps)

        if self.failure_weight_max_uniform_ratio is None:
            return probs

        eligible_count = torch.sum(eligible.to(dtype=torch.float32), dim=1, keepdim=True)
        if bool(torch.any(eligible_count <= 0).item()):
            raise ValueError("eligible_mask must include at least one entry in every row.")

        max_prob = float(self.failure_weight_max_uniform_ratio) / eligible_count
        cap_needed = (max_prob < 1.0) & torch.any(probs > max_prob + eps, dim=1, keepdim=True)
        finite_rows = torch.all(torch.isfinite(probs), dim=1, keepdim=True)
        positive_rows = torch.sum(probs, dim=1, keepdim=True) > eps
        cap_needed = cap_needed & finite_rows & positive_rows
        if not bool(torch.any(cap_needed).item()):
            return torch.where(finite_rows & positive_rows, probs, uniform_probs)

        inf = torch.full_like(probs, float("inf"))
        lower = torch.min(torch.where(eligible, probs - max_prob, inf), dim=1, keepdim=True).values
        upper = torch.max(torch.where(eligible, probs, torch.zeros_like(probs)), dim=1, keepdim=True).values
        for _ in range(64):
            threshold = 0.5 * (lower + upper)
            projected_probs = torch.clamp(probs - threshold, min=0.0)
            projected_probs = torch.minimum(projected_probs, max_prob)
            projected_probs = torch.where(eligible, projected_probs, torch.zeros_like(projected_probs))
            lower = torch.where(torch.sum(projected_probs, dim=1, keepdim=True) > 1.0, threshold, lower)
            upper = torch.where(torch.sum(projected_probs, dim=1, keepdim=True) > 1.0, upper, threshold)

        capped_probs = torch.clamp(probs - upper, min=0.0)
        capped_probs = torch.minimum(capped_probs, max_prob)
        capped_probs = torch.where(eligible, capped_probs, torch.zeros_like(capped_probs))
        capped_sum = torch.sum(capped_probs, dim=1, keepdim=True)
        capped_probs = torch.where(capped_sum > eps, capped_probs / torch.clamp(capped_sum, min=eps), uniform_probs)
        return torch.where(cap_needed, capped_probs, probs)

    def _build_guarded_sampling_probabilities_batched(
        self,
        fail_counts: torch.Tensor,
        sample_counts: torch.Tensor,
        *,
        temperature: float,
        eligible_mask: torch.Tensor,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        fail_counts = torch.as_tensor(fail_counts, dtype=torch.float32, device=self.device)
        sample_counts = torch.as_tensor(sample_counts, dtype=torch.float32, device=self.device)
        eligible = torch.as_tensor(eligible_mask, dtype=torch.bool, device=self.device)
        if fail_counts.shape != sample_counts.shape or fail_counts.shape != eligible.shape:
            raise ValueError("Batched probability inputs must have the same shape.")
        if fail_counts.ndim != 2:
            raise ValueError("Batched probability inputs must be 2-D.")

        eps = torch.finfo(fail_counts.dtype).eps
        eligible_float = eligible.to(dtype=torch.float32)
        eligible_count = torch.sum(eligible_float, dim=1, keepdim=True)
        if bool(torch.any(eligible_count <= 0.0).item()):
            raise ValueError("eligible_mask must include at least one entry in every row.")
        uniform_probs = eligible_float / torch.clamp(eligible_count, min=eps)

        eligible_sample_counts = torch.where(
            eligible,
            torch.clamp(sample_counts, min=0.0),
            torch.zeros_like(sample_counts),
        )
        total_eligible_samples = torch.sum(eligible_sample_counts, dim=1, keepdim=True)
        exploration_scale = torch.log(total_eligible_samples + eligible_count + 1.0)
        exploration = torch.sqrt(exploration_scale / (eligible_sample_counts + 1.0))

        fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
        score = fail_rate + self.failure_weight_exploration_bonus * exploration
        learned_weights = score.pow(1.0 / temperature)
        learned_weights = torch.where(eligible, learned_weights, torch.zeros_like(learned_weights))

        learned_sum = torch.sum(learned_weights, dim=1, keepdim=True)
        learned_valid = torch.all(torch.isfinite(learned_weights), dim=1, keepdim=True) & (learned_sum > 0.0)
        learned_probs = torch.where(
            learned_valid,
            learned_weights / torch.clamp(learned_sum, min=eps),
            uniform_probs,
        )

        probs = (
            (1.0 - self.failure_weight_uniform_mix) * learned_probs
            + self.failure_weight_uniform_mix * uniform_probs
        )
        probs = torch.where(eligible, probs, torch.zeros_like(probs))
        probs_sum = torch.sum(probs, dim=1, keepdim=True)
        probs = torch.where(
            probs_sum > 0.0,
            probs / torch.clamp(probs_sum, min=eps),
            uniform_probs,
        )

        return self._apply_max_uniform_probability_cap_batched(
            probs,
            eligible_mask=eligible,
            uniform_probs=uniform_probs,
        )

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

        if self._bin_fail_counts_padded is not None and self._bin_sample_counts_padded is not None:
            motion_fail_counts = torch.sum(self._bin_fail_counts_padded, dim=1)
            motion_sample_counts = torch.sum(self._bin_sample_counts_padded, dim=1)
        else:
            motion_fail_counts = torch.stack([fail_counts.sum() for fail_counts in self.bin_fail_counts], dim=0)
            motion_sample_counts = torch.stack([sample_counts.sum() for sample_counts in self.bin_sample_counts], dim=0)
        motion_probs = self._build_guarded_sampling_probabilities(
            motion_fail_counts,
            motion_sample_counts,
            temperature=temperature,
        )
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

        if not torch.any(self.motion_lib.motion_has_segments[motion_ids]):
            return times, target_bin_indices

        if (
            self._has_padded_failure_bins()
            and self._bin_start_times_padded is not None
            and bool(torch.all(self.motion_lib.motion_has_segments[motion_ids]).item())
        ):
            num_bins = self.num_bins_per_motion[motion_ids]
            segment_indices = torch.floor(torch.rand(motion_ids.shape, device=self.device) * num_bins).long()
            segment_indices = torch.clamp(segment_indices, max=num_bins - 1)
            times = self._bin_start_times_padded[motion_ids, segment_indices]
            target_bin_indices = segment_indices
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
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
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
            probs = self._build_guarded_sampling_probabilities_batched(
                fail_counts,
                sample_counts,
                temperature=temperature,
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
            probs = self._build_guarded_sampling_probabilities(
                fail_counts,
                sample_counts,
                temperature=temperature,
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
        self.episode_start_sampling_strategy_values[resolved_env_ids] = strategy.value
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

        segment_end_times = clip.anchor_segment_end_times.to(device=self.device)
        segment_labels = clip.anchor_segment_labels.to(device=self.device)
        segment_indices = torch.searchsorted(segment_end_times, anchor_times, right=True)
        segment_indices = torch.clamp(segment_indices, max=max(int(segment_labels.numel()) - 1, 0))
        anchor_labels = segment_labels[segment_indices]

        bin_size = self.bin_size if self.bin_size is not None else DEFAULT_ANCHOR_FAILURE_BIN_SIZE
        start_time_values: list[float] = []
        end_time_values: list[float] = []
        label_values: list[int] = []
        reset_time_values: list[float] = []
        eligible_values: list[bool] = []
        clip_duration = float(clip.duration)

        for anchor_index in range(int(anchor_times.numel())):
            span_start = 0.0 if anchor_index == 0 else float(anchor_times[anchor_index].item())
            if anchor_index + 1 < int(anchor_times.numel()):
                span_end = float(anchor_times[anchor_index + 1].item())
            else:
                span_end = clip_duration
            span_start = max(0.0, min(span_start, clip_duration))
            span_end = max(0.0, min(span_end, clip_duration))
            if span_end <= span_start:
                continue

            num_sub_bins = max(int(math.ceil((span_end - span_start) / bin_size - 1.0e-6)), 1)
            anchor_label = int(anchor_labels[anchor_index].item())
            reset_time = float(anchor_times[anchor_index].item())
            reset_eligible = anchor_label != int(ANCHOR_FRAME_LABEL_RED)
            for sub_bin_index in range(num_sub_bins):
                sub_start = span_start + float(sub_bin_index) * bin_size
                sub_end = min(sub_start + bin_size, span_end)
                if sub_end <= sub_start:
                    continue
                start_time_values.append(sub_start)
                end_time_values.append(sub_end)
                label_values.append(anchor_label)
                reset_time_values.append(reset_time)
                eligible_values.append(reset_eligible)

        if not start_time_values:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return (
                empty,
                empty,
                torch.empty(0, dtype=torch.long, device=self.device),
                empty,
                torch.empty(0, dtype=torch.bool, device=self.device),
            )

        return (
            torch.as_tensor(start_time_values, dtype=torch.float32, device=self.device),
            torch.as_tensor(end_time_values, dtype=torch.float32, device=self.device),
            torch.as_tensor(label_values, dtype=torch.long, device=self.device),
            torch.as_tensor(reset_time_values, dtype=torch.float32, device=self.device),
            torch.as_tensor(eligible_values, dtype=torch.bool, device=self.device),
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
        self._build_padded_bin_tensors()
        self.supports_failure_weighted_sampling = all(self.bin_uses_segment_metadata)

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

    def record_failures(
        self,
        env_ids: IndexLike | None = None,
        times: torch.Tensor | None = None,
    ) -> None:
        if not self._has_failure_bins():
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

        failure_weighted_mask = (
            self.episode_start_sampling_strategy_values[env_ids] == SamplingStrategy.FailureWeighted.value
        )
        if not bool(torch.any(failure_weighted_mask).item()):
            return

        env_ids = env_ids[failure_weighted_mask]
        times = times[failure_weighted_mask]
        motion_ids = self.current_motion_ids[env_ids]
        # For anchor bins, this attributes the failure to the nearest previous anchor.
        bin_indices = self._times_to_bins(motion_ids, times)

        self._accumulate_bin_counts(self.bin_fail_counts, motion_ids, bin_indices)

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
            and self._has_padded_failure_bins()
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
            if self._bin_fail_counts_padded is not None and self._bin_sample_counts_padded is not None:
                self._bin_fail_counts_padded.mul_(self.failure_decay)
                self._bin_sample_counts_padded.mul_(self.failure_decay)
            else:
                for fail_counts, sample_counts in zip(self.bin_fail_counts, self.bin_sample_counts, strict=False):
                    fail_counts.mul_(self.failure_decay)
                    sample_counts.mul_(self.failure_decay)
        self._accumulate_bin_counts(self.bin_sample_counts, motion_ids, target_bin_indices)


Sampler = MotionSampler

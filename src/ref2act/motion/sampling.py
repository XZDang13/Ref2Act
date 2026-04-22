from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from ref2act.common.utils import IndexLike

from .library import MotionClip, MotionLib
from .segments import ANCHOR_FRAME_LABEL_GREEN, build_legacy_time_segments


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
        segment_source: SegmentSource = SegmentSource.Time,
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

        self.current_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_start_times = torch.zeros(num_envs, device=self.device)

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
        if uniform_sum <= 0:
            raise ValueError("eligible_mask must include at least one entry.")
        uniform_probs = uniform_probs / uniform_sum

        fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
        learned_weights = fail_rate.pow(1.0 / temperature)
        learned_weights = torch.where(eligible, learned_weights, torch.zeros_like(learned_weights))

        learned_sum = torch.sum(learned_weights)
        if bool(torch.all(torch.isfinite(learned_weights)).item()) and float(learned_sum.item()) > 0.0:
            learned_probs = learned_weights / learned_sum
        else:
            learned_probs = uniform_probs

        probs = (1.0 - self.failure_weight_uniform_mix) * learned_probs + self.failure_weight_uniform_mix * uniform_probs
        return probs / torch.clamp(torch.sum(probs), min=torch.finfo(probs.dtype).eps)

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
            eligible_bin_indices = torch.nonzero(self.bin_reset_eligible[motion_id], as_tuple=False).squeeze(-1)
            if eligible_bin_indices.numel() == 0:
                raise self._anchor_sampling_error(
                    "Anchor segment sampling requires at least one eligible reset anchor for every motion clip."
                )
            sampled_eligible_indices = torch.randint(
                eligible_bin_indices.numel(),
                (num_samples,),
                device=self.device,
            )
            sampled_bin_indices = eligible_bin_indices[sampled_eligible_indices]
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

        start_times = clip.anchor_segment_start_times.to(device=self.device)
        end_times = clip.anchor_segment_end_times.to(device=self.device)
        segment_labels = clip.anchor_segment_labels.to(device=self.device)
        anchor_times = clip.anchor_times.to(device=self.device)

        reset_times = torch.zeros(start_times.shape, dtype=torch.float32, device=self.device)
        eligible = torch.zeros(start_times.shape, dtype=torch.bool, device=self.device)
        if anchor_times.numel() == 0:
            return start_times, end_times, segment_labels, reset_times, eligible

        first_anchor_in_segment = torch.searchsorted(anchor_times, start_times, right=False)
        first_anchor_after_segment = torch.searchsorted(anchor_times, end_times, right=False)
        latest_anchor_before_segment = first_anchor_in_segment - 1

        for segment_index in range(int(start_times.shape[0])):
            selected_anchor_index: int | None = None
            if (
                int(segment_labels[segment_index].item()) == int(ANCHOR_FRAME_LABEL_GREEN)
                and first_anchor_in_segment[segment_index] < first_anchor_after_segment[segment_index]
            ):
                selected_anchor_index = int(first_anchor_in_segment[segment_index].item())
            elif latest_anchor_before_segment[segment_index] >= 0:
                selected_anchor_index = int(latest_anchor_before_segment[segment_index].item())

            if selected_anchor_index is None:
                continue

            reset_times[segment_index] = anchor_times[selected_anchor_index]
            eligible[segment_index] = True

        return start_times, end_times, segment_labels, reset_times, eligible

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
        if times is None:
            times = self.current_times[env_ids]
        else:
            times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
            if times.shape[0] != env_ids.shape[0]:
                raise ValueError("times must have the same batch size as env_ids.")
        motion_ids = self.current_motion_ids[env_ids]

        self._accumulate_bin_counts(self.bin_fail_counts, motion_ids, self._times_to_bins(motion_ids, times))

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


Sampler = MotionSampler

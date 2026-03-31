from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from ref2act.common.utils import IndexLike

from .library import MotionLib
from .segments import build_legacy_time_segments


class SamplerMod(Enum):
    Cycle = 0
    Clamp = 1


class SamplingStrategy(Enum):
    Start = 0
    Random = 1
    FailureWeighted = 2


@dataclass(frozen=True)
class ResetSample:
    env_ids: torch.Tensor
    motion_ids: torch.Tensor
    times: torch.Tensor


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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.motion_lib = motion_lib
        self.anchor_body_index = anchor_body_index
        if not (0.0 < failure_decay <= 1.0):
            raise ValueError("failure_decay must be in (0, 1].")
        self.failure_decay = failure_decay

        self.current_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_times = torch.zeros(num_envs, device=self.device)
        self.episode_start_motion_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_start_times = torch.zeros(num_envs, device=self.device)

        self.bin_size: float | None = None
        self.num_bins = 0
        self.num_bins_per_motion: torch.Tensor | None = None
        self.bin_start_times: list[torch.Tensor] | None = None
        self.bin_end_times: list[torch.Tensor] | None = None
        self.bin_types: list[torch.Tensor] | None = None
        self.bin_uses_segment_metadata: list[bool] | None = None
        self.bin_fail_counts: list[torch.Tensor] | None = None
        self.bin_sample_counts: list[torch.Tensor] | None = None
        self.supports_failure_weighted_sampling = False

        if bin_size is not None:
            self.init_failure_bins(bin_size)

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

    def _sample_start_times_for_motion_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(motion_ids.shape, dtype=torch.float32, device=self.device)

    def _sample_rand_times_for_motion_ids(self, motion_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        durations = self.motion_lib.get_duration(motion_ids)
        times = torch.rand_like(durations) * durations

        if not torch.any(self.motion_lib.motion_has_segments[motion_ids]):
            return times

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

        return times

    def _sample_failure_weighted_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self._check_failure_bins()
        self._check_failure_weighted_support()

        if min_weight < 0.0:
            raise ValueError("min_weight must be >= 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        times = torch.empty(motion_ids.shape, dtype=torch.float32, device=self.device)

        for motion_id in torch.unique(motion_ids, sorted=True).tolist():
            mask = motion_ids == motion_id
            num_samples = int(mask.sum().item())
            fail_counts = self.bin_fail_counts[motion_id]
            sample_counts = self.bin_sample_counts[motion_id]
            fail_rate = fail_counts / torch.clamp(sample_counts, min=1.0)
            weights = (fail_rate + min_weight).pow(1.0 / temperature)

            if torch.sum(weights) <= 0:
                weights = torch.ones_like(weights)

            bin_indices = torch.multinomial(weights, num_samples, replacement=True)
            bin_starts = self.bin_start_times[motion_id][bin_indices]
            bin_ends = self.bin_end_times[motion_id][bin_indices]
            times[mask] = bin_starts + torch.rand(num_samples, device=self.device) * (bin_ends - bin_starts)

        return times

    def sample_times_for_motion_ids(
        self,
        motion_ids: torch.Tensor,
        strategy: SamplingStrategy,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if strategy == SamplingStrategy.Start:
            return self._sample_start_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.Random:
            return self._sample_rand_times_for_motion_ids(motion_ids)
        if strategy == SamplingStrategy.FailureWeighted:
            return self._sample_failure_weighted_times_for_motion_ids(
                motion_ids,
                min_weight=min_weight,
                temperature=temperature,
            )
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    def reset(
        self,
        env_ids: IndexLike | None = None,
        *,
        strategy: SamplingStrategy = SamplingStrategy.Random,
        min_weight: float = 0.001,
        temperature: float = 1.0,
    ) -> ResetSample:
        resolved_env_ids = self._normalize_env_ids(env_ids)
        motion_ids = self.sample_motion_ids(resolved_env_ids)
        times = self.sample_times_for_motion_ids(
            motion_ids,
            strategy,
            min_weight=min_weight,
            temperature=temperature,
        )
        self.current_motion_ids[resolved_env_ids] = motion_ids
        self.current_times[resolved_env_ids] = times
        self.episode_start_motion_ids[resolved_env_ids] = motion_ids
        self.episode_start_times[resolved_env_ids] = times
        self._record_sample_bins(motion_ids, times)
        return ResetSample(env_ids=resolved_env_ids, motion_ids=motion_ids, times=times)

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

    def init_failure_bins(self, bin_size: float) -> None:
        if bin_size <= 0.0:
            raise ValueError("bin_size must be > 0")

        self.bin_size = float(bin_size)
        self.bin_start_times = []
        self.bin_end_times = []
        self.bin_types = []
        self.bin_uses_segment_metadata = []
        num_bins_per_motion: list[int] = []

        for clip in self.motion_lib.clips:
            if clip.has_segments:
                start_times = clip.segment_start_times.to(device=self.device)
                end_times = clip.segment_end_times.to(device=self.device)
                segment_types = clip.segment_types.to(device=self.device)
                self.bin_uses_segment_metadata.append(True)
            else:
                start_times_np, end_times_np, segment_types_np = build_legacy_time_segments(
                    duration=clip.duration,
                    bin_size=self.bin_size,
                )
                start_times = torch.as_tensor(start_times_np, dtype=torch.float32, device=self.device)
                end_times = torch.as_tensor(end_times_np, dtype=torch.float32, device=self.device)
                segment_types = torch.as_tensor(segment_types_np, dtype=torch.long, device=self.device)
                self.bin_uses_segment_metadata.append(False)

            self.bin_start_times.append(start_times)
            self.bin_end_times.append(end_times)
            self.bin_types.append(segment_types)
            num_bins_per_motion.append(int(start_times.shape[0]))

        self.supports_failure_weighted_sampling = all(self.bin_uses_segment_metadata)
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
        self._check_failure_bins()
        env_ids = self._normalize_env_ids(env_ids)
        if times is None:
            times = self.current_times[env_ids]
        else:
            times = torch.as_tensor(times, dtype=torch.float32, device=self.device)
            if times.shape[0] != env_ids.shape[0]:
                raise ValueError("times must have the same batch size as env_ids.")
        motion_ids = self.current_motion_ids[env_ids]

        self._accumulate_bin_counts(self.bin_fail_counts, motion_ids, self._times_to_bins(motion_ids, times))

    def _check_failure_bins(self) -> None:
        if (
            self.bin_size is None
            or self.num_bins_per_motion is None
            or self.bin_start_times is None
            or self.bin_end_times is None
            or self.bin_types is None
            or self.bin_uses_segment_metadata is None
            or self.bin_fail_counts is None
            or self.bin_sample_counts is None
        ):
            raise RuntimeError("Failure bins not initialized. Call init_failure_bins(...) first.")

    def _check_failure_weighted_support(self) -> None:
        if not self.supports_failure_weighted_sampling:
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

    def _record_sample_bins(self, motion_ids: torch.Tensor, times: torch.Tensor) -> None:
        if (
            self.bin_size is None
            or self.num_bins_per_motion is None
            or self.bin_fail_counts is None
            or self.bin_sample_counts is None
        ):
            return
        if motion_ids.numel() == 0:
            return
        if self.failure_decay < 1.0 and self.bin_fail_counts is not None:
            for fail_counts, sample_counts in zip(self.bin_fail_counts, self.bin_sample_counts, strict=False):
                fail_counts.mul_(self.failure_decay)
                sample_counts.mul_(self.failure_decay)
        self._accumulate_bin_counts(self.bin_sample_counts, motion_ids, self._times_to_bins(motion_ids, times))


Sampler = MotionSampler

from __future__ import annotations

import numpy as np


SEGMENT_TYPE_TIME_BIN = 0
SEGMENT_TYPE_AIR_MERGE = 1
DEFAULT_AIRBORNE_HEIGHT_MARGIN = 0.06
DEFAULT_FOOT_HEIGHT_BASELINE_PERCENTILE = 5.0


def build_base_time_bins(duration: float, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if bin_size <= 0.0:
        raise ValueError("bin_size must be > 0")

    num_bins = max(int(np.ceil(duration / bin_size)), 1)
    start_times = np.arange(num_bins, dtype=np.float32) * float(bin_size)
    end_times = np.minimum(start_times + float(bin_size), np.float32(duration)).astype(np.float32)
    return start_times, end_times


def build_legacy_time_segments(duration: float, bin_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_times, end_times = build_base_time_bins(duration=duration, bin_size=bin_size)
    segment_types = np.full(start_times.shape, SEGMENT_TYPE_TIME_BIN, dtype=np.int64)
    return start_times, end_times, segment_types


def build_contact_segments(
    has_ground_contact: np.ndarray,
    dt: float,
    duration: float,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    ground_contact = np.asarray(has_ground_contact, dtype=bool).reshape(-1)
    if ground_contact.size == 0:
        raise ValueError("has_ground_contact must contain at least one frame.")

    base_starts, base_ends = build_base_time_bins(duration=duration, bin_size=bin_size)
    airborne_runs = _find_true_runs(~ground_contact)
    merged_air_runs = _merge_overlapping_runs(
        _expand_airborne_runs_to_bin_context(airborne_runs, base_starts, base_ends, dt, duration)
    )

    segment_starts: list[np.float32] = []
    segment_ends: list[np.float32] = []
    segment_types: list[int] = []

    current_bin = 0
    air_run_index = 0
    while current_bin < len(base_starts):
        if air_run_index < len(merged_air_runs) and current_bin == merged_air_runs[air_run_index][0]:
            air_start_bin, air_end_bin = merged_air_runs[air_run_index]
            segment_starts.append(base_starts[air_start_bin])
            segment_ends.append(base_ends[air_end_bin])
            segment_types.append(SEGMENT_TYPE_AIR_MERGE)
            current_bin = air_end_bin + 1
            air_run_index += 1
            continue

        segment_starts.append(base_starts[current_bin])
        segment_ends.append(base_ends[current_bin])
        segment_types.append(SEGMENT_TYPE_TIME_BIN)
        current_bin += 1

    start_times = np.asarray(segment_starts, dtype=np.float32)
    end_times = np.asarray(segment_ends, dtype=np.float32)
    types = np.asarray(segment_types, dtype=np.int64)
    validate_segment_arrays(start_times, end_times, duration=duration, segment_types=types)
    return start_times, end_times, types


def infer_ground_contact_from_foot_heights(
    foot_heights: np.ndarray,
    airborne_height_margin: float = DEFAULT_AIRBORNE_HEIGHT_MARGIN,
    foot_height_baseline_percentile: float = DEFAULT_FOOT_HEIGHT_BASELINE_PERCENTILE,
) -> np.ndarray:
    heights = np.asarray(foot_heights, dtype=np.float32)
    if heights.ndim != 2 or heights.shape[1] == 0:
        raise ValueError("foot_heights must be shaped [num_frames, num_feet].")
    if airborne_height_margin <= 0.0:
        raise ValueError("airborne_height_margin must be > 0.")
    if not 0.0 <= foot_height_baseline_percentile <= 100.0:
        raise ValueError("foot_height_baseline_percentile must be in [0, 100].")

    foot_height_baseline = np.percentile(
        heights,
        foot_height_baseline_percentile,
        axis=0,
        keepdims=True,
    )
    airborne_frames = np.all((heights - foot_height_baseline) > airborne_height_margin, axis=1)
    return ~airborne_frames


def validate_segment_arrays(
    start_times: np.ndarray,
    end_times: np.ndarray,
    duration: float,
    segment_types: np.ndarray | None = None,
    *,
    atol: float = 1.0e-5,
) -> None:
    starts = np.asarray(start_times, dtype=np.float64).reshape(-1)
    ends = np.asarray(end_times, dtype=np.float64).reshape(-1)

    if starts.size == 0:
        raise ValueError("Segment metadata must contain at least one segment.")
    if starts.shape != ends.shape:
        raise ValueError("segment_start_times and segment_end_times must have the same shape.")
    if segment_types is not None:
        types = np.asarray(segment_types, dtype=np.int64).reshape(-1)
        if types.shape != starts.shape:
            raise ValueError("segment_types must have the same shape as segment_start_times.")
        if np.any((types != SEGMENT_TYPE_TIME_BIN) & (types != SEGMENT_TYPE_AIR_MERGE)):
            raise ValueError("segment_types contains unknown segment ids.")

    if np.any(ends <= starts):
        raise ValueError("Every segment must have positive duration.")
    if np.any(starts[1:] + atol < starts[:-1]):
        raise ValueError("segment_start_times must be non-decreasing.")
    if not np.isclose(starts[0], 0.0, atol=atol):
        raise ValueError("Segment coverage must start at t=0.")
    if not np.isclose(ends[-1], duration, atol=max(atol, abs(duration) * 1.0e-6)):
        raise ValueError("Segment coverage must end at the clip duration.")
    if not np.allclose(starts[1:], ends[:-1], atol=atol):
        raise ValueError("Segments must form a contiguous partition of the clip duration.")


def _find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, is_true in enumerate(values):
        if is_true and run_start is None:
            run_start = index
        elif not is_true and run_start is not None:
            runs.append((run_start, index - 1))
            run_start = None

    if run_start is not None:
        runs.append((run_start, len(values) - 1))
    return runs


def _expand_airborne_runs_to_bin_context(
    airborne_runs: list[tuple[int, int]],
    base_starts: np.ndarray,
    base_ends: np.ndarray,
    dt: float,
    duration: float,
) -> list[tuple[int, int]]:
    expanded_runs: list[tuple[int, int]] = []
    if len(base_starts) == 0:
        return expanded_runs

    for frame_start, frame_end in airborne_runs:
        air_start_time = frame_start * dt
        air_end_time = min((frame_end + 1) * dt, duration)
        overlapping_bins = np.nonzero((base_starts < air_end_time) & (base_ends > air_start_time))[0]
        if overlapping_bins.size == 0:
            continue

        start_bin = max(int(overlapping_bins[0]) - 1, 0)
        end_bin = min(int(overlapping_bins[-1]) + 1, len(base_starts) - 1)
        expanded_runs.append((start_bin, end_bin))

    return expanded_runs


def _merge_overlapping_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not runs:
        return []

    ordered_runs = sorted(runs)
    merged_runs: list[tuple[int, int]] = [ordered_runs[0]]

    for start, end in ordered_runs[1:]:
        previous_start, previous_end = merged_runs[-1]
        if start <= previous_end + 1:
            merged_runs[-1] = (previous_start, max(previous_end, end))
        else:
            merged_runs.append((start, end))

    return merged_runs

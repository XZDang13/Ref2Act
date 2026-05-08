from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEGMENT_TYPE_TIME_BIN = 0
SEGMENT_TYPE_AIR_MERGE = 1
DEFAULT_AIRBORNE_HEIGHT_MARGIN = 0.06
DEFAULT_FOOT_HEIGHT_BASELINE_PERCENTILE = 5.0
ANCHOR_SUPPORT_MODE_NONE = np.int8(0)
ANCHOR_SUPPORT_MODE_LEFT = np.int8(1)
ANCHOR_SUPPORT_MODE_RIGHT = np.int8(2)
ANCHOR_SUPPORT_MODE_DOUBLE = np.int8(3)
ANCHOR_REQUIRED_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)
DEFAULT_ANCHOR_CONTACT_HEIGHT = 0.04
DEFAULT_ANCHOR_CONTACT_XY_SPEED = 0.20
DEFAULT_ANCHOR_CONTACT_Z_SPEED = 0.15
DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG = 35.0
DEFAULT_ANCHOR_MIN_SPACING = 0.35


@dataclass(frozen=True)
class AnchorSelectionMetadata:
    frame_indices: np.ndarray
    times: np.ndarray
    joint_kinetic_energy: np.ndarray

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "anchor_selection_version": np.asarray(3, dtype=np.int64),
            "anchor_frame_indices": self.frame_indices.astype(np.int64, copy=False),
            "anchor_times": self.times.astype(np.float32, copy=False),
            "anchor_joint_kinetic_energy": self.joint_kinetic_energy.astype(np.float32, copy=False),
        }


@dataclass(frozen=True)
class AnchorSelectionDiagnostics:
    metadata: AnchorSelectionMetadata
    frame_times: np.ndarray
    foot_height_above_ground: np.ndarray
    foot_height_baseline: np.ndarray
    ground_contact: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    support_modes: np.ndarray
    safe_mask: np.ndarray
    torso_tilt_deg: np.ndarray
    joint_kinetic_energy: np.ndarray
    no_support: np.ndarray
    used_lowest_energy_fallback: bool


def build_base_time_bins(duration: float, bin_size: float) -> tuple[np.ndarray, np.ndarray]:
    if duration <= 0.0:
        raise ValueError("duration must be > 0")
    if bin_size <= 0.0:
        raise ValueError("bin_size must be > 0")

    resolved_duration = float(duration)
    resolved_bin_size = float(bin_size)
    ratio = resolved_duration / resolved_bin_size
    # Floating point division can turn an exact multiple such as 2.1 / 0.3 into
    # 7.000000000000001, which would create a final zero-duration segment.
    num_bins = max(int(np.ceil(ratio - 1.0e-9)), 1)
    boundaries_64 = np.arange(num_bins + 1, dtype=np.float64) * resolved_bin_size
    boundaries_64 = np.minimum(boundaries_64, resolved_duration)
    boundaries_64[-1] = resolved_duration
    boundaries = boundaries_64.astype(np.float32)

    # A real but sub-float32 tail, for example 10.0000002s after a 10.0s bin,
    # would otherwise become a zero-duration final segment after export.
    positive_boundary_mask = np.concatenate(([True], boundaries[1:] > boundaries[:-1]))
    boundaries = boundaries[positive_boundary_mask]
    if boundaries.shape[0] < 2:
        raise ValueError("duration is too small to represent a positive segment.")

    start_times = boundaries[:-1]
    end_times = boundaries[1:]
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


def build_anchor_selection_metadata(
    log: dict[str, object],
    *,
    airborne_height_margin: float = DEFAULT_AIRBORNE_HEIGHT_MARGIN,
) -> AnchorSelectionMetadata:
    return build_anchor_selection_diagnostics(
        log,
        airborne_height_margin=airborne_height_margin,
    ).metadata


def build_anchor_selection_diagnostics(
    log: dict[str, object],
    *,
    airborne_height_margin: float = DEFAULT_AIRBORNE_HEIGHT_MARGIN,
) -> AnchorSelectionDiagnostics:
    fps = float(np.asarray(log["fps"]).item())
    if fps <= 0.0:
        raise ValueError("Anchor selection requires a positive fps.")

    joint_vel = np.asarray(log["joint_vel"], dtype=np.float32)
    body_pos_w = np.asarray(log["body_pos_w"], dtype=np.float32)
    body_lin_vel_w = np.asarray(log["body_lin_vel_w"], dtype=np.float32)
    num_frames = int(joint_vel.shape[0])
    if num_frames < 1:
        raise ValueError("Anchor selection requires at least one frame.")
    if body_pos_w.shape[0] != num_frames or body_lin_vel_w.shape[0] != num_frames:
        raise ValueError("Anchor selection requires body positions and velocities for every frame.")

    body_name_to_index = _resolve_anchor_body_indices(log["body_names"])
    pelvis_index = body_name_to_index["pelvis"]
    torso_index = body_name_to_index["torso_link"]
    left_foot_index = body_name_to_index["left_ankle_roll_link"]
    right_foot_index = body_name_to_index["right_ankle_roll_link"]

    dt = 1.0 / fps
    duration = dt * num_frames
    frame_times = np.arange(num_frames, dtype=np.float32) * np.float32(dt)

    foot_heights = body_pos_w[:, [left_foot_index, right_foot_index], 2]
    foot_height_baseline = np.percentile(
        foot_heights,
        DEFAULT_FOOT_HEIGHT_BASELINE_PERCENTILE,
        axis=0,
        keepdims=True,
    ).astype(np.float32)
    foot_height_above_ground = foot_heights - foot_height_baseline
    ground_contact = infer_ground_contact_from_foot_heights(
        foot_heights,
        airborne_height_margin=airborne_height_margin,
    )

    left_foot_vel = body_lin_vel_w[:, left_foot_index]
    right_foot_vel = body_lin_vel_w[:, right_foot_index]
    left_contact = _infer_single_foot_contact(foot_height_above_ground[:, 0], left_foot_vel)
    right_contact = _infer_single_foot_contact(foot_height_above_ground[:, 1], right_foot_vel)
    support_modes = _build_support_modes(left_contact, right_contact)
    joint_kinetic_energy = np.sum(np.abs(joint_vel), axis=1).astype(np.float32)

    body_up = body_pos_w[:, torso_index] - body_pos_w[:, pelvis_index]
    body_up_norm = np.linalg.norm(body_up, axis=1)
    cos_tilt = np.divide(body_up[:, 2], body_up_norm, out=np.zeros_like(body_up_norm), where=body_up_norm > 1.0e-6)
    torso_tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))).astype(np.float32)

    no_support = support_modes == ANCHOR_SUPPORT_MODE_NONE
    safe_mask = ground_contact & (~no_support) & (torso_tilt_deg <= DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG)
    anchor_frame_indices, used_lowest_energy_fallback = _select_safe_low_kinetic_anchor_frame_indices(
        safe_mask,
        joint_kinetic_energy,
        dt=dt,
        min_spacing_seconds=DEFAULT_ANCHOR_MIN_SPACING,
    )
    anchor_times = anchor_frame_indices.astype(np.float32) * np.float32(dt)

    metadata = AnchorSelectionMetadata(
        frame_indices=anchor_frame_indices.astype(np.int64, copy=False),
        times=anchor_times,
        joint_kinetic_energy=joint_kinetic_energy[anchor_frame_indices].astype(np.float32, copy=False),
    )
    return AnchorSelectionDiagnostics(
        metadata=metadata,
        frame_times=frame_times.astype(np.float32, copy=False),
        foot_height_above_ground=foot_height_above_ground.astype(np.float32, copy=False),
        foot_height_baseline=foot_height_baseline.astype(np.float32, copy=False),
        ground_contact=ground_contact.astype(bool, copy=False),
        left_contact=left_contact.astype(bool, copy=False),
        right_contact=right_contact.astype(bool, copy=False),
        support_modes=support_modes.astype(np.int8, copy=False),
        safe_mask=safe_mask.astype(bool, copy=False),
        torso_tilt_deg=torso_tilt_deg.astype(np.float32, copy=False),
        joint_kinetic_energy=joint_kinetic_energy.astype(np.float32, copy=False),
        no_support=no_support.astype(bool, copy=False),
        used_lowest_energy_fallback=used_lowest_energy_fallback,
    )


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


def _resolve_anchor_body_indices(body_names: object) -> dict[str, int]:
    resolved_names = list(np.asarray(body_names).tolist())
    indices: dict[str, int] = {}
    missing: list[str] = []
    for body_name in ANCHOR_REQUIRED_BODY_NAMES:
        try:
            indices[body_name] = resolved_names.index(body_name)
        except ValueError:
            missing.append(body_name)
    if missing:
        raise ValueError(f"Anchor selection requires body names: {', '.join(missing)}")
    return indices


def _infer_single_foot_contact(foot_height_above_ground: np.ndarray, foot_velocity: np.ndarray) -> np.ndarray:
    heights = np.asarray(foot_height_above_ground, dtype=np.float32).reshape(-1)
    velocity = np.asarray(foot_velocity, dtype=np.float32)
    if velocity.shape != (heights.shape[0], 3):
        raise ValueError("foot_velocity must be shaped [num_frames, 3].")

    foot_xy_speed = np.linalg.norm(velocity[:, :2], axis=1)
    foot_z_speed = np.abs(velocity[:, 2])
    return (
        (heights < DEFAULT_ANCHOR_CONTACT_HEIGHT)
        & (foot_xy_speed < DEFAULT_ANCHOR_CONTACT_XY_SPEED)
        & (foot_z_speed < DEFAULT_ANCHOR_CONTACT_Z_SPEED)
    )


def _build_support_modes(left_contact: np.ndarray, right_contact: np.ndarray) -> np.ndarray:
    left = np.asarray(left_contact, dtype=bool).reshape(-1)
    right = np.asarray(right_contact, dtype=bool).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("left_contact and right_contact must have matching shapes.")

    support_modes = np.full(left.shape, ANCHOR_SUPPORT_MODE_NONE, dtype=np.int8)
    support_modes[left & ~right] = ANCHOR_SUPPORT_MODE_LEFT
    support_modes[~left & right] = ANCHOR_SUPPORT_MODE_RIGHT
    support_modes[left & right] = ANCHOR_SUPPORT_MODE_DOUBLE
    return support_modes


def _select_safe_low_kinetic_anchor_frame_indices(
    candidate_mask: np.ndarray,
    joint_kinetic_energy: np.ndarray,
    *,
    dt: float,
    min_spacing_seconds: float,
) -> tuple[np.ndarray, bool]:
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    energy = np.asarray(joint_kinetic_energy, dtype=np.float32).reshape(-1)
    if mask.shape != energy.shape:
        raise ValueError("candidate_mask and joint_kinetic_energy must have matching shapes.")
    if energy.size == 0:
        raise ValueError("Anchor selection requires at least one frame.")

    candidate_indices = [index for index in _find_local_minima(energy) if mask[index]]
    fallback_index: int | None = None
    if not candidate_indices and np.any(mask):
        safe_indices = np.nonzero(mask)[0]
        fallback_index = int(safe_indices[int(np.argmin(energy[safe_indices]))])
        candidate_indices = [fallback_index]

    min_spacing_frames = max(int(np.ceil(min_spacing_seconds / dt)), 1)
    selected_frames: list[int] = [0]
    for frame_index in sorted(candidate_indices, key=lambda index: (energy[index], index)):
        if frame_index == 0:
            continue
        if all(abs(frame_index - selected_frame) >= min_spacing_frames for selected_frame in selected_frames):
            selected_frames.append(int(frame_index))

    used_lowest_energy_fallback = fallback_index is not None and fallback_index in selected_frames
    return np.asarray(sorted(selected_frames), dtype=np.int64), used_lowest_energy_fallback


def _find_local_minima(values: np.ndarray) -> list[int]:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    minima: list[int] = []
    for index in range(1, scores.size - 1):
        left_score = scores[index - 1]
        right_score = scores[index + 1]
        if scores[index] <= left_score and scores[index] <= right_score:
            minima.append(index)
    return minima

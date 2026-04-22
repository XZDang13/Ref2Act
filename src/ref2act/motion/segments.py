from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEGMENT_TYPE_TIME_BIN = 0
SEGMENT_TYPE_AIR_MERGE = 1
DEFAULT_AIRBORNE_HEIGHT_MARGIN = 0.06
DEFAULT_FOOT_HEIGHT_BASELINE_PERCENTILE = 5.0
ANCHOR_FRAME_LABEL_RED = np.int8(0)
ANCHOR_FRAME_LABEL_YELLOW = np.int8(1)
ANCHOR_FRAME_LABEL_GREEN = np.int8(2)
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
DEFAULT_ANCHOR_SMOOTH_AIRBORNE_TIME = 0.05
DEFAULT_ANCHOR_SUPPORT_STABLE_TIME = 0.10
DEFAULT_ANCHOR_TAKEOFF_PRE_TIME = 0.10
DEFAULT_ANCHOR_TAKEOFF_POST_TIME = 0.10
DEFAULT_ANCHOR_LANDING_PRE_TIME = 0.15
DEFAULT_ANCHOR_LANDING_POST_TIME = 0.20
DEFAULT_ANCHOR_HIGH_SWING_HEIGHT = 0.30
DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG = 35.0
DEFAULT_ANCHOR_FALLBACK_TILT_THRESHOLD_DEG = 45.0
DEFAULT_ANCHOR_SCORE_NORMALIZATION_PERCENTILE = 90.0
DEFAULT_ANCHOR_GREEN_SCORE_PERCENTILE = 60.0
DEFAULT_ANCHOR_FALLBACK_COUNT = 3
DEFAULT_ANCHOR_MIN_FUTURE_HORIZON = 0.30
DEFAULT_ANCHOR_ENERGY_THRESHOLD = 0.70
DEFAULT_ANCHOR_POSE_THRESHOLD = 1.00
DEFAULT_ANCHOR_TILT_THRESHOLD_DEG = 30.0
DEFAULT_ANCHOR_MIN_INTERVAL_LENGTH = 0.15
DEFAULT_ANCHOR_MIN_SPACING = 0.35
DEFAULT_ANCHOR_POSE_WEIGHT = 0.75
DEFAULT_ANCHOR_TILT_WEIGHT = 0.75


@dataclass(frozen=True)
class AnchorSelectionMetadata:
    frame_labels: np.ndarray
    segment_start_times: np.ndarray
    segment_end_times: np.ndarray
    segment_labels: np.ndarray
    frame_indices: np.ndarray
    times: np.ndarray
    scores: np.ndarray
    support_modes: np.ndarray
    energy_norm: np.ndarray
    pose_extreme: np.ndarray
    torso_tilt_deg: np.ndarray

    def as_npz_dict(self) -> dict[str, np.ndarray]:
        return {
            "anchor_selection_version": np.asarray(1, dtype=np.int64),
            "anchor_frame_labels": self.frame_labels.astype(np.int8, copy=False),
            "anchor_segment_start_times": self.segment_start_times.astype(np.float32, copy=False),
            "anchor_segment_end_times": self.segment_end_times.astype(np.float32, copy=False),
            "anchor_segment_labels": self.segment_labels.astype(np.int8, copy=False),
            "anchor_frame_indices": self.frame_indices.astype(np.int64, copy=False),
            "anchor_times": self.times.astype(np.float32, copy=False),
            "anchor_scores": self.scores.astype(np.float32, copy=False),
            "anchor_support_modes": self.support_modes.astype(np.int8, copy=False),
            "anchor_energy_norm": self.energy_norm.astype(np.float32, copy=False),
            "anchor_pose_extreme": self.pose_extreme.astype(np.float32, copy=False),
            "anchor_torso_tilt_deg": self.torso_tilt_deg.astype(np.float32, copy=False),
        }


@dataclass(frozen=True)
class AnchorSelectionDiagnostics:
    metadata: AnchorSelectionMetadata
    frame_times: np.ndarray
    foot_height_above_ground: np.ndarray
    foot_height_baseline: np.ndarray
    airborne: np.ndarray
    smoothed_airborne: np.ndarray
    near_air_transition: np.ndarray
    near_landing_impact: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    support_modes: np.ndarray
    support_stable: np.ndarray
    frame_scores: np.ndarray
    energy_norm: np.ndarray
    pose_extreme: np.ndarray
    torso_tilt_deg: np.ndarray
    high_swing_pose: np.ndarray
    no_support: np.ndarray
    unstable_support: np.ndarray
    energy_fail: np.ndarray
    pose_fail: np.ndarray
    tilt_fail: np.ndarray
    strict_anchor_frame_indices: np.ndarray
    used_fallback_promotion: bool
    bootstrap_start_anchor_inserted: bool
    num_tail_trimmed_anchors: int


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

    joint_pos = np.asarray(log["joint_pos"], dtype=np.float32)
    joint_vel = np.asarray(log["joint_vel"], dtype=np.float32)
    body_pos_w = np.asarray(log["body_pos_w"], dtype=np.float32)
    body_lin_vel_w = np.asarray(log["body_lin_vel_w"], dtype=np.float32)
    num_frames = int(joint_pos.shape[0])
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
    airborne = ~ground_contact
    smoothed_airborne = _centered_majority_filter(
        airborne,
        _centered_window_size(DEFAULT_ANCHOR_SMOOTH_AIRBORNE_TIME, dt),
    )
    near_air_transition, near_landing_impact = _build_air_transition_masks(smoothed_airborne, frame_times)

    left_foot_vel = body_lin_vel_w[:, left_foot_index]
    right_foot_vel = body_lin_vel_w[:, right_foot_index]
    left_contact = _infer_single_foot_contact(foot_height_above_ground[:, 0], left_foot_vel)
    right_contact = _infer_single_foot_contact(foot_height_above_ground[:, 1], right_foot_vel)
    support_modes = _build_support_modes(left_contact, right_contact)
    support_stable = _support_mode_stable(
        support_modes,
        _centered_window_size(DEFAULT_ANCHOR_SUPPORT_STABLE_TIME, dt),
    )

    left_foot_speed = np.linalg.norm(left_foot_vel, axis=1)
    right_foot_speed = np.linalg.norm(right_foot_vel, axis=1)
    support_foot_speed = np.where(
        support_modes == ANCHOR_SUPPORT_MODE_LEFT,
        left_foot_speed,
        np.where(
            support_modes == ANCHOR_SUPPORT_MODE_RIGHT,
            right_foot_speed,
            np.minimum(left_foot_speed, right_foot_speed),
        ),
    ).astype(np.float32)
    root_vertical_speed = np.abs(body_lin_vel_w[:, pelvis_index, 2]).astype(np.float32)
    support_foot_speed_norm = _normalize_by_percentile(
        support_foot_speed,
        percentile=DEFAULT_ANCHOR_SCORE_NORMALIZATION_PERCENTILE,
    )
    root_vertical_speed_norm = _normalize_by_percentile(
        root_vertical_speed,
        percentile=DEFAULT_ANCHOR_SCORE_NORMALIZATION_PERCENTILE,
    )
    energy_norm = (0.5 * (support_foot_speed_norm + root_vertical_speed_norm)).astype(np.float32)

    joint_median = np.median(joint_pos, axis=0, keepdims=True)
    joint_scale = np.percentile(joint_pos, 95.0, axis=0, keepdims=True) - np.percentile(
        joint_pos,
        5.0,
        axis=0,
        keepdims=True,
    )
    pose_extreme = np.mean(np.abs(joint_pos - joint_median) / np.maximum(joint_scale, 1.0e-6), axis=1).astype(
        np.float32
    )
    pose_extreme_norm = _normalize_by_percentile(
        pose_extreme,
        percentile=DEFAULT_ANCHOR_SCORE_NORMALIZATION_PERCENTILE,
    )

    body_up = body_pos_w[:, torso_index] - body_pos_w[:, pelvis_index]
    body_up_norm = np.linalg.norm(body_up, axis=1)
    cos_tilt = np.divide(body_up[:, 2], body_up_norm, out=np.zeros_like(body_up_norm), where=body_up_norm > 1.0e-6)
    torso_tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))).astype(np.float32)
    torso_tilt_norm = _normalize_by_percentile(
        torso_tilt_deg,
        percentile=DEFAULT_ANCHOR_SCORE_NORMALIZATION_PERCENTILE,
    )

    single_support = (support_modes == ANCHOR_SUPPORT_MODE_LEFT) | (support_modes == ANCHOR_SUPPORT_MODE_RIGHT)
    high_swing_pose = single_support & (
        ((support_modes == ANCHOR_SUPPORT_MODE_LEFT) & (foot_height_above_ground[:, 1] > DEFAULT_ANCHOR_HIGH_SWING_HEIGHT))
        | ((support_modes == ANCHOR_SUPPORT_MODE_RIGHT) & (foot_height_above_ground[:, 0] > DEFAULT_ANCHOR_HIGH_SWING_HEIGHT))
    )

    no_support = support_modes == ANCHOR_SUPPORT_MODE_NONE
    unstable_support = (~support_stable) & (~no_support)
    strict_safe_mask = ~(
        airborne
        | no_support
        | (torso_tilt_deg > DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG)
        | near_landing_impact
        | high_swing_pose
    )
    relaxed_safe_mask = ~(
        airborne
        | no_support
        | (torso_tilt_deg > DEFAULT_ANCHOR_FALLBACK_TILT_THRESHOLD_DEG)
        | near_landing_impact
        | high_swing_pose
    )
    transition_penalty = near_air_transition.astype(np.float32)
    frame_scores = (
        energy_norm
        + DEFAULT_ANCHOR_POSE_WEIGHT * pose_extreme_norm
        + DEFAULT_ANCHOR_TILT_WEIGHT * torso_tilt_norm
        + transition_penalty
    ).astype(np.float32)
    green_score_cutoff = _masked_percentile(
        frame_scores,
        strict_safe_mask,
        percentile=DEFAULT_ANCHOR_GREEN_SCORE_PERCENTILE,
        default=np.inf,
    )

    frame_labels = np.full(num_frames, ANCHOR_FRAME_LABEL_YELLOW, dtype=np.int8)
    frame_labels[~strict_safe_mask] = ANCHOR_FRAME_LABEL_RED
    frame_labels[strict_safe_mask & (frame_scores <= green_score_cutoff)] = ANCHOR_FRAME_LABEL_GREEN
    frame_labels = _downgrade_short_green_runs(
        frame_labels,
        dt=dt,
        min_interval_seconds=DEFAULT_ANCHOR_MIN_INTERVAL_LENGTH,
    )
    strict_anchor_frame_indices = _select_anchor_frame_indices(
        frame_labels,
        frame_scores,
        dt=dt,
        min_spacing_seconds=DEFAULT_ANCHOR_MIN_SPACING,
    )

    used_fallback_promotion = False
    if strict_anchor_frame_indices.size == 0:
        fallback_anchor_indices = _select_fallback_anchor_frame_indices(
            relaxed_safe_mask,
            frame_scores,
            dt=dt,
            min_spacing_seconds=DEFAULT_ANCHOR_MIN_SPACING,
            max_anchors=DEFAULT_ANCHOR_FALLBACK_COUNT,
        )
        if fallback_anchor_indices.size > 0:
            frame_labels = _promote_mask_runs_to_green(
                frame_labels,
                relaxed_safe_mask,
                fallback_anchor_indices,
            )
            used_fallback_promotion = True

    anchor_frame_indices = _select_anchor_frame_indices(
        frame_labels,
        frame_scores,
        dt=dt,
        min_spacing_seconds=DEFAULT_ANCHOR_MIN_SPACING,
    )
    anchor_frame_indices, bootstrap_start_anchor_inserted, num_tail_trimmed_anchors = _postprocess_anchor_frame_indices(
        anchor_frame_indices,
        dt=dt,
        duration=duration,
        min_start_spacing_seconds=DEFAULT_ANCHOR_MIN_SPACING,
        min_future_horizon_seconds=DEFAULT_ANCHOR_MIN_FUTURE_HORIZON,
    )
    anchor_times = anchor_frame_indices.astype(np.float32) * np.float32(dt)

    segment_start_times, segment_end_times, segment_labels = _build_anchor_segments(
        frame_labels,
        dt=dt,
        duration=duration,
    )

    metadata = AnchorSelectionMetadata(
        frame_labels=frame_labels.astype(np.int8, copy=False),
        segment_start_times=segment_start_times,
        segment_end_times=segment_end_times,
        segment_labels=segment_labels,
        frame_indices=anchor_frame_indices.astype(np.int64, copy=False),
        times=anchor_times,
        scores=frame_scores[anchor_frame_indices].astype(np.float32, copy=False),
        support_modes=support_modes[anchor_frame_indices].astype(np.int8, copy=False),
        energy_norm=energy_norm[anchor_frame_indices].astype(np.float32, copy=False),
        pose_extreme=pose_extreme[anchor_frame_indices].astype(np.float32, copy=False),
        torso_tilt_deg=torso_tilt_deg[anchor_frame_indices].astype(np.float32, copy=False),
    )
    energy_fail_threshold = _masked_percentile(
        energy_norm,
        strict_safe_mask,
        percentile=DEFAULT_ANCHOR_GREEN_SCORE_PERCENTILE,
        default=np.inf,
    )
    pose_fail_threshold = _masked_percentile(
        pose_extreme,
        strict_safe_mask,
        percentile=DEFAULT_ANCHOR_GREEN_SCORE_PERCENTILE,
        default=np.inf,
    )
    tilt_fail_threshold = _masked_percentile(
        torso_tilt_deg,
        strict_safe_mask,
        percentile=DEFAULT_ANCHOR_GREEN_SCORE_PERCENTILE,
        default=np.inf,
    )
    energy_fail = strict_safe_mask & (energy_norm > energy_fail_threshold)
    pose_fail = strict_safe_mask & (pose_extreme > pose_fail_threshold)
    tilt_fail = (torso_tilt_deg > DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG) | (
        strict_safe_mask & (torso_tilt_deg > tilt_fail_threshold)
    )
    return AnchorSelectionDiagnostics(
        metadata=metadata,
        frame_times=frame_times.astype(np.float32, copy=False),
        foot_height_above_ground=foot_height_above_ground.astype(np.float32, copy=False),
        foot_height_baseline=foot_height_baseline.astype(np.float32, copy=False),
        airborne=airborne.astype(bool, copy=False),
        smoothed_airborne=smoothed_airborne.astype(bool, copy=False),
        near_air_transition=near_air_transition.astype(bool, copy=False),
        near_landing_impact=near_landing_impact.astype(bool, copy=False),
        left_contact=left_contact.astype(bool, copy=False),
        right_contact=right_contact.astype(bool, copy=False),
        support_modes=support_modes.astype(np.int8, copy=False),
        support_stable=support_stable.astype(bool, copy=False),
        frame_scores=frame_scores.astype(np.float32, copy=False),
        energy_norm=energy_norm.astype(np.float32, copy=False),
        pose_extreme=pose_extreme.astype(np.float32, copy=False),
        torso_tilt_deg=torso_tilt_deg.astype(np.float32, copy=False),
        high_swing_pose=high_swing_pose.astype(bool, copy=False),
        no_support=no_support.astype(bool, copy=False),
        unstable_support=unstable_support.astype(bool, copy=False),
        energy_fail=energy_fail.astype(bool, copy=False),
        pose_fail=pose_fail.astype(bool, copy=False),
        tilt_fail=tilt_fail.astype(bool, copy=False),
        strict_anchor_frame_indices=strict_anchor_frame_indices.astype(np.int64, copy=False),
        used_fallback_promotion=used_fallback_promotion,
        bootstrap_start_anchor_inserted=bootstrap_start_anchor_inserted,
        num_tail_trimmed_anchors=num_tail_trimmed_anchors,
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


def _centered_window_size(window_seconds: float, dt: float) -> int:
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    frames = max(int(np.ceil(window_seconds / dt)), 1)
    if frames % 2 == 0:
        frames += 1
    return frames


def _centered_majority_filter(mask: np.ndarray, window_size: int) -> np.ndarray:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if window_size <= 1 or values.size <= 1:
        return values.copy()

    radius = window_size // 2
    smoothed = np.zeros_like(values)
    for index in range(values.size):
        start = max(index - radius, 0)
        end = min(index + radius + 1, values.size)
        window = values[start:end]
        smoothed[index] = bool(np.count_nonzero(window) * 2 >= window.size)
    return smoothed


def _build_air_transition_masks(airborne: np.ndarray, frame_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(airborne, dtype=bool).reshape(-1)
    times = np.asarray(frame_times, dtype=np.float32).reshape(-1)
    if values.shape != times.shape:
        raise ValueError("airborne and frame_times must have matching shapes.")

    transition_mask = np.zeros_like(values)
    landing_mask = np.zeros_like(values)
    transition_frames = np.flatnonzero(values[1:] != values[:-1]) + 1
    for transition_frame in transition_frames.tolist():
        is_takeoff = bool(values[transition_frame])
        pre_time = DEFAULT_ANCHOR_TAKEOFF_PRE_TIME if is_takeoff else DEFAULT_ANCHOR_LANDING_PRE_TIME
        post_time = DEFAULT_ANCHOR_TAKEOFF_POST_TIME if is_takeoff else DEFAULT_ANCHOR_LANDING_POST_TIME
        transition_time = float(times[transition_frame])
        transition_window = (times >= transition_time - pre_time) & (times <= transition_time + post_time)
        transition_mask |= transition_window
        if not is_takeoff:
            landing_mask |= transition_window
    return transition_mask, landing_mask


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


def _support_mode_stable(support_modes: np.ndarray, window_size: int) -> np.ndarray:
    modes = np.asarray(support_modes, dtype=np.int8).reshape(-1)
    if window_size <= 1 or modes.size <= 1:
        return modes != ANCHOR_SUPPORT_MODE_NONE

    radius = window_size // 2
    stable = np.zeros(modes.shape, dtype=bool)
    for index in range(modes.size):
        start = max(index - radius, 0)
        end = min(index + radius + 1, modes.size)
        stable[index] = bool(
            modes[index] != ANCHOR_SUPPORT_MODE_NONE and np.all(modes[start:end] == modes[index])
        )
    return stable


def _normalize_by_percentile(values: np.ndarray, *, percentile: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    scale = max(float(np.percentile(array, percentile)), 1.0e-6)
    return array / scale


def _masked_percentile(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    percentile: float,
    default: float,
) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    resolved_mask = np.asarray(mask, dtype=bool).reshape(-1)
    if array.shape != resolved_mask.shape:
        raise ValueError("values and mask must have matching shapes.")
    if not np.any(resolved_mask):
        return float(default)
    return float(np.percentile(array[resolved_mask], percentile))


def _iter_label_runs(labels: np.ndarray) -> list[tuple[int, int, np.int8]]:
    values = np.asarray(labels, dtype=np.int8).reshape(-1)
    if values.size == 0:
        return []

    runs: list[tuple[int, int, np.int8]] = []
    run_start = 0
    run_label = values[0]
    for index in range(1, values.size):
        if values[index] != run_label:
            runs.append((run_start, index, np.int8(run_label)))
            run_start = index
            run_label = values[index]
    runs.append((run_start, values.size, np.int8(run_label)))
    return runs


def _downgrade_short_green_runs(labels: np.ndarray, *, dt: float, min_interval_seconds: float) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8).reshape(-1).copy()
    for run_start, run_end, run_label in _iter_label_runs(values):
        if run_label != ANCHOR_FRAME_LABEL_GREEN:
            continue
        if (run_end - run_start) * dt < min_interval_seconds:
            values[run_start:run_end] = ANCHOR_FRAME_LABEL_YELLOW
    return values


def _build_anchor_segments(
    frame_labels: np.ndarray,
    *,
    dt: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_start_times: list[np.float32] = []
    segment_end_times: list[np.float32] = []
    segment_labels: list[np.int8] = []

    for run_start, run_end, run_label in _iter_label_runs(frame_labels):
        segment_start_times.append(np.float32(run_start * dt))
        segment_end_times.append(np.float32(min(run_end * dt, duration)))
        segment_labels.append(np.int8(run_label))

    return (
        np.asarray(segment_start_times, dtype=np.float32),
        np.asarray(segment_end_times, dtype=np.float32),
        np.asarray(segment_labels, dtype=np.int8),
    )


def _select_anchor_frame_indices(
    frame_labels: np.ndarray,
    frame_scores: np.ndarray,
    *,
    dt: float,
    min_spacing_seconds: float,
) -> np.ndarray:
    labels = np.asarray(frame_labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(frame_scores, dtype=np.float32).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("frame_labels and frame_scores must have matching shapes.")

    min_spacing_frames = max(int(np.ceil(min_spacing_seconds / dt)), 1)
    anchor_indices: list[int] = []

    for run_start, run_end, run_label in _iter_label_runs(labels):
        if run_label != ANCHOR_FRAME_LABEL_GREEN:
            continue

        run_scores = scores[run_start:run_end]
        selected_frames = [run_start + int(np.argmin(run_scores))]
        local_minima = [
            run_start + relative_index
            for relative_index in _find_local_minima(run_scores)
        ]
        for frame_index in sorted(local_minima, key=lambda index: (scores[index], index)):
            if frame_index in selected_frames:
                continue
            if all(abs(frame_index - selected_frame) >= min_spacing_frames for selected_frame in selected_frames):
                selected_frames.append(frame_index)
        anchor_indices.extend(sorted(selected_frames))

    return np.asarray(anchor_indices, dtype=np.int64)


def _find_local_minima(values: np.ndarray) -> list[int]:
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    minima: list[int] = []
    for index in range(scores.size):
        left_score = scores[index - 1] if index > 0 else scores[index]
        right_score = scores[index + 1] if index + 1 < scores.size else scores[index]
        if scores[index] <= left_score and scores[index] <= right_score:
            minima.append(index)
    return minima


def _select_fallback_anchor_frame_indices(
    candidate_mask: np.ndarray,
    frame_scores: np.ndarray,
    *,
    dt: float,
    min_spacing_seconds: float,
    max_anchors: int,
) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
    scores = np.asarray(frame_scores, dtype=np.float32).reshape(-1)
    if mask.shape != scores.shape:
        raise ValueError("candidate_mask and frame_scores must have matching shapes.")
    if max_anchors < 1:
        return np.empty(0, dtype=np.int64)

    candidate_indices: list[int] = []
    for run_start, run_end in _find_true_runs(mask):
        run_scores = scores[run_start : run_end + 1]
        candidate_indices.extend(
            run_start + relative_index for relative_index in _find_local_minima(run_scores)
        )

    if not candidate_indices:
        return np.empty(0, dtype=np.int64)

    min_spacing_frames = max(int(np.ceil(min_spacing_seconds / dt)), 1)
    selected_frames: list[int] = []
    for frame_index in sorted(candidate_indices, key=lambda index: (scores[index], index)):
        if all(abs(frame_index - selected_frame) >= min_spacing_frames for selected_frame in selected_frames):
            selected_frames.append(frame_index)
            if len(selected_frames) >= max_anchors:
                break
    return np.asarray(sorted(selected_frames), dtype=np.int64)


def _promote_mask_runs_to_green(
    labels: np.ndarray,
    promotion_mask: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8).reshape(-1).copy()
    mask = np.asarray(promotion_mask, dtype=bool).reshape(-1)
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError("labels and promotion_mask must have matching shapes.")
    if indices.size == 0:
        return values

    for run_start, run_end in _find_true_runs(mask):
        if np.any((indices >= run_start) & (indices <= run_end)):
            values[run_start : run_end + 1] = ANCHOR_FRAME_LABEL_GREEN
    return values


def _postprocess_anchor_frame_indices(
    frame_indices: np.ndarray,
    *,
    dt: float,
    duration: float,
    min_start_spacing_seconds: float,
    min_future_horizon_seconds: float,
) -> tuple[np.ndarray, bool, int]:
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if min_start_spacing_seconds < 0.0:
        raise ValueError("min_start_spacing_seconds must be >= 0.")
    if min_future_horizon_seconds < 0.0:
        raise ValueError("min_future_horizon_seconds must be >= 0.")

    indices = np.unique(np.asarray(frame_indices, dtype=np.int64).reshape(-1))
    if np.any(indices < 0):
        raise ValueError("frame_indices must be non-negative.")

    bootstrap_start_anchor_inserted = indices.size == 0 or int(indices[0]) != 0
    if bootstrap_start_anchor_inserted:
        indices = np.concatenate((np.asarray([0], dtype=np.int64), indices))

    min_start_spacing_frames = int(np.ceil(float(min_start_spacing_seconds) / float(dt)))
    if min_start_spacing_frames > 0 and indices.size > 1:
        keep_mask = np.ones(indices.shape, dtype=bool)
        keep_mask[1:] = indices[1:] >= min_start_spacing_frames
        indices = indices[keep_mask]

    tail_cutoff_time = float(duration) - float(min_future_horizon_seconds)
    keep_mask = (indices == 0) | ((indices.astype(np.float32) * np.float32(dt)) <= np.float32(tail_cutoff_time))
    num_tail_trimmed_anchors = int(indices.size - int(np.count_nonzero(keep_mask)))
    indices = indices[keep_mask]
    return indices.astype(np.int64, copy=False), bootstrap_start_anchor_inserted, num_tail_trimmed_anchors

from __future__ import annotations

import numpy as np


SEGMENT_TYPE_TIME_BIN = 0


def build_time_bins(duration: float, bin_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed time bins without consulting motion or anchor metadata."""
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and > 0")
    if not np.isfinite(bin_size) or bin_size <= 0.0:
        raise ValueError("bin_size must be finite and > 0")

    ratio = float(duration) / float(bin_size)
    num_bins = max(int(np.ceil(ratio - 1.0e-9)), 1)
    boundaries = np.minimum(
        np.arange(num_bins + 1, dtype=np.float64) * float(bin_size),
        float(duration),
    )
    boundaries[-1] = float(duration)
    boundaries = boundaries.astype(np.float32)
    boundaries = boundaries[np.concatenate(([True], boundaries[1:] > boundaries[:-1]))]
    if boundaries.size < 2:
        raise ValueError("duration is too small to represent a positive time bin")
    starts = boundaries[:-1]
    ends = boundaries[1:]
    types = np.full(starts.shape, SEGMENT_TYPE_TIME_BIN, dtype=np.int64)
    return starts, ends, types

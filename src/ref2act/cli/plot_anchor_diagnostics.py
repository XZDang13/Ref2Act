from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ref2act.motion.segments import (
    AnchorSelectionMetadata,
    DEFAULT_AIRBORNE_HEIGHT_MARGIN,
    DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG,
    build_anchor_selection_diagnostics,
)


STORED_ANCHOR_METADATA_KEYS = (
    "anchor_selection_version",
    "anchor_frame_indices",
    "anchor_times",
    "anchor_joint_kinetic_energy",
)


@dataclass(frozen=True)
class AnchorDiagnosticsOutput:
    input_file: Path
    output_dir: Path
    overview_file: Path
    reasons_file: Path
    num_anchors: int
    selected_metadata: AnchorSelectionMetadata
    used_stored_anchor_metadata: bool
    stored_metadata_matches_recomputed: bool | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot anchor-selection diagnostics for a Ref2Act motion .npz file."
    )
    parser.add_argument(
        "--input_file",
        "-f",
        type=str,
        required=True,
        help="Path to a converted Ref2Act motion .npz file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the diagnostic figures. Defaults to <input_dir>/anchor_diagnostics.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Filename prefix for the output figures. Defaults to the input file stem.",
    )
    parser.add_argument(
        "--airborne-height-threshold",
        type=float,
        default=DEFAULT_AIRBORNE_HEIGHT_MARGIN,
        help="Airborne threshold used when recomputing per-frame diagnostics from the stored motion data.",
    )
    return parser


def _load_motion_log(motion_data: object) -> dict[str, object]:
    required_keys = (
        "fps",
        "joint_names",
        "body_names",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing_keys = [key for key in required_keys if key not in motion_data]
    if missing_keys:
        raise ValueError(f"Motion file is missing required arrays: {', '.join(missing_keys)}")
    return {
        "fps": np.asarray(motion_data["fps"]),
        "joint_names": np.asarray(motion_data["joint_names"]).tolist(),
        "body_names": np.asarray(motion_data["body_names"]).tolist(),
        "joint_pos": np.asarray(motion_data["joint_pos"], dtype=np.float32),
        "joint_vel": np.asarray(motion_data["joint_vel"], dtype=np.float32),
        "body_pos_w": np.asarray(motion_data["body_pos_w"], dtype=np.float32),
        "body_quat_w": np.asarray(motion_data["body_quat_w"], dtype=np.float32),
        "body_lin_vel_w": np.asarray(motion_data["body_lin_vel_w"], dtype=np.float32),
        "body_ang_vel_w": np.asarray(motion_data["body_ang_vel_w"], dtype=np.float32),
    }


def _load_stored_anchor_metadata(motion_data: object) -> AnchorSelectionMetadata | None:
    present_keys = [key for key in STORED_ANCHOR_METADATA_KEYS if key in motion_data]
    if not present_keys:
        return None
    missing_keys = [key for key in STORED_ANCHOR_METADATA_KEYS if key not in motion_data]
    if missing_keys:
        raise ValueError(f"Motion file contains partial anchor metadata: {', '.join(missing_keys)}")

    version = int(np.asarray(motion_data["anchor_selection_version"]).item())
    if version != 3:
        raise ValueError(
            f"Motion file has unsupported anchor_selection_version={version}; reconvert it with the v3 exporter."
        )

    return AnchorSelectionMetadata(
        frame_indices=np.asarray(motion_data["anchor_frame_indices"], dtype=np.int64).reshape(-1),
        times=np.asarray(motion_data["anchor_times"], dtype=np.float32).reshape(-1),
        joint_kinetic_energy=np.asarray(motion_data["anchor_joint_kinetic_energy"], dtype=np.float32).reshape(-1),
    )


def _metadata_matches(lhs: AnchorSelectionMetadata, rhs: AnchorSelectionMetadata) -> bool:
    return (
        np.array_equal(lhs.frame_indices, rhs.frame_indices)
        and np.allclose(lhs.times, rhs.times)
        and np.allclose(lhs.joint_kinetic_energy, rhs.joint_kinetic_energy)
    )


def _plot_overview(
    selected_metadata: AnchorSelectionMetadata,
    diagnostics,
    *,
    output_file: Path,
    clip_name: str,
) -> None:
    frame_times = diagnostics.frame_times
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"{clip_name} start-anchor plus safe local-minimum anchors: anchors={selected_metadata.frame_indices.size}",
        fontsize=14,
    )

    axes[0].plot(frame_times, diagnostics.joint_kinetic_energy, color="#1f77b4", label="sum(abs(joint_vel))")
    axes[0].scatter(
        selected_metadata.times,
        selected_metadata.joint_kinetic_energy,
        color="black",
        s=28,
        label="selected anchors",
        zorder=3,
    )
    axes[0].set_ylabel("kinetic proxy")
    axes[0].legend(loc="upper right")

    axes[1].fill_between(
        frame_times,
        0,
        1,
        where=diagnostics.safe_mask,
        transform=axes[1].get_xaxis_transform(),
        color="#2ca02c",
        alpha=0.25,
        label="safe candidate",
    )
    axes[1].fill_between(
        frame_times,
        0,
        1,
        where=~diagnostics.ground_contact,
        transform=axes[1].get_xaxis_transform(),
        color="#d62728",
        alpha=0.18,
        label="no ground contact",
    )
    axes[1].step(frame_times, diagnostics.support_modes, where="post", color="#2c3e50", label="support mode")
    axes[1].set_yticks([0, 1, 2, 3], ["none", "left", "right", "double"])
    axes[1].set_ylabel("support")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    axes[2].plot(frame_times, diagnostics.torso_tilt_deg, color="#9467bd", label="torso tilt")
    axes[2].axhline(DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG, color="#9467bd", ls="--", lw=1)
    axes[2].plot(frame_times, diagnostics.foot_height_above_ground[:, 0], color="#1f77b4", label="left foot height")
    axes[2].plot(frame_times, diagnostics.foot_height_above_ground[:, 1], color="#ff7f0e", label="right foot height")
    axes[2].set_ylabel("tilt deg / height m")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    for axis in axes:
        for anchor_time in selected_metadata.times.tolist():
            axis.axvline(float(anchor_time), color="black", lw=1, alpha=0.22)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def _plot_reasons(
    selected_metadata: AnchorSelectionMetadata,
    diagnostics,
    *,
    output_file: Path,
    clip_name: str,
) -> None:
    frame_times = diagnostics.frame_times
    rows = np.vstack(
        [
            diagnostics.ground_contact,
            diagnostics.support_modes != 0,
            diagnostics.torso_tilt_deg <= DEFAULT_ANCHOR_STRICT_TILT_THRESHOLD_DEG,
            diagnostics.safe_mask,
        ]
    ).astype(np.int8)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.4]})
    fig.suptitle(f"{clip_name} anchor safety mask", fontsize=14)
    axes[0].imshow(
        rows,
        aspect="auto",
        interpolation="nearest",
        cmap="Greens",
        extent=[float(frame_times[0]), float(frame_times[-1]), rows.shape[0], 0],
        vmin=0,
        vmax=1,
    )
    axes[0].set_yticks(np.arange(rows.shape[0]) + 0.5, ["ground", "support", "tilt ok", "safe"])

    axes[1].plot(frame_times, diagnostics.joint_kinetic_energy, color="#1f77b4")
    axes[1].scatter(
        selected_metadata.times,
        selected_metadata.joint_kinetic_energy,
        color="black",
        s=28,
        zorder=3,
    )
    axes[1].set_ylabel("kinetic proxy")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        for anchor_time in selected_metadata.times.tolist():
            axis.axvline(float(anchor_time), color="black", lw=1, alpha=0.22)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def generate_anchor_diagnostic_plots(
    input_file: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
    airborne_height_threshold: float = DEFAULT_AIRBORNE_HEIGHT_MARGIN,
) -> AnchorDiagnosticsOutput:
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {input_path}")

    destination_dir = Path(output_dir) if output_dir is not None else input_path.parent / "anchor_diagnostics"
    prefix = output_prefix or input_path.stem

    with np.load(input_path, allow_pickle=False) as motion_data:
        motion_log = _load_motion_log(motion_data)
        stored_metadata = _load_stored_anchor_metadata(motion_data)

    diagnostics = build_anchor_selection_diagnostics(
        motion_log,
        airborne_height_margin=airborne_height_threshold,
    )
    recomputed_metadata = diagnostics.metadata
    selected_metadata = stored_metadata if stored_metadata is not None else recomputed_metadata
    metadata_matches = None if stored_metadata is None else _metadata_matches(stored_metadata, recomputed_metadata)

    destination_dir.mkdir(parents=True, exist_ok=True)
    overview_file = destination_dir / f"{prefix}_overview.png"
    reasons_file = destination_dir / f"{prefix}_reasons.png"

    _plot_overview(selected_metadata, diagnostics, output_file=overview_file, clip_name=input_path.name)
    _plot_reasons(selected_metadata, diagnostics, output_file=reasons_file, clip_name=input_path.name)

    return AnchorDiagnosticsOutput(
        input_file=input_path,
        output_dir=destination_dir,
        overview_file=overview_file,
        reasons_file=reasons_file,
        num_anchors=int(selected_metadata.frame_indices.shape[0]),
        selected_metadata=selected_metadata,
        used_stored_anchor_metadata=stored_metadata is not None,
        stored_metadata_matches_recomputed=metadata_matches,
    )


def _print_summary(result: AnchorDiagnosticsOutput, selected_metadata: AnchorSelectionMetadata) -> None:
    print(
        "[INFO]: Anchor diagnostics summary: "
        f"anchors={result.num_anchors}, "
        f"stored_metadata={'yes' if result.used_stored_anchor_metadata else 'no'}, "
        f"matches_recomputed={result.stored_metadata_matches_recomputed}"
    )
    for anchor_index, (frame_index, anchor_time, energy) in enumerate(
        zip(
            selected_metadata.frame_indices,
            selected_metadata.times,
            selected_metadata.joint_kinetic_energy,
            strict=True,
        )
    ):
        print(
            f"[INFO]: Anchor A{anchor_index}: "
            f"frame={int(frame_index)}, "
            f"time={float(anchor_time):.3f}s, "
            f"energy={float(energy):.3f}"
        )
    print(f"[INFO]: Wrote overview: {result.overview_file}")
    print(f"[INFO]: Wrote reasons: {result.reasons_file}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_anchor_diagnostic_plots(
        args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        airborne_height_threshold=args.airborne_height_threshold,
    )
    if result.used_stored_anchor_metadata:
        print("[INFO]: Using stored anchor metadata")
        print(f"[INFO]: Stored metadata matches recomputed: {result.stored_metadata_matches_recomputed}")
    else:
        print("[INFO]: No stored anchor metadata found; recomputed anchor metadata")
    _print_summary(result, result.selected_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

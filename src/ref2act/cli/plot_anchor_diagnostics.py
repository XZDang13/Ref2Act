from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from ref2act.motion.segments import (
    ANCHOR_FRAME_LABEL_GREEN,
    ANCHOR_FRAME_LABEL_RED,
    ANCHOR_FRAME_LABEL_YELLOW,
    ANCHOR_SUPPORT_MODE_DOUBLE,
    ANCHOR_SUPPORT_MODE_LEFT,
    ANCHOR_SUPPORT_MODE_NONE,
    ANCHOR_SUPPORT_MODE_RIGHT,
    AnchorSelectionMetadata,
    DEFAULT_AIRBORNE_HEIGHT_MARGIN,
    build_anchor_selection_diagnostics,
)


STORED_ANCHOR_METADATA_KEYS = (
    "anchor_frame_labels",
    "anchor_segment_start_times",
    "anchor_segment_end_times",
    "anchor_segment_labels",
    "anchor_frame_indices",
    "anchor_times",
    "anchor_scores",
    "anchor_support_modes",
    "anchor_energy_norm",
    "anchor_pose_extreme",
    "anchor_torso_tilt_deg",
)
OPTIONAL_STORED_ANCHOR_METADATA_KEYS = ("anchor_joint_kinetic_energy",)
SUPPORT_MODE_LABELS = {
    ANCHOR_SUPPORT_MODE_NONE: "none",
    ANCHOR_SUPPORT_MODE_LEFT: "left",
    ANCHOR_SUPPORT_MODE_RIGHT: "right",
    ANCHOR_SUPPORT_MODE_DOUBLE: "double",
}


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
    present_keys = [
        key
        for key in (*STORED_ANCHOR_METADATA_KEYS, *OPTIONAL_STORED_ANCHOR_METADATA_KEYS)
        if key in motion_data
    ]
    if not present_keys:
        return None

    missing_keys = [key for key in STORED_ANCHOR_METADATA_KEYS if key not in motion_data]
    if missing_keys:
        raise ValueError(f"Motion file contains partial anchor metadata: {', '.join(missing_keys)}")

    anchor_times = np.asarray(motion_data["anchor_times"], dtype=np.float32).reshape(-1)
    if "anchor_joint_kinetic_energy" in motion_data:
        joint_kinetic_energy = np.asarray(motion_data["anchor_joint_kinetic_energy"], dtype=np.float32).reshape(-1)
    else:
        joint_kinetic_energy = np.zeros(anchor_times.shape, dtype=np.float32)

    return AnchorSelectionMetadata(
        frame_labels=np.asarray(motion_data["anchor_frame_labels"], dtype=np.int8).reshape(-1),
        segment_start_times=np.asarray(motion_data["anchor_segment_start_times"], dtype=np.float32).reshape(-1),
        segment_end_times=np.asarray(motion_data["anchor_segment_end_times"], dtype=np.float32).reshape(-1),
        segment_labels=np.asarray(motion_data["anchor_segment_labels"], dtype=np.int8).reshape(-1),
        frame_indices=np.asarray(motion_data["anchor_frame_indices"], dtype=np.int64).reshape(-1),
        times=anchor_times,
        scores=np.asarray(motion_data["anchor_scores"], dtype=np.float32).reshape(-1),
        support_modes=np.asarray(motion_data["anchor_support_modes"], dtype=np.int8).reshape(-1),
        energy_norm=np.asarray(motion_data["anchor_energy_norm"], dtype=np.float32).reshape(-1),
        pose_extreme=np.asarray(motion_data["anchor_pose_extreme"], dtype=np.float32).reshape(-1),
        torso_tilt_deg=np.asarray(motion_data["anchor_torso_tilt_deg"], dtype=np.float32).reshape(-1),
        joint_kinetic_energy=joint_kinetic_energy,
    )


def _metadata_matches(lhs: AnchorSelectionMetadata, rhs: AnchorSelectionMetadata) -> bool:
    return (
        np.array_equal(lhs.frame_labels, rhs.frame_labels)
        and np.array_equal(lhs.segment_labels, rhs.segment_labels)
        and np.array_equal(lhs.frame_indices, rhs.frame_indices)
        and np.array_equal(lhs.support_modes, rhs.support_modes)
        and np.allclose(lhs.segment_start_times, rhs.segment_start_times)
        and np.allclose(lhs.segment_end_times, rhs.segment_end_times)
        and np.allclose(lhs.times, rhs.times)
        and np.allclose(lhs.scores, rhs.scores)
        and np.allclose(lhs.energy_norm, rhs.energy_norm)
        and np.allclose(lhs.pose_extreme, rhs.pose_extreme)
        and np.allclose(lhs.torso_tilt_deg, rhs.torso_tilt_deg)
        and np.allclose(lhs.joint_kinetic_energy, rhs.joint_kinetic_energy)
    )


def _format_green_intervals(metadata: AnchorSelectionMetadata) -> str:
    intervals = [
        f"{float(start):.2f}-{float(end):.2f}s"
        for start, end, label in zip(
            metadata.segment_start_times,
            metadata.segment_end_times,
            metadata.segment_labels,
            strict=True,
        )
        if int(label) == int(ANCHOR_FRAME_LABEL_GREEN)
    ]
    return ", ".join(intervals) if intervals else "none"


def _plot_overview(
    selected_metadata: AnchorSelectionMetadata,
    diagnostics,
    *,
    output_file: Path,
    clip_name: str,
) -> None:
    frame_times = diagnostics.frame_times
    dt = float(frame_times[1] - frame_times[0]) if frame_times.shape[0] > 1 else 0.0
    label_counts = {
        "red": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_RED)),
        "yellow": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_YELLOW)),
        "green": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_GREEN)),
    }

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(16, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [0.45, 1.0, 1.0, 1.0]},
    )
    fig.suptitle(
        f"{clip_name} anchor overview: anchors={selected_metadata.frame_indices.size}, "
        f"red={label_counts['red']}, yellow={label_counts['yellow']}, green={label_counts['green']}",
        fontsize=14,
    )

    label_cmap = ListedColormap(["#d62728", "#f2c14e", "#2ca02c"])
    axes[0].imshow(
        selected_metadata.frame_labels[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=label_cmap,
        extent=[float(frame_times[0]), float(frame_times[-1] + dt), 0, 1],
        vmin=0,
        vmax=2,
    )
    axes[0].set_yticks([])
    axes[0].set_ylabel("labels")
    for anchor_index, frame_index in enumerate(selected_metadata.frame_indices.tolist()):
        anchor_time = float(frame_index) * dt
        axes[0].axvline(anchor_time, color="k", lw=1, alpha=0.7)
        axes[0].text(anchor_time, 1.03, f"A{anchor_index}", rotation=90, va="bottom", ha="center", fontsize=8)

    axes[1].plot(frame_times, diagnostics.foot_height_above_ground[:, 0], label="left foot height", color="#1f77b4")
    axes[1].plot(frame_times, diagnostics.foot_height_above_ground[:, 1], label="right foot height", color="#ff7f0e")
    axes[1].axhline(0.04, color="0.3", ls="--", lw=1, label="contact height")
    axes[1].axhline(0.30, color="0.5", ls=":", lw=1, label="high swing threshold")
    axes[1].fill_between(
        frame_times,
        0,
        1,
        where=diagnostics.airborne,
        transform=axes[1].get_xaxis_transform(),
        color="#d62728",
        alpha=0.12,
        label="airborne",
    )
    axes[1].fill_between(
        frame_times,
        0,
        1,
        where=diagnostics.near_air_transition,
        transform=axes[1].get_xaxis_transform(),
        color="#9467bd",
        alpha=0.10,
        label="near transition",
    )
    for anchor_time in selected_metadata.times.tolist():
        axes[1].axvline(float(anchor_time), color="k", lw=1, alpha=0.25)
    axes[1].set_ylabel("meters")
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    axes[2].step(frame_times, diagnostics.support_modes, where="post", color="#2c3e50", label="support mode")
    axes[2].fill_between(
        frame_times,
        0,
        1,
        where=diagnostics.support_stable,
        transform=axes[2].get_xaxis_transform(),
        color="#2ca02c",
        alpha=0.12,
        label="support stable",
    )
    axes[2].fill_between(
        frame_times,
        0,
        1,
        where=diagnostics.no_support,
        transform=axes[2].get_xaxis_transform(),
        color="#d62728",
        alpha=0.10,
        label="no support",
    )
    for anchor_time in selected_metadata.times.tolist():
        axes[2].axvline(float(anchor_time), color="k", lw=1, alpha=0.25)
    axes[2].set_yticks([0, 1, 2, 3], ["none", "left", "right", "double"])
    axes[2].set_ylabel("support")
    axes[2].legend(loc="upper right", fontsize=8)

    axes[3].plot(frame_times, diagnostics.energy_norm, label="energy_norm", color="#1f77b4")
    axes[3].plot(frame_times, diagnostics.pose_extreme, label="pose_extreme", color="#ff7f0e")
    axes[3].plot(
        frame_times,
        diagnostics.torso_tilt_deg / 30.0,
        label="torso_tilt / 30deg",
        color="#2ca02c",
    )
    axes[3].axhline(0.70, color="#1f77b4", ls="--", lw=1)
    axes[3].axhline(1.00, color="#ff7f0e", ls="--", lw=1)
    axes[3].axhline(1.00, color="#2ca02c", ls="--", lw=1, alpha=0.6)
    for frame_index in selected_metadata.frame_indices.tolist():
        frame_time = float(frame_index) * dt
        axes[3].axvline(frame_time, color="k", lw=1, alpha=0.25)
        axes[3].scatter([frame_time], [float(diagnostics.energy_norm[frame_index])], color="#1f77b4", s=28)
        axes[3].scatter([frame_time], [float(diagnostics.pose_extreme[frame_index])], color="#ff7f0e", s=28)
        axes[3].scatter(
            [frame_time],
            [float(diagnostics.torso_tilt_deg[frame_index] / 30.0)],
            color="#2ca02c",
            s=28,
        )
    axes[3].set_ylabel("normalized")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    dt = float(frame_times[1] - frame_times[0]) if frame_times.shape[0] > 1 else 0.0
    reason_names = [
        "airborne",
        "near_transition",
        "no_support",
        "unstable_support",
        "high_swing",
        "energy_fail",
        "pose_fail",
        "tilt_fail",
    ]
    reason_rows = np.vstack(
        [
            diagnostics.airborne,
            diagnostics.near_air_transition,
            diagnostics.no_support,
            diagnostics.unstable_support,
            diagnostics.high_swing_pose,
            diagnostics.energy_fail,
            diagnostics.pose_fail,
            diagnostics.tilt_fail,
        ]
    ).astype(np.int8)

    fig = plt.figure(figsize=(16, 10))
    grid_spec = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 0.9])
    ax_heatmap = fig.add_subplot(grid_spec[0])
    ax_metrics = fig.add_subplot(grid_spec[1], sharex=ax_heatmap)
    ax_table = fig.add_subplot(grid_spec[2])
    fig.suptitle(f"{clip_name} anchor diagnostics", fontsize=14)

    ax_heatmap.imshow(
        reason_rows,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#ffffff", "#444444"]),
        extent=[float(frame_times[0]), float(frame_times[-1] + dt), len(reason_names) - 0.5, -0.5],
    )
    ax_heatmap.set_yticks(range(len(reason_names)), reason_names)
    ax_heatmap.set_ylabel("rejection reason")
    for anchor_index, anchor_time in enumerate(selected_metadata.times.tolist()):
        ax_heatmap.axvline(float(anchor_time), color="#d62728", lw=1.2, alpha=0.75)
        ax_heatmap.text(float(anchor_time), -0.8, f"A{anchor_index}", color="#d62728", rotation=90, va="bottom", ha="center", fontsize=8)

    ax_metrics.plot(frame_times, diagnostics.energy_norm, label="energy_norm", color="#1f77b4")
    ax_metrics.plot(frame_times, diagnostics.pose_extreme, label="pose_extreme", color="#ff7f0e")
    ax_metrics.plot(frame_times, diagnostics.torso_tilt_deg, label="torso_tilt_deg", color="#2ca02c")
    ax_metrics.axhline(0.70, color="#1f77b4", ls="--", lw=1)
    ax_metrics.axhline(1.00, color="#ff7f0e", ls="--", lw=1)
    ax_metrics.axhline(30.0, color="#2ca02c", ls="--", lw=1)
    for anchor_time in selected_metadata.times.tolist():
        ax_metrics.axvline(float(anchor_time), color="#d62728", lw=1, alpha=0.5)
    ax_metrics.set_ylabel("metric value")
    ax_metrics.legend(loc="upper right", ncol=3, fontsize=8)
    ax_metrics.set_xlabel("time [s]")

    ax_table.axis("off")
    cell_text = [
        [
            f"A{anchor_index}",
            str(int(frame_index)),
            f"{float(anchor_time):.2f}",
            SUPPORT_MODE_LABELS.get(int(support_mode), str(int(support_mode))),
            f"{float(score):.3f}",
            f"{float(energy):.3f}",
            f"{float(pose):.3f}",
            f"{float(tilt):.2f}",
        ]
        for anchor_index, (frame_index, anchor_time, support_mode, score, energy, pose, tilt) in enumerate(
            zip(
                selected_metadata.frame_indices,
                selected_metadata.times,
                selected_metadata.support_modes,
                selected_metadata.scores,
                selected_metadata.energy_norm,
                selected_metadata.pose_extreme,
                selected_metadata.torso_tilt_deg,
                strict=True,
            )
        )
    ]
    if not cell_text:
        cell_text = [["-", "-", "-", "-", "-", "-", "-", "-"]]

    table = ax_table.table(
        cellText=cell_text,
        colLabels=["anchor", "frame", "time [s]", "support", "score", "energy", "pose", "tilt deg"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
    label_counts = {
        "red": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_RED)),
        "yellow": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_YELLOW)),
        "green": int(np.sum(selected_metadata.frame_labels == ANCHOR_FRAME_LABEL_GREEN)),
    }
    print(
        "[INFO]: Anchor diagnostics summary: "
        f"anchors={result.num_anchors}, "
        f"red={label_counts['red']}, "
        f"yellow={label_counts['yellow']}, "
        f"green={label_counts['green']}"
    )
    print(f"[INFO]: Green intervals: {_format_green_intervals(selected_metadata)}")
    for anchor_index, (frame_index, anchor_time, support_mode, score) in enumerate(
        zip(
            selected_metadata.frame_indices,
            selected_metadata.times,
            selected_metadata.support_modes,
            selected_metadata.scores,
            strict=True,
        )
    ):
        support_label = SUPPORT_MODE_LABELS.get(int(support_mode), str(int(support_mode)))
        print(
            f"[INFO]: Anchor A{anchor_index}: "
            f"frame={int(frame_index)}, "
            f"time={float(anchor_time):.3f}s, "
            f"support={support_label}, "
            f"score={float(score):.4f}"
        )
    if result.used_stored_anchor_metadata:
        if result.stored_metadata_matches_recomputed:
            print("[INFO]: Stored anchor metadata matches the current diagnostics implementation.")
        else:
            print("[WARN]: Stored anchor metadata differs from the current diagnostics implementation.")
    else:
        print("[INFO]: No stored anchor metadata found; figures use recomputed anchors.")
    print(f"[INFO]: Saved overview figure to {result.overview_file}")
    print(f"[INFO]: Saved reason figure to {result.reasons_file}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_anchor_diagnostic_plots(
        args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        airborne_height_threshold=args.airborne_height_threshold,
    )
    _print_summary(result, result.selected_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

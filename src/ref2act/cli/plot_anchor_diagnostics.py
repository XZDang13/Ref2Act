from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ref2act.motion.library import MotionLib


@dataclass(frozen=True)
class AnchorDiagnosticsOutput:
    input_file: Path
    output_dir: Path
    overview_file: Path
    num_anchors: int
    anchor_frames: tuple[int, ...]
    anchor_times: tuple[float, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the reset anchors already stored beside a Retargeter final_motion.npz."
    )
    parser.add_argument("--input_file", "-f", required=True, help="Motion directory or final_motion.npz path.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output-prefix", default=None)
    return parser


def generate_anchor_diagnostic_plots(
    input_file: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> AnchorDiagnosticsOutput:
    motion_lib = MotionLib(input_file)
    clip = motion_lib.clips[0]
    input_path = Path(clip.source)
    destination = Path(output_dir) if output_dir is not None else input_path.parent / "anchor_diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or input_path.parent.name
    overview_file = destination / f"{prefix}_anchors.png"

    frame_times = torch.arange(clip.num_frames, dtype=torch.float32) / clip.fps
    joint_speed = clip.joint_vel.abs().sum(dim=-1).cpu()
    anchor_frames = () if clip.anchor_frame_indices is None else tuple(int(v) for v in clip.anchor_frame_indices.cpu())
    anchor_times = () if clip.anchor_times is None else tuple(float(v) for v in clip.anchor_times.cpu())

    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(frame_times.numpy(), joint_speed.numpy(), label="sum(abs(joint_vel))")
    for index, anchor_time in enumerate(anchor_times):
        axis.axvline(anchor_time, color="black", alpha=0.35, linewidth=1)
        axis.text(anchor_time, axis.get_ylim()[1], f"A{index}", va="top", ha="left", fontsize=8)
    axis.set_title(f"{clip.name}: configured reset anchors ({len(anchor_times)})")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("joint speed proxy")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(overview_file, dpi=180)
    plt.close(fig)

    return AnchorDiagnosticsOutput(
        input_file=input_path,
        output_dir=destination,
        overview_file=overview_file,
        num_anchors=len(anchor_times),
        anchor_frames=anchor_frames,
        anchor_times=anchor_times,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_anchor_diagnostic_plots(
        args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
    )
    print(f"[INFO]: anchors={result.num_anchors} frames={list(result.anchor_frames)}")
    print(f"[INFO]: wrote {result.overview_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

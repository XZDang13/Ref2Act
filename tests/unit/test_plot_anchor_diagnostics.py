from pathlib import Path

import numpy as np

from ref2act.cli.plot_anchor_diagnostics import build_parser, generate_anchor_diagnostic_plots, main
from ref2act.motion.segments import build_anchor_selection_diagnostics

_BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]


def _build_motion_log(*, fps: float = 10.0, num_frames: int = 30) -> dict[str, object]:
    dt = 1.0 / fps
    frame_index = np.arange(num_frames, dtype=np.float32)
    torso_x = 0.1 + 0.05 * np.cos((2.0 * np.pi * frame_index) / 10.0)

    joint_pos = np.zeros((num_frames, 2), dtype=np.float32)
    joint_pos[:, 0] = 0.1 * np.sin((2.0 * np.pi * frame_index) / 10.0)
    body_pos_w = np.zeros((num_frames, len(_BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = 1.0
    body_pos_w[:, 1, 0] = torso_x
    body_pos_w[:, 1, 2] = 2.0
    body_pos_w[:, 2, 1] = -0.1
    body_pos_w[:, 3, 1] = 0.1
    body_quat_w = np.zeros((num_frames, len(_BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0

    return {
        "fps": np.asarray(fps, dtype=np.float32),
        "joint_names": np.asarray(["joint_0", "joint_1"]),
        "body_names": np.asarray(_BODY_NAMES),
        "joint_pos": joint_pos,
        "joint_vel": np.gradient(joint_pos, dt, axis=0).astype(np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": np.gradient(body_pos_w, dt, axis=0).astype(np.float32),
        "body_ang_vel_w": np.zeros((num_frames, len(_BODY_NAMES), 3), dtype=np.float32),
    }


def _write_motion_file(path: Path, *, include_anchor_metadata: bool) -> None:
    motion_log = _build_motion_log()
    payload = dict(motion_log)
    if include_anchor_metadata:
        payload.update(build_anchor_selection_diagnostics(motion_log).metadata.as_npz_dict())
    np.savez(path, **payload)


def test_plot_anchor_diagnostics_parser_defaults() -> None:
    args = build_parser().parse_args(["--input_file", "motion.npz"])

    assert args.input_file == "motion.npz"
    assert args.output_dir is None
    assert args.output_prefix is None
    assert args.airborne_height_threshold == 0.06


def test_generate_anchor_diagnostic_plots_uses_stored_anchor_metadata(tmp_path: Path) -> None:
    input_file = tmp_path / "stored_anchor.npz"
    output_dir = tmp_path / "figures"
    _write_motion_file(input_file, include_anchor_metadata=True)

    result = generate_anchor_diagnostic_plots(
        input_file,
        output_dir=output_dir,
        output_prefix="stored_anchor",
    )

    assert result.used_stored_anchor_metadata is True
    assert result.stored_metadata_matches_recomputed is True
    assert result.num_anchors > 0
    assert result.overview_file == output_dir / "stored_anchor_overview.png"
    assert result.reasons_file == output_dir / "stored_anchor_reasons.png"
    assert result.overview_file.exists()
    assert result.reasons_file.exists()
    assert result.overview_file.stat().st_size > 0
    assert result.reasons_file.stat().st_size > 0


def test_plot_anchor_diagnostics_main_recomputes_when_anchor_metadata_missing(
    tmp_path: Path,
    capsys,
) -> None:
    input_file = tmp_path / "recompute_anchor.npz"
    _write_motion_file(input_file, include_anchor_metadata=False)

    exit_code = main(
        [
            "--input_file",
            str(input_file),
            "--output_dir",
            str(tmp_path / "diagnostics"),
            "--output-prefix",
            "recompute_anchor",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No stored anchor metadata found" in captured.out
    assert (tmp_path / "diagnostics" / "recompute_anchor_overview.png").exists()
    assert (tmp_path / "diagnostics" / "recompute_anchor_reasons.png").exists()

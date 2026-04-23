from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from ref2act.motion.library import MotionLib
from ref2act.motion.sampling import MotionSampler, SamplingStrategy, SegmentSource
from ref2act.motion.segments import ANCHOR_FRAME_LABEL_GREEN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_anchor_motion_file(
    path: Path,
    *,
    fps: float = 20.0,
    num_frames: int = 20,
    num_anchors: int = 10,
) -> None:
    joint_pos = np.zeros((num_frames, 1), dtype=np.float32)
    joint_vel = np.zeros_like(joint_pos)
    body_pos_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_quat_w = np.zeros((num_frames, 1, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, 1, 3), dtype=np.float32)
    duration = num_frames / fps
    anchor_times = np.linspace(0.05, duration - 0.05, num_anchors, dtype=np.float32)

    np.savez(
        path,
        fps=np.asarray(fps, dtype=np.float32),
        joint_names=np.asarray(["joint_0"]),
        body_names=np.asarray(["body_0"]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        anchor_segment_start_times=np.asarray([0.0], dtype=np.float32),
        anchor_segment_end_times=np.asarray([duration], dtype=np.float32),
        anchor_segment_labels=np.asarray([ANCHOR_FRAME_LABEL_GREEN], dtype=np.int64),
        anchor_frame_indices=np.arange(num_anchors, dtype=np.int64),
        anchor_times=anchor_times,
    )


def _snapshot_anchor_probabilities(
    sampler: MotionSampler,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion_fail_counts = torch.stack([fail_counts.sum() for fail_counts in sampler.bin_fail_counts], dim=0)
    motion_sample_counts = torch.stack([sample_counts.sum() for sample_counts in sampler.bin_sample_counts], dim=0)
    motion_probs = sampler._build_guarded_sampling_probabilities(
        motion_fail_counts,
        motion_sample_counts,
        temperature=temperature,
    )

    per_motion_anchor_probs: list[torch.Tensor] = []
    global_anchor_probs: list[torch.Tensor] = []
    for motion_id in range(sampler.motion_lib.num_motions):
        anchor_probs = sampler._build_guarded_sampling_probabilities(
            sampler.bin_fail_counts[motion_id],
            sampler.bin_sample_counts[motion_id],
            temperature=temperature,
            eligible_mask=sampler.bin_reset_eligible[motion_id],
        )
        per_motion_anchor_probs.append(anchor_probs)
        global_anchor_probs.append(anchor_probs * motion_probs[motion_id])

    return torch.stack(per_motion_anchor_probs, dim=0), torch.stack(global_anchor_probs, dim=0)


def _plot_anchor_adaptation(
    *,
    snapshot_steps: list[int],
    mean_anchor_slot_probs: list[np.ndarray],
    hard_anchor_index: int,
    output_file: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    heatmap = np.stack(mean_anchor_slot_probs, axis=0)
    image = axes[0].imshow(heatmap, aspect="auto", cmap="viridis")
    axes[0].set_title("Mean within-motion anchor distribution")
    axes[0].set_xlabel("Anchor index")
    axes[0].set_ylabel("Step")
    axes[0].set_xticks(np.arange(heatmap.shape[1]))
    axes[0].set_yticks(np.arange(len(snapshot_steps)), labels=[str(step) for step in snapshot_steps])
    axes[0].axvline(hard_anchor_index, color="white", linestyle="--", linewidth=1.0)
    fig.colorbar(image, ax=axes[0], label="reset probability")

    anchor_indices = np.arange(heatmap.shape[1])
    for step, probs in zip(snapshot_steps, mean_anchor_slot_probs, strict=True):
        axes[1].plot(anchor_indices, probs, marker="o", linewidth=1.5, label=f"step {step}")
    axes[1].axvline(hard_anchor_index, color="black", linestyle="--", linewidth=1.0, label="hard anchor")
    axes[1].set_title("Anchor distribution snapshots every 10 steps")
    axes[1].set_xlabel("Anchor index")
    axes[1].set_ylabel("mean within-motion reset probability")
    axes[1].set_xticks(anchor_indices)
    axes[1].legend(ncol=4, fontsize=8)

    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def test_anchor_failure_weighted_sampling_plot_tracks_learning_dynamics(tmp_path: Path) -> None:
    torch.manual_seed(7)

    num_motions = 100
    num_anchors = 10
    num_steps = 100
    num_envs = 1024
    hard_anchor_index = 2
    easy_fail_prob = 0.05
    hard_fail_prob_before_learning = 0.45
    hard_fail_prob_after_learning = easy_fail_prob
    failure_decay = 0.9

    motion_files = []
    for motion_index in range(num_motions):
        motion_file = tmp_path / f"motion_{motion_index:03d}.npz"
        _write_anchor_motion_file(motion_file, num_anchors=num_anchors)
        motion_files.append(motion_file)

    motion_lib = MotionLib(motion_files)
    sampler = MotionSampler(
        num_envs=num_envs,
        motion_lib=motion_lib,
        dt=0.05,
        failure_decay=failure_decay,
        failure_weight_uniform_mix=0.1,
        failure_weight_max_uniform_ratio=2.5,
        segment_source=SegmentSource.Anchor,
        device=torch.device("cpu"),
    )

    env_ids = torch.arange(num_envs, device=sampler.device)
    snapshot_steps = list(range(0, num_steps + 1, 10))
    mean_anchor_slot_probs: list[np.ndarray] = []
    global_hard_anchor_mass: list[float] = []

    per_motion_anchor_probs, global_anchor_probs = _snapshot_anchor_probabilities(sampler)
    mean_anchor_slot_probs.append(per_motion_anchor_probs.mean(dim=0).cpu().numpy())
    global_hard_anchor_mass.append(float(global_anchor_probs[:, hard_anchor_index].sum().item()))

    for step in range(num_steps):
        reset_sample = sampler.reset(env_ids, strategy=SamplingStrategy.FailureWeighted)

        sampled_is_hard = reset_sample.target_bin_indices == hard_anchor_index
        fail_probs = torch.full((num_envs,), easy_fail_prob, dtype=torch.float32, device=sampler.device)
        fail_probs[sampled_is_hard] = (
            hard_fail_prob_before_learning if step < 60 else hard_fail_prob_after_learning
        )
        fail_mask = torch.rand(num_envs, device=sampler.device) < fail_probs
        if torch.any(fail_mask):
            failure_times = torch.clamp(
                reset_sample.times[fail_mask] + 0.01,
                max=float(motion_lib.motion_durations.max().item()),
            )
            sampler.record_failures(env_ids[fail_mask], times=failure_times)

        if (step + 1) % 10 == 0:
            per_motion_anchor_probs, global_anchor_probs = _snapshot_anchor_probabilities(sampler)
            mean_anchor_slot_probs.append(per_motion_anchor_probs.mean(dim=0).cpu().numpy())
            global_hard_anchor_mass.append(float(global_anchor_probs[:, hard_anchor_index].sum().item()))

    plot_file = tmp_path / "anchor_adaptation_distribution.png"
    _plot_anchor_adaptation(
        snapshot_steps=snapshot_steps,
        mean_anchor_slot_probs=mean_anchor_slot_probs,
        hard_anchor_index=hard_anchor_index,
        output_file=plot_file,
    )

    assert len(mean_anchor_slot_probs) == len(snapshot_steps)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0

    initial_hard_mass = global_hard_anchor_mass[0]
    peak_hard_mass = max(global_hard_anchor_mass[1:7])
    hard_mass_at_60 = global_hard_anchor_mass[snapshot_steps.index(60)]
    final_hard_mass = global_hard_anchor_mass[-1]

    assert initial_hard_mass == pytest.approx(1.0 / num_anchors, abs=1.0e-6)
    assert peak_hard_mass > initial_hard_mass + 0.10
    assert hard_mass_at_60 > initial_hard_mass + 0.10
    assert final_hard_mass < hard_mass_at_60 - 0.05

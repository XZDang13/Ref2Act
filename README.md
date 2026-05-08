# Ref2Act

Ref2Act provides motion-tracking environments built on Isaac Lab, offline motion tooling, and a MuJoCo bridge for sim-to-sim evaluation. The codebase now uses a `src/ref2act` layout with explicit subpackages for runtime envs, motion processing, robot configs, shared utilities, and bridge code.

## Installation

Core install:

```bash
python -m pip install -e .
```

MuJoCo bridge install:

```bash
python -m pip install -e ".[sim2sim]"
```

Dependency boundaries:

- `ref2act.common` and `ref2act.motion` are pure Python/Torch modules.
- `ref2act.envs` and `ref2act.robots` require Isaac Lab.
- `ref2act.bridges.mujoco` requires the `sim2sim` extra.

## Architecture

The package is organized by responsibility:

- `ref2act.common`: shared math, interpolation, and buffer utilities
- `ref2act.motion`: motion loading, sampling strategies, segmentation, smoothing, and GMR I/O
- `ref2act.envs.motion_tracking`: the Isaac Lab motion-tracking environment and its action, observation, reward, termination, curriculum, and visualization helpers
- `ref2act.robots.g1` and `ref2act.robots.pi_plus`: robot-specific articulation and env config exports
- `ref2act.bridges.mujoco`: MuJoCo runtime bridge
- `ref2act.assets`: packaged robot and scene assets accessed through `importlib.resources`

See [docs/repo-layout.md](/home/xdang/Desktop/Ref2Act/docs/repo-layout.md) for the layout contract.

## Training Usage

Canonical runtime imports:

```python
from ref2act.envs.motion_tracking import MotionTrackingEnv
from ref2act.motion import SamplingStrategy
from ref2act.robots.g1 import G1MotionTrackingEnvCfg

cfg = G1MotionTrackingEnvCfg()
cfg.expert_motion_file = "path/to/motion.npz"
cfg.scene.num_envs = 32
cfg.sampling_strategy = SamplingStrategy.FailureWeighted
cfg.weight_fail = 0.5
cfg.weight_novel = 0.3
cfg.cap_beta = 2.0
cfg.adaptive_uniform_ratio = 0.1
cfg.adaptive_alpha = 0.001

env = MotionTrackingEnv(cfg)
obs, info = env.reset()
```

With `SamplingStrategy.FailureWeighted`, Ref2Act biases both motion choice across clips and reset choice within the chosen clip using a MOSAIC-style mixture:

```text
p = weight_fail * p_fail + weight_novel * p_uct + weight_uniform * p_uniform
```

`p_uct` is the novelty term, computed from raw sample counts. `weight_fail` and `weight_novel` can be warm-started with `motion_sampling_warmup_s`, ramped with `motion_sampling_ramp_s`, and scheduled by `motion_sampling_schedule`; any remaining mass becomes uniform probability. Motion-level failure rates are capped by `cap_beta * mean_fail_rate` before normalization. Bin-level failure scores use an EMA controlled by `adaptive_alpha`, add a MOSAIC uniform floor through `adaptive_uniform_ratio`, and can be smoothed with `adaptive_kernel_size` and `adaptive_lambda`.

When `cfg.segment_source == SegmentSource.Time`, the within-clip distribution is learned over time bins. When `cfg.segment_source == SegmentSource.Anchor`, conversion always includes frame `0` as a reset anchor and adds safe local minima of `sum(abs(joint_vel))`; the same MOSAIC/UCT mixer then runs over those anchor bins. Long spans between reset anchors are split using `cfg.bin_size`, while each bin still resets from the nearest selected anchor.

The package registers these Gym environments when Isaac Lab is available:

- `G1MotionTracking-v0`
- `PiPlusMotionTracking-v0`
- `G1MotionTrackingRough-v0`
- `PiPlusMotionTrackingRough-v0`

## Motion Conversion

Single-file conversion:

```bash
ref2act-convert --input_file path/to/motion.pkl --output_file path/to/motion.npz
```

Batch conversion:

```bash
ref2act-convert-batch --input_dir path/to/mocap --output_dir path/to/converted_mocap
```

Useful options:

- `--target-fps`: resample the exported clip
- `--smooth-motion`: smooth root and joint trajectories before export
- `--segment-bin-size`: emit segment metadata for failure-weighted sampling
- `--segment-method anchor`: keep legacy `segment_*` output and also export anchor metadata

The converter emits `.npz` clips compatible with `ref2act.motion.MotionLib` and the motion-tracking env configs.

### Tutorial: Using `ref2act-convert`

`ref2act-convert` converts one retargeted GMR `.pkl` motion file into the `.npz` format used by Ref2Act training and tooling.

1. Install the package in editable mode:

```bash
python -m pip install -e .
```

2. Check the available flags:

```bash
ref2act-convert --help
```

3. Run the simplest conversion:

```bash
ref2act-convert \
  --input_file data/motions/walk.pkl \
  --output_file data/motions/walk.npz
```

If `--output_file` is omitted, the converter writes the output next to the input file with the same stem and an `.npz` suffix.

4. Convert a motion with settings that are usually useful for training:

```bash
ref2act-convert \
  --input_file data/motions/walk.pkl \
  --output_file data/motions/walk_train.npz \
  --target-fps 60 \
  --smooth-motion \
  --smoothing-profile medium \
  --segment-bin-size 0.3
```

This does four things:

- resamples the clip to `60 Hz`
- smooths root and joint trajectories before export
- writes `segment_start_times`, `segment_end_times`, and `segment_types`
- produces a clip that can be used directly by `MotionLib` and the motion-tracking env

5. Export anchor-aware metadata in addition to the standard segment metadata:

```bash
ref2act-convert \
  --input_file data/motions/jump.pkl \
  --output_file data/motions/jump_anchor.npz \
  --segment-method anchor \
  --segment-bin-size 0.3
```

Anchor mode still writes the normal `segment_*` arrays for compatibility, and also writes the v3 anchor metadata contract:

- `anchor_selection_version`
- `anchor_frame_indices`
- `anchor_times`
- `anchor_joint_kinetic_energy`

Use this mode when you want the converted file to carry stable reset-anchor annotations alongside the current time-segment metadata. Anchor selection treats frame `0` as a safe reset anchor, then uses local minima of `sum(abs(joint_vel))` that pass contact/support and torso-tilt safety checks. With `cfg.segment_source = SegmentSource.Anchor`, the sampler uses every exported `anchor_time` as a reset unit and splits long anchor spans into `cfg.bin_size` failure-attribution bins. Older anchor files must be reconverted because the v2 label and segment fields are no longer loaded in anchor mode.

6. Adjust the vertical offset or airborne detection if the imported motion needs it:

```bash
ref2act-convert \
  --input_file data/motions/custom.pkl \
  --output_file data/motions/custom.npz \
  --height_offset 0.02 \
  --airborne-height-threshold 0.08
```

`--height_offset` shifts the root height before export. `--airborne-height-threshold` changes how aggressively the converter marks both feet as airborne when it builds segment metadata.

For many files at once, use `ref2act-convert-batch`:

```bash
ref2act-convert-batch \
  --input_dir data/motions/raw \
  --output_dir data/motions/converted \
  --target-fps 60 \
  --smooth-motion \
  --segment-method anchor
```

## Anchor Diagnostics

If a motion was converted with `--segment-method anchor`, you can inspect the selected anchors with:

```bash
ref2act-plot-anchor-diagnostics --input_file data/motions/jump_anchor.npz
```

The tool writes two figures by default under `anchor_diagnostics/` next to the input file:

- `<stem>_overview.png`: kinetic energy, selected anchors, contact/support state, and torso tilt
- `<stem>_reasons.png`: safety mask and rejection reasons used by the v3 start-anchor plus local-minimum selector

Example with explicit output settings:

```bash
ref2act-plot-anchor-diagnostics \
  --input_file data/motions/jump_anchor.npz \
  --output_dir data/motions/anchor_diagnostics \
  --output-prefix jump_anchor
```

Useful options:

- `--output_dir`: choose where the figures are saved
- `--output-prefix`: control the output filename prefix
- `--airborne-height-threshold`: recompute diagnostics with a different airborne threshold

The CLI prints a short summary to stdout, including:

- how many anchors were found
- whether the lowest-energy safe fallback was used
- frame, time, support mode, and raw joint-kinetic-energy proxy for each anchor

If the `.npz` file already contains v3 `anchor_*` arrays, the tool uses them for the anchor overlay and checks whether they still match the current diagnostics implementation.

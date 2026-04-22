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
cfg.failure_weight_uniform_mix = 0.1
cfg.failure_weight_max_uniform_ratio = 2.5

env = MotionTrackingEnv(cfg)
obs, info = env.reset()
```

With `SamplingStrategy.FailureWeighted`, Ref2Act now biases both motion choice across clips and reset-bin choice within the chosen clip. `cfg.failure_weight_uniform_mix` blends each learned failure distribution with uniform mass, and `cfg.failure_weight_max_uniform_ratio` caps each eligible motion/bin at a multiple of its uniform share so the sampler cannot collapse onto only a few motions or bins.

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

Anchor mode still writes the normal `segment_*` arrays for compatibility, and also writes `anchor_*` arrays such as:

- `anchor_frame_labels`
- `anchor_segment_start_times`
- `anchor_segment_end_times`
- `anchor_frame_indices`
- `anchor_times`

Use this mode when you want the converted file to carry stable reset-anchor annotations alongside the current time-segment metadata.

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

- `<stem>_overview.png`: label timeline, foot-height traces, support mode, and soft metrics with anchors overlaid
- `<stem>_reasons.png`: rejection-reason heatmap plus a per-anchor summary table

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
- green interval ranges
- frame, time, support mode, and score for each anchor

If the `.npz` file already contains `anchor_*` arrays, the tool uses them for the anchor overlay and also checks whether they still match the current diagnostics implementation.

# Ref2Act

Ref2Act provides motion-tracking environments built on Isaac Lab. The package registers
`G1MotionTracking-v0`, includes robot and environment configuration, and ships `convert`
and `convert-batch` CLIs for turning GMR `.pkl` motion files into the `.npz` format
consumed by the runtime.

## Prerequisites

- Python 3.10 or newer
- An Isaac Lab / Isaac Sim environment that provides the `isaaclab` Python package
- A working PyTorch install compatible with that Isaac Lab environment

## Install

Install the package in editable mode:

```bash
python -m pip install -e .
```

If you need the MuJoCo bridge in `Ref2Act.sim2sim`, install the optional extra:

```bash
python -m pip install -e ".[sim2sim]"
```

## Motion Data

Ref2Act expects expert motions in `.npz` format. Convert a GMR pickle file with:

```bash
convert --input_file path/to/motion.pkl --output_file path/to/motion.npz
```

If `--output_file` is omitted, the converter writes the `.npz` file next to the input file.

Convert a whole folder tree while preserving the directory layout under a new output root:

```bash
convert-batch --input_dir path/to/mocap --output_dir path/to/converted_mocap
```

Use `--num-agents` to convert multiple same-FPS motions concurrently in one shared Isaac runtime:

```bash
convert-batch --input_dir path/to/mocap --output_dir path/to/converted_mocap --num-agents 4
```

Batch runs print a summary that now includes total wall-clock time, for example:

```text
[INFO]: Batch conversion summary: discovered=120, converted=118, skipped=1, failed=1, elapsed=84.237s
```

## Minimal Usage

```python
from Ref2Act.env import G1MotionTrackingEnv
from Ref2Act.config.env_cfg import G1MotionTrackingEnvCfg
from Ref2Act.sampler import SamplingStrategy

cfg = G1MotionTrackingEnvCfg()
cfg.expert_motion_file = "path/to/motion.npz"
cfg.scene.num_envs = 32
cfg.sampling_strategy = SamplingStrategy.FailureWeighted

env = G1MotionTrackingEnv(cfg)
obs, info = env.reset()
```

`cfg.expert_motion_file` also accepts a sequence of `.npz` paths, in which case each
environment samples a clip ID plus a local time inside that clip.

The default configs are tuned for large Isaac Lab runs. Lower `cfg.scene.num_envs` for local
debugging before attempting full-scale training.

The motion-tracking configs now attach structured domain randomization by default through
`cfg.events`. For evaluation or ablations, you can drop interval pushes while keeping
domain randomization with:

```python
from Ref2Act.config.env_cfg import G1DomainRandCfg

cfg.events = G1DomainRandCfg()
```

## Repository Notes

- Runtime behavior is split across `Ref2Act/env.py`, `Ref2Act/sampler.py`, `Ref2Act/observation.py`,
  `Ref2Act/rewards.py`, and `Ref2Act/termination.py`.
- The tracked package surface is the `Ref2Act/` package plus the `convert` console script.
- Automated tests are not wired up in this repository yet, so verification is currently manual.

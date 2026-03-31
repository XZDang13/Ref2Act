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

env = MotionTrackingEnv(cfg)
obs, info = env.reset()
```

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

The converter emits `.npz` clips compatible with `ref2act.motion.MotionLib` and the motion-tracking env configs.

# Ref2Act

Ref2Act is a humanoid motion-tracking reinforcement-learning task for Isaac Lab 3.0. Motion retargeting is intentionally outside this repository; training and sim2sim consume the current Retargeter output directly.

## Runtime contract

- Isaac Lab 3.0 is the only supported Isaac runtime.
- Every quaternion inside Ref2Act is scalar-last XYZW. MuJoCo `qpos` and `xquat` are converted explicitly at the bridge boundary because MuJoCo uses WXYZ.
- Body positions and velocities refer to the body-link origin in world coordinates. CoM quantities are derived explicitly from link state and the local CoM offset.
- `cfg.expert_motion_file` is the only motion input. It accepts one explicit path or a list of explicit paths; each path must be either a motion directory containing `final_motion.npz` or that exact file.
- Dataset-root discovery, manifests, CSV catalogs, GMR pickle conversion, and legacy NPZ schemas are not supported.

## Retargeter motion layout

`final_motion.npz` must contain finite, topology-consistent arrays:

```text
fps, robot, joint_names, body_names
joint_pos, joint_vel
body_pos_w, body_quat_xyzw
body_lin_vel_w, body_ang_vel_w
```

The body arrays are `[T, B, 3/4]`, joint arrays are `[T, J]`, and names define their topology. Quaternions must be unit XYZW values.

The motion body topology may be a subset of the Isaac articulation body topology. This supports assets such as the G1 29-DOF USD, which contains fixed/helper rigid bodies that Retargeter does not export. Every motion body must exist in the asset, and every environment-required root, anchor, key, end-effector, and foot body must exist in the motion. CoM rewards use only the shared motion/asset body set.

An optional `reset_anchors.json` may sit next to the NPZ:

```json
{
  "enabled": true,
  "anchors": [
    {"frame": 24, "time_s": 0.8},
    {"frame": 90, "time_s": 3.0}
  ]
}
```

Enabled anchors must be sorted, unique, in range, and satisfy `time_s == frame / fps`. Ref2Act never inserts frame 0. Anchor sampling uses only this sidecar; time sampling uses fixed bins from `bin_size` only.

## Configuration example

```python
cfg.expert_motion_file = [
    "/data/motions/75_07_stageii",
    "/data/motions/88_09_stageii/final_motion.npz",
]
```

Packed cache fingerprints include the resolved NPZ and sidecar paths, sizes, and modification times.

## Diagnostics and sim2sim

The read-only diagnostics command visualizes configured anchors without selecting new ones:

```bash
ref2act-plot-anchor-diagnostics -f /data/motions/75_07_stageii
```

The MuJoCo bridge builds the same noise-free policy groups as training. For the supported G1 23-DOF task the layout is `motion=29`, `robot=78`, total `107` values; it does not construct the privileged training group.

## Validation

Run CPU unit tests with:

```bash
conda run -n isaaclab pytest tests/unit
```

Isaac USD/config integration and headless environment smoke tests require the same `isaaclab` environment and a working Isaac installation.

# Repo Layout

This repository uses a `src/ref2act` package layout. New code should follow these boundaries.

## Package Map

- `src/ref2act/common`
  - Pure shared helpers: buffers, quaternion/math utilities, interpolation helpers
- `src/ref2act/motion`
  - Pure motion-domain code: motion library, sampling strategies, segment metadata, smoothing, GMR I/O, resampling
- `src/ref2act/envs/motion_tracking`
  - Isaac Lab runtime logic for the motion-tracking task
  - Keep env orchestration here, including reset-time state application and env-facing visualization helpers
- `src/ref2act/robots/g1`
  - G1 articulation/config exports and robot-specific presets/constants
- `src/ref2act/robots/pi_plus`
  - Pi Plus articulation/config exports and robot-specific presets/constants
- `src/ref2act/bridges/mujoco`
  - MuJoCo bridge runtime and viewer glue
- `src/ref2act/assets`
  - Packaged robot and scene assets
- `src/ref2act/cli`
  - Thin orchestration layer for motion conversion commands

## Import Rules

- `ref2act.common` must stay independent of Isaac Lab and MuJoCo.
- `ref2act.motion` must stay independent of Isaac Lab and MuJoCo.
- `ref2act.envs` may depend on Isaac Lab but should depend on `ref2act.motion`, not the reverse.
- `ref2act.robots` may depend on Isaac Lab and env config types.
- `ref2act.bridges.mujoco` may depend on MuJoCo and `ref2act.motion`, not on Isaac Lab.
- Asset access should go through `ref2act.assets.asset_path`, `robot_asset_path`, or `scene_asset_path`.

## Tests

Tests live under `tests/` only.

- `tests/unit`: pure or stubbed unit tests
- `tests/integration`: asset- and config-level integration checks
- `tests/fixtures/motions`: packaged motion fixtures used by tests

Do not add new tests under the legacy `test/` path.

## Packaging

- The distribution name remains `Ref2Act`.
- The Python import package is `ref2act`.
- Console entry points are:
  - `ref2act-convert`
  - `ref2act-convert-batch`

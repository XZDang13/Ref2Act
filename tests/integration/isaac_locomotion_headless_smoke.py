"""Manual GPU smoke test for G1 flat locomotion without a motion file."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--terrain",
        choices=("flat", "slope", "uneven", "mixed"),
        default="flat",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    launcher = AppLauncher(args)

    import gymnasium as gym
    import torch

    import ref2act  # noqa: F401
    from ref2act.robots.g1 import (
        G1FlatLocomotionEnvCfg,
        G1MixedTerrainLocomotionEnvCfg,
        G1SlopeLocomotionEnvCfg,
        G1UnevenLocomotionEnvCfg,
    )

    cfg_classes = {
        "flat": G1FlatLocomotionEnvCfg,
        "slope": G1SlopeLocomotionEnvCfg,
        "uneven": G1UnevenLocomotionEnvCfg,
        "mixed": G1MixedTerrainLocomotionEnvCfg,
    }
    cfg = cfg_classes[args.terrain]()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("G1FlatLocomotion-v0", cfg=cfg)
    failure: BaseException | None = None
    try:
        observation, _ = env.reset()
        expected_shapes = {
            "command": (args.num_envs, 3),
            "robot": (args.num_envs, 78),
            "privilege": (args.num_envs, 84),
        }
        actual_shapes = {name: tuple(value.shape) for name, value in observation.items()}
        if actual_shapes != expected_shapes:
            raise RuntimeError(f"Unexpected observations: {actual_shapes}")

        for _ in range(args.steps):
            observation, reward, terminated, truncated, _ = env.step(
                torch.zeros((args.num_envs, 23), device=env.unwrapped.device)
            )
            tensors = list(observation.values()) + [reward, terminated, truncated]
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise RuntimeError("Locomotion smoke produced a non-finite tensor.")
        print(
            f"locomotion smoke passed: terrain={args.terrain}, "
            f"num_envs={args.num_envs}, steps={args.steps}",
            flush=True,
        )
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
    finally:
        env.close()
        launcher.app.close()
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

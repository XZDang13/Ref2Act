"""Manual Isaac Lab headless smoke for the current Retargeter contract."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion")
    parser.add_argument("--num-envs", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    launcher = AppLauncher(args)

    import gymnasium as gym
    import torch

    import ref2act  # noqa: F401
    from ref2act.isaac_compat import to_torch
    from ref2act.robots.g1 import G1MotionTrackingEnvCfg

    cfg = G1MotionTrackingEnvCfg()
    cfg.expert_motion_file = args.motion
    cfg.scene.num_envs = args.num_envs
    env = gym.make("G1MotionTracking-v0", cfg=cfg)
    failure: BaseException | None = None
    try:
        observation, _ = env.reset()
        sensor = env.unwrapped.contact_sensor
        required_support_bodies = {"left_ankle_roll_link", "right_ankle_roll_link"}
        if not required_support_bodies.issubset(sensor.body_names):
            raise RuntimeError(f"Missing support bodies in nested contact sensor: {sensor.body_names}")
        contact_forces = to_torch(sensor.data.net_forces_w)
        if tuple(contact_forces.shape) != (args.num_envs, len(sensor.body_names), 3):
            raise RuntimeError(f"Unexpected contact tensor shape: {tuple(contact_forces.shape)}")
        print("isaac smoke reset passed", flush=True)
        for _ in range(10):
            observation, reward, terminated, truncated, _ = env.step(
                torch.zeros((args.num_envs, 23), device=env.unwrapped.device)
            )
            tensors = list(observation.values()) + [reward, terminated, truncated]
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise RuntimeError("Isaac smoke produced a non-finite observation, reward, or state tensor.")
        if observation["motion"].shape[-1] != 29 or observation["robot"].shape[-1] != 78:
            raise RuntimeError(f"Unexpected policy groups: {[(key, value.shape) for key, value in observation.items()]}")
        policy_dim = observation["motion"].shape[-1] + observation["robot"].shape[-1]
        print(f"isaac smoke passed: num_envs={args.num_envs}, policy_dim={policy_dim}", flush=True)
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
    finally:
        env.close()
        launcher.app.close()
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

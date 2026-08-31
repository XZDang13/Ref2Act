#!/usr/bin/env python3
"""Benchmark Ref2Act motion tracking without running a learning algorithm."""

from __future__ import annotations

import argparse
import subprocess
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--warmup-steps", type=int, default=4)
parser.add_argument("--action-mode", choices=("zero", "random"), default="zero")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args.steps <= 0:
    parser.error("--steps must be positive")
if args.warmup_steps < 0:
    parser.error("--warmup-steps must be non-negative")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _gpu_process_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    pid = str(__import__("os").getpid())
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == pid:
            return int(fields[1])
    return None


def main() -> None:
    import gymnasium as gym
    import torch

    from ref2act.envs.motion_tracking.registry import register_envs
    from ref2act.robots.g1 import G1MotionTrackingEnvCfg

    register_envs()
    cfg = G1MotionTrackingEnvCfg()
    cfg.expert_motion_file = args.motion
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.reference_motion_viewer_enabled = False

    env = gym.make("G1MotionTracking-v0", cfg=cfg)
    try:
        env.reset()
        device = env.unwrapped.device
        generator = torch.Generator(device=device).manual_seed(0)

        def make_actions() -> torch.Tensor:
            if args.action_mode == "zero":
                return torch.zeros((args.num_envs, int(cfg.action_space)), device=device)
            return torch.randn(
                (args.num_envs, int(cfg.action_space)),
                generator=generator,
                device=device,
            )

        for _ in range(args.warmup_steps):
            env.step(make_actions())
        env.reset()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        terminated_count = 0
        timeout_count = 0
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(make_actions())
            terminated_count += int(terminated.sum().item())
            timeout_count += int(truncated.sum().item())
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        transitions = args.num_envs * args.steps
        print(
            "MOTION_TRACKING_RUNTIME "
            f"terrain=flat action_mode={args.action_mode} "
            f"envs={args.num_envs} warmup_steps={args.warmup_steps} steps={args.steps} "
            f"elapsed_s={elapsed:.6f} transitions_per_s={transitions / elapsed:.1f} "
            f"terminated={terminated_count} timeouts={timeout_count} "
            f"gpu_process_memory_mib={_gpu_process_memory_mib()}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

#!/usr/bin/env python3
"""Benchmark Ref2Act locomotion simulation without running a learning algorithm."""

from __future__ import annotations

import argparse
import subprocess
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--terrain",
    choices=("flat", "generated-flat", "slope", "uneven", "mixed"),
    default="mixed",
)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--warmup-steps", type=int, default=4)
parser.add_argument("--action-mode", choices=("zero", "random"), default="zero")
parser.add_argument("--disable-collision-filter", action="store_true")
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
    """Return this process' GPU allocation reported by nvidia-smi."""

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

    from ref2act.envs.locomotion.registry import register_envs
    from ref2act.envs.locomotion.terrain import Ref2ActFlatTerrainCfg
    from ref2act.robots.g1 import (
        G1FlatLocomotionEnvCfg,
        G1MixedTerrainLocomotionEnvCfg,
        G1SlopeLocomotionEnvCfg,
        G1UnevenLocomotionEnvCfg,
    )

    cfg_types = {
        "flat": G1FlatLocomotionEnvCfg,
        "generated-flat": G1MixedTerrainLocomotionEnvCfg,
        "slope": G1SlopeLocomotionEnvCfg,
        "uneven": G1UnevenLocomotionEnvCfg,
        "mixed": G1MixedTerrainLocomotionEnvCfg,
    }
    env_ids = {
        "flat": "G1FlatLocomotion-v0",
        "generated-flat": "G1MixedTerrainLocomotion-v0",
        "slope": "G1SlopeLocomotion-v0",
        "uneven": "G1UnevenLocomotion-v0",
        "mixed": "G1MixedTerrainLocomotion-v0",
    }
    register_envs()
    cfg = cfg_types[args.terrain]()
    if args.terrain == "generated-flat":
        cfg.terrain.terrain_generator.sub_terrains = {
            "flat": Ref2ActFlatTerrainCfg(proportion=1.0)
        }
    cfg.scene.num_envs = args.num_envs
    if args.disable_collision_filter:
        cfg.scene.filter_collisions = False
    cfg.sim.device = args.device or "cuda:0"
    explicit_collision_filter = bool(cfg.scene.filter_collisions)

    env = gym.make(env_ids[args.terrain], cfg=cfg)
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
        termination_breakdown = {
            "fallen": 0.0,
            "tilted": 0.0,
            "illegal_contact": 0.0,
        }
        for _ in range(args.steps):
            _, _, terminated, truncated, extras = env.step(make_actions())
            terminated_count += int(terminated.sum().item())
            timeout_count += int(truncated.sum().item())
            log = extras.get("log", {})
            for name in termination_breakdown:
                value = log.get(f"Termination/{name}")
                if value is not None:
                    termination_breakdown[name] += float(value.item()) * args.num_envs
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        transitions = args.num_envs * args.steps
        print(
            "LOCOMOTION_RUNTIME "
            f"terrain={args.terrain} action_mode={args.action_mode} "
            f"explicit_collision_filter={explicit_collision_filter} "
            f"envs={args.num_envs} warmup_steps={args.warmup_steps} steps={args.steps} "
            f"elapsed_s={elapsed:.6f} transitions_per_s={transitions / elapsed:.1f} "
            f"terminated={terminated_count} timeouts={timeout_count} "
            f"fallen={round(termination_breakdown['fallen'])} "
            f"tilted={round(termination_breakdown['tilted'])} "
            f"illegal_contact={round(termination_breakdown['illegal_contact'])} "
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

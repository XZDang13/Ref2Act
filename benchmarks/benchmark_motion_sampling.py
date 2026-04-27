from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ref2act.motion import MotionLib


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_calls(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync(device)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark packed vs grouped MotionLib sampling.")
    parser.add_argument("motion_files", nargs="+", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    motion_lib = MotionLib(args.motion_files, device=device)
    motion_ids = torch.randint(motion_lib.num_motions, (args.batch_size,), device=device)
    durations = motion_lib.get_duration(motion_ids)
    times = torch.rand(args.batch_size, device=device) * durations
    position_offsets = torch.randn(args.batch_size, 3, device=device)

    packed_elapsed = _time_calls(
        lambda: motion_lib.sample_motion(motion_ids, times, position_offsets=position_offsets),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    grouped_elapsed = _time_calls(
        lambda: motion_lib._sample_motion_grouped(motion_ids, times, position_offsets=position_offsets),
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
    )

    print(f"packed_enabled={motion_lib._packed_sampling_enabled}")
    print(f"batch_size={args.batch_size} iterations={args.iterations} device={device}")
    print(f"packed_total_s={packed_elapsed:.6f} packed_ms={packed_elapsed / args.iterations * 1000.0:.3f}")
    print(f"grouped_total_s={grouped_elapsed:.6f} grouped_ms={grouped_elapsed / args.iterations * 1000.0:.3f}")
    if packed_elapsed > 0.0:
        print(f"speedup={grouped_elapsed / packed_elapsed:.2f}x")


if __name__ == "__main__":
    main()

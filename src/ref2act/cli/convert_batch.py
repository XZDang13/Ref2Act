from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from isaaclab.app import AppLauncher

from .convert import (
    ConversionFailure,
    ConversionOptions,
    add_conversion_arguments,
    convert_motion_files,
    create_conversion_runtime,
    peek_motion_fps,
)


@dataclass(frozen=True)
class BatchFailure:
    input_file: Path
    output_file: Path
    error: str


@dataclass(frozen=True)
class PendingBatchJob:
    input_file: Path
    output_file: Path
    fps: int


@dataclass
class BatchConversionSummary:
    discovered: int = 0
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0
    failures: list[BatchFailure] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 0 if self.failed == 0 else 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a folder of GMR pickle motion files into the Ref2Act .npz format."
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing GMR .pkl files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination folder for converted .npz files.",
    )
    parser.add_argument(
        "--num-agents",
        type=_positive_int,
        default=1,
        help="Number of motions to convert concurrently in one shared Isaac runtime.",
    )
    add_conversion_arguments(parser)
    return parser


def discover_motion_files(input_dir: str | Path) -> list[Path]:
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input motion directory does not exist or is not a directory: {input_path}")
    return sorted(path for path in input_path.rglob("*.pkl") if path.is_file())


def map_output_file(input_file: str | Path, input_dir: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_file)
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    relative_path = input_path.relative_to(input_root)
    return (output_root / relative_path).with_suffix(".npz")


def build_runtime_cache(
    device: str,
    num_agents: int,
) -> tuple[Callable[[int], object], Callable[[], None]]:
    runtimes: dict[int, object] = {}

    def resolve(motion_fps: int):
        runtime = runtimes.get(motion_fps)
        if runtime is None:
            runtime = create_conversion_runtime(device, motion_fps, num_agents=num_agents)
            runtimes[motion_fps] = runtime
            print(f"[INFO]: Initialized conversion runtime for fps={motion_fps} with num_agents={num_agents}.")
        return runtime

    def close() -> None:
        for runtime in runtimes.values():
            runtime.sim.clear_instance()
        runtimes.clear()

    return resolve, close


def _record_failure(
    summary: BatchConversionSummary,
    input_file: Path,
    output_file: Path,
    error: str,
    *,
    prefix: str = "Failed to convert",
) -> None:
    summary.failed += 1
    failure = BatchFailure(input_file=input_file, output_file=output_file, error=error)
    summary.failures.append(failure)
    print(f"[ERROR]: {prefix} {input_file} -> {output_file}: {error}")


def prepare_batch_jobs(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fps_for_input: Callable[[Path], int] = peek_motion_fps,
) -> tuple[BatchConversionSummary, list[PendingBatchJob]]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    motion_files = discover_motion_files(input_path)
    summary = BatchConversionSummary(discovered=len(motion_files))
    pending_jobs: list[PendingBatchJob] = []

    for input_file in motion_files:
        destination = map_output_file(input_file, input_path, output_path)
        if destination.exists():
            summary.skipped += 1
            print(f"[INFO]: Skipping existing output {destination}")
            continue

        try:
            pending_jobs.append(
                PendingBatchJob(
                    input_file=input_file,
                    output_file=destination,
                    fps=fps_for_input(input_file),
                )
            )
        except Exception as exc:
            _record_failure(
                summary,
                input_file,
                destination,
                str(exc),
                prefix="Failed to inspect",
            )

    return summary, pending_jobs


def group_jobs_by_fps(jobs: Iterable[PendingBatchJob]) -> list[list[PendingBatchJob]]:
    jobs_by_fps: dict[int, list[PendingBatchJob]] = {}
    for job in jobs:
        jobs_by_fps.setdefault(job.fps, []).append(job)
    return list(jobs_by_fps.values())


def chunk_jobs(jobs: Sequence[PendingBatchJob], chunk_size: int) -> Iterable[list[PendingBatchJob]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    for start_index in range(0, len(jobs), chunk_size):
        yield list(jobs[start_index : start_index + chunk_size])


def print_batch_summary(summary: BatchConversionSummary) -> None:
    print(
        "[INFO]: Batch conversion summary: "
        f"discovered={summary.discovered}, "
        f"converted={summary.converted}, "
        f"skipped={summary.skipped}, "
        f"failed={summary.failed}, "
        f"elapsed={summary.elapsed_seconds:.3f}s"
    )


def _record_chunk_failures(summary: BatchConversionSummary, failures: Sequence[ConversionFailure]) -> None:
    for failure in failures:
        _record_failure(
            summary,
            failure.input_file,
            failure.output_file,
            failure.error,
        )


def run_batch_conversion(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    num_agents: int,
    runtime_for_fps: Callable[[int], object],
    convert_files: Callable[..., Sequence[ConversionFailure]],
    fps_for_input: Callable[[Path], int] = peek_motion_fps,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> BatchConversionSummary:
    if num_agents < 1:
        raise ValueError("num_agents must be at least 1.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {output_path}")

    start_time = perf_counter()
    summary, pending_jobs = prepare_batch_jobs(input_path, output_path, fps_for_input=fps_for_input)

    for fps_group in group_jobs_by_fps(pending_jobs):
        for chunk in chunk_jobs(fps_group, num_agents):
            try:
                runtime = runtime_for_fps(chunk[0].fps)
                failures = list(
                    convert_files(
                        [(job.input_file, job.output_file) for job in chunk],
                        runtime=runtime,
                    )
                )
            except Exception as exc:
                for job in chunk:
                    _record_failure(summary, job.input_file, job.output_file, str(exc))
                continue

            summary.converted += len(chunk) - len(failures)
            _record_chunk_failures(summary, failures)

    summary.elapsed_seconds = perf_counter() - start_time
    print_batch_summary(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args_cli = build_parser().parse_args(argv)
    input_dir = Path(args_cli.input_dir)
    output_dir = Path(args_cli.output_dir)
    options = ConversionOptions.from_args(args_cli)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    runtime_for_fps, close_runtime = build_runtime_cache(str(options.device), args_cli.num_agents)
    convert_files = partial(convert_motion_files, simulation_app=simulation_app, options=options)

    try:
        summary = run_batch_conversion(
            input_dir,
            output_dir,
            num_agents=args_cli.num_agents,
            runtime_for_fps=runtime_for_fps,
            convert_files=convert_files,
        )
    finally:
        close_runtime()
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)

    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

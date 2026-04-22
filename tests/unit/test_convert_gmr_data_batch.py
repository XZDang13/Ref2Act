from pathlib import Path

import ref2act.cli.convert_batch as batch_mod
from ref2act.cli.convert import ConversionFailure, build_parser as build_single_parser
from ref2act.cli.convert_batch import (
    BatchConversionSummary,
    PendingBatchJob,
    build_parser as build_batch_parser,
    build_runtime_cache,
    discover_motion_files,
    map_output_file,
    prepare_batch_jobs,
    run_batch_conversion,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_discover_motion_files_recursively_finds_sorted_pkls(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _touch(input_dir / "b.pkl")
    _touch(input_dir / "nested" / "a.pkl")
    _touch(input_dir / "nested" / "ignore.txt")
    _touch(input_dir / "UPPER.PKL")

    motion_files = discover_motion_files(input_dir)

    assert motion_files == [
        input_dir / "b.pkl",
        input_dir / "nested" / "a.pkl",
    ]


def test_map_output_file_preserves_relative_tree_and_suffix(tmp_path: Path) -> None:
    input_dir = tmp_path / "mocap"
    output_dir = tmp_path / "converted"
    input_file = input_dir / "subdir" / "walk.pkl"

    output_file = map_output_file(input_file, input_dir, output_dir)

    assert output_file == output_dir / "subdir" / "walk.npz"


def test_batch_parser_includes_shared_conversion_defaults_and_num_agents() -> None:
    batch_args = build_batch_parser().parse_args(["--input_dir", "mocap", "--output_dir", "converted"])
    single_args = build_single_parser().parse_args(["--input_file", "motion.pkl"])

    assert batch_args.num_agents == 1
    assert batch_args.height_offset == single_args.height_offset
    assert batch_args.segment_bin_size == single_args.segment_bin_size
    assert batch_args.airborne_height_threshold == single_args.airborne_height_threshold
    assert batch_args.segment_method == single_args.segment_method
    assert batch_args.smooth_motion == single_args.smooth_motion
    assert batch_args.smoothing_profile == single_args.smoothing_profile
    assert batch_args.target_fps == single_args.target_fps
    assert not hasattr(single_args, "num_agents")


def test_batch_parser_accepts_segment_smoothing_and_num_agents_flags() -> None:
    args = build_batch_parser().parse_args(
        [
            "--input_dir",
            "mocap",
            "--output_dir",
            "converted",
            "--num-agents",
            "4",
            "--height_offset",
            "0.15",
            "--segment-bin-size",
            "0.45",
            "--airborne-height-threshold",
            "0.08",
            "--segment-method",
            "anchor",
            "--smooth-motion",
            "--smoothing-profile",
            "strong",
            "--target-fps",
            "100",
        ]
    )

    assert args.num_agents == 4
    assert args.height_offset == 0.15
    assert args.segment_bin_size == 0.45
    assert args.airborne_height_threshold == 0.08
    assert args.segment_method == "anchor"
    assert args.smooth_motion is True
    assert args.smoothing_profile == "strong"
    assert args.target_fps == 100


def test_prepare_batch_jobs_skips_existing_outputs_before_scheduling(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    first_input = input_dir / "a.pkl"
    skipped_input = input_dir / "nested" / "b.pkl"
    _touch(first_input)
    _touch(skipped_input)
    _touch(map_output_file(skipped_input, input_dir, output_dir))

    fps_by_input = {
        first_input: 60,
    }

    summary, pending_jobs = prepare_batch_jobs(
        input_dir,
        output_dir,
        fps_for_input=lambda path: fps_by_input[path],
    )

    assert summary == BatchConversionSummary(discovered=2, skipped=1)
    assert pending_jobs == [
        PendingBatchJob(
            input_file=first_input,
            output_file=map_output_file(first_input, input_dir, output_dir),
            fps=60,
        )
    ]

    captured = capsys.readouterr()
    assert "Skipping existing output" in captured.out


def test_run_batch_conversion_groups_jobs_by_fps_chunks_and_continues_after_failures(
    tmp_path: Path,
    capsys,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    inputs = {
        name: input_dir / f"{name}.pkl"
        for name in ("a", "b", "c", "d", "e")
    }
    for path in inputs.values():
        _touch(path)

    _touch(map_output_file(inputs["e"], input_dir, output_dir))

    fps_by_input = {
        inputs["a"]: 60,
        inputs["b"]: 30,
        inputs["c"]: 60,
        inputs["d"]: 60,
        inputs["e"]: 30,
    }
    runtime_calls: list[int] = []
    chunk_calls: list[tuple[object, list[str]]] = []

    def runtime_for_fps(fps: int):
        runtime_calls.append(fps)
        return f"runtime-{fps}"

    def convert_files(motion_files, *, runtime):
        chunk_calls.append((runtime, [Path(input_file).name for input_file, _ in motion_files]))
        failing_pair = next(
            ((Path(input_file), Path(output_file)) for input_file, output_file in motion_files if Path(input_file).name == "d.pkl"),
            None,
        )
        if failing_pair is not None:
            return [
                ConversionFailure(
                    input_file=failing_pair[0],
                    output_file=failing_pair[1],
                    error="boom",
                )
            ]
        return []

    perf_samples = iter((100.0, 104.25))
    summary = run_batch_conversion(
        input_dir,
        output_dir,
        num_agents=2,
        runtime_for_fps=runtime_for_fps,
        convert_files=convert_files,
        fps_for_input=lambda path: fps_by_input[path],
        perf_counter=lambda: next(perf_samples),
    )

    assert summary == BatchConversionSummary(
        discovered=5,
        converted=3,
        skipped=1,
        failed=1,
        elapsed_seconds=4.25,
        failures=[summary.failures[0]],
    )
    assert summary.failures[0].input_file == inputs["d"]
    assert summary.failures[0].output_file == map_output_file(inputs["d"], input_dir, output_dir)
    assert summary.failures[0].error == "boom"
    assert runtime_calls == [60, 60, 30]
    assert chunk_calls == [
        ("runtime-60", ["a.pkl", "c.pkl"]),
        ("runtime-60", ["d.pkl"]),
        ("runtime-30", ["b.pkl"]),
    ]

    captured = capsys.readouterr()
    assert "Skipping existing output" in captured.out
    assert "Failed to convert" in captured.out
    assert "elapsed=4.250s" in captured.out


def test_build_runtime_cache_reuses_runtime_by_fps_and_closes_all(monkeypatch) -> None:
    created: list[tuple[str, int, int]] = []
    cleared: list[int] = []

    class DummySim:
        def __init__(self, fps: int) -> None:
            self.fps = fps

        def clear_instance(self) -> None:
            cleared.append(self.fps)

    class DummyRuntime:
        def __init__(self, fps: int) -> None:
            self.sim = DummySim(fps)

    def fake_create_conversion_runtime(device: str, fps: int, num_agents: int):
        created.append((device, fps, num_agents))
        return DummyRuntime(fps)

    monkeypatch.setattr(batch_mod, "create_conversion_runtime", fake_create_conversion_runtime)

    runtime_for_fps, close_runtime = build_runtime_cache("cpu", 3)

    runtime_a = runtime_for_fps(60)
    runtime_b = runtime_for_fps(60)
    runtime_c = runtime_for_fps(30)
    close_runtime()

    assert runtime_a is runtime_b
    assert runtime_a is not runtime_c
    assert created == [("cpu", 60, 3), ("cpu", 30, 3)]
    assert cleared == [60, 30]


def test_batch_summary_exit_code_is_zero_when_no_failures() -> None:
    summary = BatchConversionSummary(discovered=2, converted=2, skipped=0, failed=0)

    assert summary.exit_code == 0


def test_main_uses_multi_agent_batch_converter_with_shared_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyApp:
        def close(self, **kwargs) -> None:
            captured["app_close_kwargs"] = kwargs

    class DummyLauncher:
        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")

        def __init__(self, args) -> None:
            captured["launcher_args"] = args
            self.app = DummyApp()

    def fake_build_runtime_cache(device: str, num_agents: int):
        captured["resolver_device"] = device
        captured["resolver_num_agents"] = num_agents

        def runtime_for_fps(fps: int):
            captured["runtime_fps"] = fps
            return object()

        def close_runtime() -> None:
            captured["runtime_closed"] = True

        return runtime_for_fps, close_runtime

    def fake_run_batch_conversion(input_dir, output_dir, *, num_agents, runtime_for_fps, convert_files):
        captured["input_dir"] = Path(input_dir)
        captured["output_dir"] = Path(output_dir)
        captured["num_agents"] = num_agents
        captured["convert_files_func"] = convert_files.func
        captured["convert_options"] = convert_files.keywords["options"]
        captured["simulation_app"] = convert_files.keywords["simulation_app"]
        runtime_for_fps(30)
        return BatchConversionSummary(discovered=1, converted=1, skipped=0, failed=0)

    monkeypatch.setattr(batch_mod, "AppLauncher", DummyLauncher)
    monkeypatch.setattr(batch_mod, "build_runtime_cache", fake_build_runtime_cache)
    monkeypatch.setattr(batch_mod, "run_batch_conversion", fake_run_batch_conversion)

    exit_code = batch_mod.main(
        [
            "--input_dir",
            "mocap",
            "--output_dir",
            "converted",
            "--num-agents",
            "4",
            "--height_offset",
            "0.2",
            "--segment-bin-size",
            "0.5",
            "--airborne-height-threshold",
            "0.09",
            "--segment-method",
            "anchor",
            "--smooth-motion",
            "--smoothing-profile",
            "medium",
            "--target-fps",
            "50",
        ]
    )

    options = captured["convert_options"]
    assert exit_code == 0
    assert captured["convert_files_func"] is batch_mod.convert_motion_files
    assert captured["num_agents"] == 4
    assert options.height_offset == 0.2
    assert options.segment_bin_size == 0.5
    assert options.airborne_height_threshold == 0.09
    assert options.segment_method == "anchor"
    assert options.smooth_motion is True
    assert options.smoothing_profile == "medium"
    assert options.target_fps == 50
    assert captured["resolver_device"] == options.device
    assert captured["resolver_num_agents"] == 4
    assert captured["runtime_fps"] == 30
    assert captured["runtime_closed"] is True
    assert captured["app_close_kwargs"] == {"wait_for_replicator": False, "skip_cleanup": True}

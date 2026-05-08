import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_SHARED = REPO_ROOT / "src" / "ref2act" / "robots" / "_env_cfg_shared.py"
MOTION_TRACKING_ENV = REPO_ROOT / "src" / "ref2act" / "envs" / "motion_tracking" / "env.py"


def _find_class(tree: ast.AST, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Class {class_name!r} not found.")


def _find_class_assignment(class_def: ast.ClassDef, target_name: str) -> ast.AST:
    for node in class_def.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == target_name:
                    return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == target_name:
            return node.value
    raise AssertionError(f"Assignment {target_name!r} not found in class {class_def.name!r}.")


def _motion_sampler_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MotionSampler"
    ]


def _assert_sampler_keyword_threads_cfg_attr(tree: ast.AST, keyword_name: str) -> None:
    sampler_calls = _motion_sampler_calls(tree)
    assert sampler_calls, "Expected MotionSampler(...) call in motion_tracking/env.py."

    matching_keywords = [
        keyword
        for call in sampler_calls
        for keyword in call.keywords
        if keyword.arg == keyword_name
    ]
    assert matching_keywords, f"Expected MotionSampler(...) to receive {keyword_name}=..."

    value = matching_keywords[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == keyword_name
    assert isinstance(value.value, ast.Attribute)
    assert value.value.attr == "cfg"
    assert isinstance(value.value.value, ast.Name)
    assert value.value.value.id == "self"


def test_shared_env_cfg_defaults_segment_source_to_time() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())

    for class_name in ("G1MotionTrackingEnvCfg",):
        class_def = _find_class(tree, class_name)
        value = _find_class_assignment(class_def, "segment_source")
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name)
        assert value.value.id == "SegmentSource"
        assert value.attr == "Time"


def test_shared_env_cfg_defaults_mosaic_sampler_knobs() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())
    expected_defaults = {
        "weight_fail": 0.5,
        "weight_novel": 0.3,
        "cap_beta": 2.0,
        "adaptive_uniform_ratio": 0.1,
        "adaptive_alpha": 0.001,
        "adaptive_kernel_size": 1,
        "adaptive_lambda": 0.8,
        "motion_sampling_warmup_s": 0.0,
        "motion_sampling_ramp_s": 0.0,
        "motion_sampling_schedule": "cosine",
    }

    for class_name in ("G1MotionTrackingEnvCfg",):
        class_def = _find_class(tree, class_name)
        for name, expected in expected_defaults.items():
            value = _find_class_assignment(class_def, name)
            assert isinstance(value, ast.Constant)
            assert value.value == expected


def test_shared_env_cfg_removes_old_failure_weight_knobs() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())
    removed_names = {
        "failure_decay",
        "failure_weight_uniform_mix",
        "failure_weight_max_uniform_ratio",
        "failure_weight_exploration_bonus",
        "failure_temperature",
    }

    for class_name in ("G1MotionTrackingEnvCfg",):
        class_def = _find_class(tree, class_name)
        assigned_names: set[str] = set()
        for node in class_def.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigned_names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assigned_names.add(node.target.id)
        assert assigned_names.isdisjoint(removed_names)


def test_motion_tracking_env_threads_segment_source_into_sampler() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())

    _assert_sampler_keyword_threads_cfg_attr(tree, "segment_source")


def test_motion_tracking_env_threads_mosaic_sampler_knobs_into_sampler() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())
    for keyword_name in (
        "weight_fail",
        "weight_novel",
        "cap_beta",
        "adaptive_uniform_ratio",
        "adaptive_alpha",
        "adaptive_kernel_size",
        "adaptive_lambda",
        "motion_sampling_warmup_s",
        "motion_sampling_ramp_s",
        "motion_sampling_schedule",
    ):
        _assert_sampler_keyword_threads_cfg_attr(tree, keyword_name)


def test_motion_tracking_env_removes_old_failure_weight_sampler_keywords() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())
    removed_names = {
        "failure_decay",
        "failure_weight_uniform_mix",
        "failure_weight_max_uniform_ratio",
        "failure_weight_exploration_bonus",
        "failure_temperature",
    }
    sampler_keyword_names = {
        keyword.arg
        for call in _motion_sampler_calls(tree)
        for keyword in call.keywords
        if keyword.arg is not None
    }
    assert sampler_keyword_names.isdisjoint(removed_names)

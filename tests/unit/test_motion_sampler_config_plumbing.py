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


def test_shared_env_cfg_defaults_segment_source_to_time() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())

    for class_name in ("G1MotionTrackingEnvCfg", "PiPlusMotionTrackingEnvCfg"):
        class_def = _find_class(tree, class_name)
        value = _find_class_assignment(class_def, "segment_source")
        assert isinstance(value, ast.Attribute)
        assert isinstance(value.value, ast.Name)
        assert value.value.id == "SegmentSource"
        assert value.attr == "Time"


def test_shared_env_cfg_defaults_failure_weight_uniform_mix() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())

    for class_name in ("G1MotionTrackingEnvCfg", "PiPlusMotionTrackingEnvCfg"):
        class_def = _find_class(tree, class_name)
        value = _find_class_assignment(class_def, "failure_weight_uniform_mix")
        assert isinstance(value, ast.Constant)
        assert value.value == 0.1


def test_shared_env_cfg_defaults_failure_weight_max_uniform_ratio() -> None:
    tree = ast.parse(ENV_CFG_SHARED.read_text())

    for class_name in ("G1MotionTrackingEnvCfg", "PiPlusMotionTrackingEnvCfg"):
        class_def = _find_class(tree, class_name)
        value = _find_class_assignment(class_def, "failure_weight_max_uniform_ratio")
        assert isinstance(value, ast.Constant)
        assert value.value == 2.5


def test_motion_tracking_env_threads_segment_source_into_sampler() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())

    sampler_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MotionSampler"
    ]
    assert sampler_calls, "Expected MotionSampler(...) call in motion_tracking/env.py."

    segment_source_keywords = [
        keyword
        for call in sampler_calls
        for keyword in call.keywords
        if keyword.arg == "segment_source"
    ]
    assert segment_source_keywords, "Expected MotionSampler(...) to receive segment_source=..."

    value = segment_source_keywords[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "segment_source"
    assert isinstance(value.value, ast.Attribute)
    assert value.value.attr == "cfg"
    assert isinstance(value.value.value, ast.Name)
    assert value.value.value.id == "self"


def test_motion_tracking_env_threads_failure_weight_uniform_mix_into_sampler() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())

    sampler_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MotionSampler"
    ]
    assert sampler_calls, "Expected MotionSampler(...) call in motion_tracking/env.py."

    uniform_mix_keywords = [
        keyword
        for call in sampler_calls
        for keyword in call.keywords
        if keyword.arg == "failure_weight_uniform_mix"
    ]
    assert uniform_mix_keywords, "Expected MotionSampler(...) to receive failure_weight_uniform_mix=..."

    value = uniform_mix_keywords[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "failure_weight_uniform_mix"
    assert isinstance(value.value, ast.Attribute)
    assert value.value.attr == "cfg"
    assert isinstance(value.value.value, ast.Name)
    assert value.value.value.id == "self"


def test_motion_tracking_env_threads_failure_weight_max_uniform_ratio_into_sampler() -> None:
    tree = ast.parse(MOTION_TRACKING_ENV.read_text())

    sampler_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "MotionSampler"
    ]
    assert sampler_calls, "Expected MotionSampler(...) call in motion_tracking/env.py."

    max_uniform_ratio_keywords = [
        keyword
        for call in sampler_calls
        for keyword in call.keywords
        if keyword.arg == "failure_weight_max_uniform_ratio"
    ]
    assert (
        max_uniform_ratio_keywords
    ), "Expected MotionSampler(...) to receive failure_weight_max_uniform_ratio=..."

    value = max_uniform_ratio_keywords[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "failure_weight_max_uniform_ratio"
    assert isinstance(value.value, ast.Attribute)
    assert value.value.attr == "cfg"
    assert isinstance(value.value.value, ast.Name)
    assert value.value.value.id == "self"

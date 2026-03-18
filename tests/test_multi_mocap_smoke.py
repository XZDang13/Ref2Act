import os
from pathlib import Path
import sys
import types

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_isaaclab_stubs() -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    scene_module = types.ModuleType("isaaclab.scene")
    assets_module = types.ModuleType("isaaclab.assets")
    utils_module = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")

    class InteractiveScene:  # pragma: no cover - stub for import only
        pass

    class Articulation:  # pragma: no cover - stub for import only
        pass

    def _not_used(*args, **kwargs):
        raise RuntimeError("IsaacLab math stubs should not be used in this smoke test.")

    scene_module.InteractiveScene = InteractiveScene
    assets_module.Articulation = Articulation
    math_module.quat_mul = _not_used
    math_module.quat_inv = _not_used
    math_module.quat_apply = _not_used
    math_module.yaw_quat = _not_used
    math_module.quat_from_euler_xyz = _not_used
    utils_module.math = math_module
    isaaclab_module.scene = scene_module
    isaaclab_module.assets = assets_module
    isaaclab_module.utils = utils_module

    sys.modules.setdefault("isaaclab", isaaclab_module)
    sys.modules.setdefault("isaaclab.scene", scene_module)
    sys.modules.setdefault("isaaclab.assets", assets_module)
    sys.modules.setdefault("isaaclab.utils", utils_module)
    sys.modules.setdefault("isaaclab.utils.math", math_module)


_install_isaaclab_stubs()

from Ref2Act.motion_lib import MotionLib
from Ref2Act.sampler import Sampler, SamplingStrategy


ASSET_DIR = REPO_ROOT / "tests" / "assests"


def test_multi_mocap_sampler_smoke() -> None:
    motion_files = sorted(ASSET_DIR.glob("*.npz"))
    assert motion_files, f"No mocap files found in {ASSET_DIR}"

    torch.manual_seed(0)

    motion_lib = MotionLib(motion_files)
    assert motion_lib.num_motions == len(motion_files)

    motion_ids = torch.arange(motion_lib.num_motions, dtype=torch.long)
    start_times = torch.zeros(motion_ids.shape, dtype=torch.float32)
    start_samples = motion_lib.sample_motion(motion_ids=motion_ids, times=start_times)

    assert start_samples["joint_pos"].shape[0] == len(motion_files)
    for motion_id in motion_ids.tolist():
        clip = motion_lib.get_clip(motion_id)
        assert torch.allclose(start_samples["joint_pos"][motion_id], clip.joint_pos[0])

    sampler = Sampler(
        num_envs=motion_lib.num_motions,
        motion_lib=motion_lib,
        dt=1.0 / 60.0,
        anchor_body_index=0,
        bin_size=0.25,
    )
    env_ids = torch.arange(motion_lib.num_motions, dtype=torch.long)

    sampled_motion_ids = sampler.sample_motion_ids(env_ids)
    assert sampled_motion_ids.shape == env_ids.shape
    assert torch.all((sampled_motion_ids >= 0) & (sampled_motion_ids < motion_lib.num_motions))

    random_times = sampler.sample_times_for_motion_ids(motion_ids, SamplingStrategy.Random)
    durations = motion_lib.get_duration(motion_ids)
    assert torch.all(random_times >= 0.0)
    assert torch.all(random_times <= durations)

    sampler._apply_reset_state(env_ids, motion_ids, random_times)
    assert torch.equal(sampler.current_motion_ids, motion_ids)
    assert torch.allclose(sampler.current_times, random_times)
    assert torch.allclose(sampler.get_current_durations(), durations)

    stepped_reference = motion_lib.sample_motion(
        motion_ids=sampler.current_motion_ids,
        times=sampler.current_times,
    )
    assert stepped_reference["body_positions"].shape[0] == motion_lib.num_motions

    previous_times = sampler.current_times.clone()
    sampler._sample_next_times(env_ids)
    assert torch.allclose(sampler.current_times, previous_times + sampler.dt)

    sampler.record_failures(env_ids[::2])
    assert sum(int(count.sum().item()) for count in sampler.bin_fail_counts) == len(env_ids[::2])

    failure_weighted_times = sampler.sample_times_for_motion_ids(
        motion_ids,
        SamplingStrategy.FailureWeighted,
    )
    assert torch.all(failure_weighted_times >= 0.0)
    assert torch.all(failure_weighted_times <= durations)

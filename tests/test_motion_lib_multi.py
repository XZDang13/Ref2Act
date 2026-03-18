from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ref2Act.motion_lib import MotionLib


TEST_DATA_DIR = REPO_ROOT / "test" / "data"


def test_motion_lib_supports_mixed_motion_batches() -> None:
    motion_lib = MotionLib(
        [
            TEST_DATA_DIR / "jab.npz",
            TEST_DATA_DIR / "pick.npz",
        ]
    )

    motion_ids = torch.tensor([0, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.0], dtype=torch.float32)
    samples = motion_lib.sample_motion(motion_ids=motion_ids, times=times)

    jab_clip = motion_lib.get_clip(0)
    pick_clip = motion_lib.get_clip(1)

    assert motion_lib.num_motions == 2
    assert motion_lib.motion_names == ["jab", "pick"]
    assert motion_lib.get_duration(motion_ids)[0] != motion_lib.get_duration(motion_ids)[1]
    assert torch.allclose(samples["joint_pos"][0], jab_clip.joint_pos[0])
    assert torch.allclose(samples["joint_pos"][1], pick_clip.joint_pos[0])


def test_motion_lib_applies_offsets_per_selected_motion() -> None:
    motion_lib = MotionLib(
        [
            TEST_DATA_DIR / "jab.npz",
            TEST_DATA_DIR / "pick.npz",
        ]
    )

    motion_ids = torch.tensor([0, 1], dtype=torch.long)
    times = torch.tensor([0.0, 0.0], dtype=torch.float32)
    position_offsets = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=torch.float32)
    samples = motion_lib.sample_motion(
        motion_ids=motion_ids,
        times=times,
        position_offsets=position_offsets,
    )

    jab_clip = motion_lib.get_clip(0)
    pick_clip = motion_lib.get_clip(1)

    assert torch.allclose(
        samples["body_positions"][0],
        jab_clip.body_positions[0] + position_offsets[0].view(1, 3),
    )
    assert torch.allclose(
        samples["body_positions"][1],
        pick_clip.body_positions[0] + position_offsets[1].view(1, 3),
    )

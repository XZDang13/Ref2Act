from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Ref2Act.buffer import DequeBuffer


def test_deque_buffer_append_preserves_history_order() -> None:
    buffer = DequeBuffer(batch_size=2, horizon=3, size=(1,))

    buffer.append(torch.tensor([[1.0], [10.0]]))
    buffer.append(torch.tensor([[2.0], [20.0]]))
    buffer.append(torch.tensor([[3.0], [30.0]]))

    assert torch.allclose(
        buffer.get_all().squeeze(-1),
        torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
    )

    buffer.append(torch.tensor([[4.0], [40.0]]))

    assert torch.allclose(
        buffer.get_all().squeeze(-1),
        torch.tensor([[2.0, 3.0, 4.0], [20.0, 30.0, 40.0]]),
    )

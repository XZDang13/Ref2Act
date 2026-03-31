from pathlib import Path

import torch

from ref2act.common.buffer import DequeBuffer


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

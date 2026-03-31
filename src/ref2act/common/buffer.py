import torch
from .utils import IndexLike


class DequeBuffer:
    """
    Batched fixed-length deque buffer (sliding window) for a single tensor item.

    - append(x): push newest step, drop oldest
    - reset(mask/indices, values): reset selected batch elements
    """

    def __init__(
        self,
        batch_size: int,
        horizon: int,
        size: tuple[int, ...],
        device: torch.device|None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.B = batch_size
        self.T = horizon
        self.size = size
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

        self.buffer = torch.zeros((self.B, self.T, *self.size), device=self.device, dtype=self.dtype)
        self.valid_len = torch.zeros((self.B,), device=self.device, dtype=torch.long)

    def _as_mask(self, which: IndexLike) -> torch.Tensor:
        """Convert indices or bool mask into a bool mask of shape [B]."""
        if isinstance(which, torch.Tensor) and which.dtype == torch.bool:
            assert which.shape == (self.B,), f"mask must be shape {(self.B,)}, got {tuple(which.shape)}"
            return which.to(self.device)

        if not isinstance(which, torch.Tensor):
            which = torch.tensor(list(which), device=self.device, dtype=torch.long)
        else:
            which = which.to(self.device, dtype=torch.long)

        mask = torch.zeros((self.B,), device=self.device, dtype=torch.bool)
        mask[which] = True
        return mask

    @torch.no_grad()
    def reset(
        self,
        which: IndexLike,
        values: torch.Tensor|None = None,
        fill_all: bool = True,
        zero: bool = True,
    ) -> None:
        """
        Reset a subset of batch elements.

        Args:
            which: indices or bool mask [B]
            values: optional tensor for the reset envs.
                - preferred: [N, *size] where N = num_reset
                - also allowed: [*size] or [1, *size] (broadcast to N)
            fill_all: if True, fill all T slots; else only fill the newest slot (-1)
            zero: if True, zero out the selected envs first
        """
        mask = self._as_mask(which)
        n = int(mask.sum().item())
        if n == 0:
            return

        if zero:
            self.buffer[mask] = 0
            self.valid_len[mask] = 0

        if values is None:
            return

        v = values.to(device=self.device, dtype=self.buffer.dtype)

        # Normalize v to shape [N, *size]
        if v.shape == self.size:
            v = v.unsqueeze(0).expand(n, *self.size)
        elif v.shape == (1, *self.size):
            v = v.expand(n, *self.size)
        else:
            assert v.shape == (n, *self.size), f"values must be {(n, *self.size)} or {self.size} or {(1, *self.size)}, got {tuple(v.shape)}"

        if fill_all:
            self.buffer[mask] = v.unsqueeze(1).expand(n, self.T, *self.size)
            self.valid_len[mask] = self.T
        else:
            self.buffer[mask, -1] = v
            self.valid_len[mask] = torch.maximum(self.valid_len[mask], torch.ones_like(self.valid_len[mask]))

    @torch.no_grad()
    def append(self, x: torch.Tensor, reset_mask: torch.Tensor|None = None) -> None:
        """
        Append one step for all batch elements.

        Args:
            x: [B, *size]
            reset_mask: optional bool mask [B], reset those envs BEFORE appending
        """
        if reset_mask is not None:
            self.reset(reset_mask)

        x = x.to(device=self.device, dtype=self.buffer.dtype)
        assert x.shape == (self.B, *self.size), f"x must be {(self.B, *self.size)}, got {tuple(x.shape)}"

        # Shift left (drop oldest), then write the newest step.
        # Clone the source view first so PyTorch does not reject the overlapping write.
        if self.T > 1:
            self.buffer[:, :-1].copy_(self.buffer[:, 1:].clone())
        self.buffer[:, -1].copy_(x)

        self.valid_len = torch.minimum(self.valid_len + 1, torch.tensor(self.T, device=self.device))

    def get(self, index: int|IndexLike) -> torch.Tensor:
        """
        Return selected step [B, *size] by time index (oldest → newest).

        - int: same index for all batch elements
        - IndexLike of shape [B]: per-batch indices
        """
        if isinstance(index, int):
            if index < -self.T or index >= self.T:
                raise IndexError(f"index {index} out of range for horizon {self.T}")
            return self.buffer[:, index]

        if not isinstance(index, torch.Tensor):
            index = torch.tensor(list(index), device=self.device, dtype=torch.long)
        else:
            index = index.to(self.device, dtype=torch.long)

        if index.ndim == 0:
            idx = int(index.item())
            if idx < -self.T or idx >= self.T:
                raise IndexError(f"index {idx} out of range for horizon {self.T}")
            return self.buffer[:, idx]

        assert index.shape == (self.B,), f"index must be shape {(self.B,)}, got {tuple(index.shape)}"
        index = torch.where(index < 0, index + self.T, index)
        if torch.any(index < 0) or torch.any(index >= self.T):
            raise IndexError(f"index out of range for horizon {self.T}")

        batch_idx = torch.arange(self.B, device=self.device)
        return self.buffer[batch_idx, index]

    def get_all(self) -> torch.Tensor:
        """Return history [B, T, *size] (oldest → newest)."""
        return self.buffer

    def latest(self) -> torch.Tensor:
        """Return newest step [B, *size]."""
        return self.buffer[:, -1]

    def __repr__(self) -> str:
        return f"DequeBuffer(B={self.B}, T={self.T}, size={self.size}, dtype={self.buffer.dtype}, device={self.device})"

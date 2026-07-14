from __future__ import annotations

import torch
import warp as wp


def to_torch(value):
    """Resolve an Isaac Lab 3 ProxyArray to its Torch view."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, wp.array):
        return torch.from_dlpack(wp.to_dlpack(value))
    torch_value = getattr(value, "torch", None)
    if torch_value is not None:
        torch_value = torch_value() if callable(torch_value) else torch_value
        if torch_value is not value:
            return to_torch(torch_value)
    try:
        converted = wp.to_torch(value)
        if isinstance(converted, torch.Tensor):
            return converted
        return torch.from_dlpack(wp.to_dlpack(converted))
    except (TypeError, RuntimeError):
        return value


__all__ = ["to_torch"]

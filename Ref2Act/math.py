import torch
from isaaclab.utils.math import subtract_frame_transforms

def relative_transform(
    anchor_pos: torch.Tensor,
    anchor_quat: torch.Tensor,
    key_pos: torch.Tensor,
    key_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B = anchor_pos.shape[0]
    K = key_pos.shape[1]

    # [B, 1, 3] → [B, K, 3]
    anchor_pos = anchor_pos.expand(-1, K, -1)
    # [B, 1, 4] → [B, K, 4]
    anchor_quat = anchor_quat.expand(-1, K, -1)

    key_pos = key_pos
    key_quat = key_quat

    pos, quat = subtract_frame_transforms(
        anchor_pos,
        anchor_quat,
        key_pos,
        key_quat,
    )

    return pos, quat

def exp_error(error: torch.Tensor, std:float):
    return torch.exp(-error / std**2)
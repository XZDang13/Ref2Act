from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SmoothingProfileSpec:
    window_seconds: float
    sigma_seconds: float


SMOOTHING_PROFILES: dict[str, SmoothingProfileSpec] = {
    "light": SmoothingProfileSpec(window_seconds=0.10, sigma_seconds=0.03),
    "medium": SmoothingProfileSpec(window_seconds=0.20, sigma_seconds=0.06),
    "strong": SmoothingProfileSpec(window_seconds=0.30, sigma_seconds=0.09),
}
DEFAULT_SMOOTHING_PROFILE = "light"


def smooth_motion_trajectory(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    joint_pos: torch.Tensor,
    fps: float,
    profile: str = DEFAULT_SMOOTHING_PROFILE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if fps <= 0.0:
        raise ValueError("fps must be > 0.")
    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError("root_pos must be shaped [num_frames, 3].")
    if root_rot.ndim != 2 or root_rot.shape[-1] != 4:
        raise ValueError("root_rot must be shaped [num_frames, 4].")
    if joint_pos.ndim != 2:
        raise ValueError("joint_pos must be shaped [num_frames, num_joints].")

    num_frames = int(root_pos.shape[0])
    if root_rot.shape[0] != num_frames or joint_pos.shape[0] != num_frames:
        raise ValueError("root_pos, root_rot, and joint_pos must have the same frame count.")

    window_size, sigma_frames = _resolve_smoothing_parameters(num_frames=num_frames, fps=fps, profile=profile)
    if window_size < 3:
        return root_pos.clone(), root_rot.clone(), joint_pos.clone()

    kernel = _build_gaussian_kernel(window_size, sigma_frames, device=root_pos.device, dtype=root_pos.dtype)
    smoothed_root_pos = _smooth_linear_signal(root_pos, kernel)
    smoothed_joint_pos = _smooth_linear_signal(joint_pos, kernel)
    smoothed_root_rot = _smooth_quaternion_signal(root_rot, kernel)
    return smoothed_root_pos, smoothed_root_rot, smoothed_joint_pos


def _resolve_smoothing_parameters(num_frames: int, fps: float, profile: str) -> tuple[int, float]:
    if num_frames < 1:
        return 0, 1.0

    spec = _get_profile_spec(profile)
    raw_window = max(3, round(spec.window_seconds * fps))
    if raw_window % 2 == 0:
        raw_window += 1

    max_window = num_frames if num_frames % 2 == 1 else num_frames - 1
    window_size = min(raw_window, max_window)
    sigma_frames = max(1.0, spec.sigma_seconds * fps)
    return window_size, sigma_frames


def _get_profile_spec(profile: str) -> SmoothingProfileSpec:
    try:
        return SMOOTHING_PROFILES[profile]
    except KeyError as exc:
        valid_profiles = ", ".join(sorted(SMOOTHING_PROFILES))
        raise ValueError(f"Unknown smoothing profile {profile!r}. Expected one of: {valid_profiles}.") from exc


def _build_gaussian_kernel(window_size: int, sigma_frames: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = (window_size - 1) / 2.0
    positions = torch.arange(window_size, device=device, dtype=dtype) - center
    kernel = torch.exp(-0.5 * torch.square(positions / sigma_frames))
    return kernel / kernel.sum()


def _smooth_linear_signal(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if data.shape[0] < 3:
        return data.clone()

    num_frames = data.shape[0]
    num_channels = data[0].numel()
    pad = kernel.shape[0] // 2

    values = data.reshape(num_frames, num_channels).transpose(0, 1).unsqueeze(0)
    values = F.pad(values, (pad, pad), mode="reflect")

    weights = kernel.to(device=data.device, dtype=data.dtype).view(1, 1, -1).repeat(num_channels, 1, 1)
    smoothed = F.conv1d(values, weights, groups=num_channels)
    return smoothed.squeeze(0).transpose(0, 1).reshape_as(data)


def _smooth_quaternion_signal(quaternions: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[0] < 3:
        return quaternions.clone()

    quaternions = _enforce_quaternion_continuity(_normalize_quaternions(quaternions))
    pad = kernel.shape[0] // 2
    padded_quaternions = _reflect_pad_time(quaternions, pad)
    weights = kernel.to(device=quaternions.device, dtype=quaternions.dtype)

    smoothed = torch.empty_like(quaternions)
    for frame_index in range(quaternions.shape[0]):
        window = padded_quaternions[frame_index : frame_index + kernel.shape[0]]
        center = quaternions[frame_index : frame_index + 1]
        aligned = _align_quaternions_to_reference(window, center)
        averaged = torch.sum(aligned * weights.unsqueeze(-1), dim=0)
        averaged_norm = torch.linalg.norm(averaged)
        if averaged_norm <= 1.0e-8:
            smoothed[frame_index] = center.squeeze(0)
        else:
            smoothed[frame_index] = averaged / averaged_norm

    return _enforce_quaternion_continuity(_normalize_quaternions(smoothed))


def _reflect_pad_time(data: torch.Tensor, pad: int) -> torch.Tensor:
    if pad == 0:
        return data.clone()

    num_frames = data.shape[0]
    num_channels = data[0].numel()
    values = data.reshape(num_frames, num_channels).transpose(0, 1).unsqueeze(0)
    values = F.pad(values, (pad, pad), mode="reflect")
    return values.squeeze(0).transpose(0, 1).reshape(num_frames + 2 * pad, *data.shape[1:])


def _normalize_quaternions(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return quaternions / norms


def _enforce_quaternion_continuity(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[0] <= 1:
        return quaternions.clone()

    continuous = quaternions.clone()
    dot_products = torch.sum(continuous[1:] * continuous[:-1], dim=-1)
    flip_signs = torch.ones(continuous.shape[0], device=continuous.device, dtype=continuous.dtype)
    flip_signs[1:] = torch.where(dot_products < 0.0, -1.0, 1.0)
    continuous *= torch.cumprod(flip_signs, dim=0).unsqueeze(-1)
    return continuous


def _align_quaternions_to_reference(quaternions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    alignment = torch.where(
        torch.sum(quaternions * reference, dim=-1, keepdim=True) < 0.0,
        -1.0,
        1.0,
    )
    return quaternions * alignment

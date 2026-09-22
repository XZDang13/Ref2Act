"""Task reset joint targets, independent of robot defaults and action offsets."""
import math
import torch


def validate_reset_pose(initial):
    pose = initial.get('joint_positions')
    if pose is None:
        return None
    if initial.get('mode') != 'fixed_supine':
        raise ValueError('Explicit reset joint_positions currently requires fixed_supine')
    if not isinstance(pose, dict) or not pose:
        raise ValueError('Reset joint_positions must be a nonempty named mapping')
    if any(not isinstance(k, str) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for k, v in pose.items()):
        raise ValueError('Reset joint_positions requires finite angles in radians')
    return dict(pose)


def reset_joint_targets(defaults, joint_names, limits, pose):
    target = defaults.clone()
    if pose is None:
        return target
    unknown = set(pose) - set(joint_names)
    if unknown:
        raise ValueError(f'Unknown reset joint names: {sorted(unknown)}')
    for name, value in pose.items():
        target[:, joint_names.index(name)] = value
    if not torch.isfinite(target).all() or ((target < limits[..., 0]) | (target > limits[..., 1])).any():
        raise ValueError('Explicit reset pose exceeds live joint limits')
    return target


def target_to_offset_action(target, offset, scale, noise):
    return (target - offset - noise) / scale

import torch

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .motion_lib import ReferenceMotions

POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

JOINT_POSITION_RANGE = (-0.1, 0.1)

def pose_noise(size:int, noise_ranges:dict[str, tuple[float, float]], device:torch.device):
    position_noise = []
    for key in ["x", "y", "z"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        position_noise.append(noise)

    euler_noise = []
    for key in ["roll", "pitch", "yaw"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, device=device).uniform_(low_range, up_range)
        euler_noise.append(noise)

    position_noise = torch.cat(position_noise, dim=-1)
    quaterions_noise = quat_from_euler_xyz(*euler_noise)

    return position_noise, quaterions_noise

def velocity_noise(size:int, noise_ranges:dict[str, tuple[float, float]], device:torch.device):
    linear_vel_noise = []
    for key in ["x", "y", "z"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        linear_vel_noise.append(noise)

    ang_vel_noise = []
    for key in ["roll", "pitch", "yaw"]:
        (low_range, up_range) = noise_ranges[key]
        noise = torch.empty(size, 1, device=device).uniform_(low_range, up_range)
        ang_vel_noise.append(noise)

    linear_vel_noise = torch.cat(linear_vel_noise, dim=-1)
    ang_vel_noise = torch.cat(ang_vel_noise, dim=-1)

    return linear_vel_noise, ang_vel_noise

class InitialSetting:
    pose_range = POSE_RANGE
    velcoity_range = VELOCITY_RANGE
    joint_position_range = JOINT_POSITION_RANGE

    @staticmethod
    def set_robot_initial_state(
        robot: Articulation,
        env_ids: torch.Tensor,
        motion_samples: ReferenceMotions,
        root_index: int,
        add_noise:bool
    ) -> None:
        joint_pos = motion_samples.joint_pos
        joint_vel = motion_samples.joint_vel
        root_pos = motion_samples.body_positions[:, root_index]
        root_quat = motion_samples.body_quaternions[:, root_index]
        root_linear_vel = motion_samples.body_linear_velocities[:, root_index]
        root_angular_vel = motion_samples.body_angular_velocities[:, root_index]

        root_state = robot.data.default_root_state[env_ids].clone()

        if add_noise:
            device = root_pos.device
            root_pos_noise, root_quat_noise = pose_noise(len(env_ids), InitialSetting.pose_range, device)
            root_linear_vel_noise, root_angular_vel_noise = velocity_noise(len(env_ids), InitialSetting.velcoity_range, device)
            joint_pose_noise = torch.empty_like(joint_pos).uniform_(InitialSetting.joint_position_range[0],
                                                                    InitialSetting.joint_position_range[1])

            root_pos += root_pos_noise
            root_quat = quat_mul(root_quat, root_quat_noise)
            root_linear_vel += root_linear_vel_noise
            root_angular_vel += root_angular_vel_noise
            joint_pos += joint_pose_noise

        root_state[:, 0:3] = root_pos
        #root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_linear_vel
        root_state[:, 10:13] = root_angular_vel

        robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
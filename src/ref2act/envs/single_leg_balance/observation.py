"""312-D robot-only history, always using post-delay normalized applied actions."""
import torch
from ref2act.common.math import quat_apply_inverse, quaternion_to_rotation_6d
from ref2act.isaac_compat import to_torch
from .logic import ACTION_HISTORY_CONTRACT


class RobotHistory:
    contract=ACTION_HISTORY_CONTRACT
    def __init__(self):
        self.proprio=self.actions=None

    def update(self,frame,action,reset_mask):
        if frame.shape[-1]!=55 or action.shape[-1]!=23:
            raise ValueError('Expected G1 55-D proprioception and 23-D applied action')
        if self.proprio is None:
            self.proprio=frame[:,None].repeat(1,4,1)
            self.actions=action[:,None].repeat(1,4,1)
        else:
            self.proprio=torch.roll(self.proprio,-1,1)
            self.actions=torch.roll(self.actions,-1,1)
            self.proprio[:,-1]=frame;self.actions[:,-1]=action
            self.proprio[reset_mask]=frame[reset_mask,None]
            self.actions[reset_mask]=action[reset_mask,None]
        value=torch.cat((self.proprio.flatten(1),self.actions.flatten(1)),-1)
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite single-leg actor observation')
        return value


def proprioception(env):
    d=env.robot.data
    quat=to_torch(d.body_link_quat_w)[:,env.anchor_body_index]
    ang=quat_apply_inverse(quat,to_torch(d.body_link_ang_vel_w)[:,env.anchor_body_index])
    return torch.cat((quaternion_to_rotation_6d(quat),ang,
                      env._sim_to_policy_order(to_torch(d.joint_pos)),
                      env._sim_to_policy_order(to_torch(d.joint_vel))),-1).float()


def noisy_frame(frame,cfg):
    if not cfg['enabled']: return frame.clone()
    scale=frame.new_tensor([cfg['orientation_6d']]*6+[cfg['angular_velocity']]*3+
                           [cfg['joint_position']]*23+[cfg['joint_velocity']]*23)
    return frame+(torch.rand_like(frame)*2-1)*scale

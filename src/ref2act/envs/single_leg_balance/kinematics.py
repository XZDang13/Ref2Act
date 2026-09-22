"""Batched G1 left-sole FK for floor-aligned resets; xyzw quaternion convention.

Fixed transforms are from the packaged g1_23dof_rubber_hand MJCF. The unit
suite compares this calculation with MuJoCo under perturbed joint/root poses.
"""
import torch
from ref2act.common.math import quat_apply, quat_mul


def left_sole_min_height(q,names,root_quat,root_height):
    pos=torch.zeros(q.shape[0],3,device=q.device,dtype=q.dtype)
    pos[:,2]=root_height
    rot=root_quat.clone()
    chain=[('left_hip_pitch_joint',[0,.064452,-.1027],[0,0,0,1],[0,1,0]),
           ('left_hip_roll_joint',[0,.052,-.030465],[0,-.0873386,0,.996179],[1,0,0]),
           ('left_hip_yaw_joint',[.025001,0,-.12412],[0,0,0,1],[0,0,1]),
           ('left_knee_joint',[-.078273,.0021489,-.17734],[0,.0873386,0,.996179],[0,1,0]),
           ('left_ankle_pitch_joint',[0,-.000094445,-.30001],[0,0,0,1],[0,1,0]),
           ('left_ankle_roll_joint',[0,0,-.017558],[0,0,0,1],[1,0,0])]
    for name,translation,fixed,axis in chain:
        pos=pos+quat_apply(rot,q.new_tensor(translation).expand(q.shape[0],-1))
        fixed=q.new_tensor(fixed);fixed=fixed/fixed.norm()
        rot=quat_mul(rot,fixed.expand(q.shape[0],-1))
        half=q[:,names.index(name)]*.5
        joint=torch.cat((half.sin()[:,None]*q.new_tensor(axis),half.cos()[:,None]),-1)
        rot=quat_mul(rot,joint)
    points=q.new_tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
    world=quat_apply(rot[:,None].expand(-1,4,-1).reshape(-1,4),points[None].expand(q.shape[0],-1,-1).reshape(-1,3))
    return (world.reshape(-1,4,3)[...,2]+pos[:,None,2]-.005).amin(-1)

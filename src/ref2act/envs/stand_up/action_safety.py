"""Explicit target guards for zero-delay offset control; not hardware certification."""
import math
import torch


def validate_safety(settings):
    if not settings or not settings.get('enabled', False):
        return None
    keys=('joint_margin_rad','leg_target_speed','arm_target_speed','waist_target_speed',
          'ankle_roll_target_speed','intervention_weight')
    for key in keys:
        value=settings.get(key)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError(f'Invalid action_safety {key}')
    return dict(settings)


class TargetGuard:
    def __init__(self,names,low,high,settings,dt):
        self.low=low+settings['joint_margin_rad']
        self.high=high-settings['joint_margin_rad']
        if (self.low>=self.high).any():
            raise ValueError('Action safety margins leave an empty joint range')
        speeds=[]
        for name in names:
            if name=='waist_yaw_joint':key='waist_target_speed'
            elif name.endswith('ankle_roll_joint'):key='ankle_roll_target_speed'
            elif any(part in name for part in ('hip_','knee_','ankle_')):key='leg_target_speed'
            elif any(part in name for part in ('shoulder_','elbow_','wrist_')):key='arm_target_speed'
            else:raise ValueError(f'Unknown guarded joint {name}')
            speeds.append(settings[key])
        self.step=low.new_tensor(speeds)*dt

    def validate_reset(self,target):
        if not torch.isfinite(target).all() or ((target<self.low)|(target>self.high)).any():
            raise ValueError('Reset target lies outside action safety soft bounds')

    def apply(self,requested,previous):
        # Reset is validated separately, so the rate/position intervals always intersect.
        finite=torch.isfinite(requested).all(-1)
        clean=torch.where(finite[:,None],requested,previous)
        lower=torch.maximum(self.low,previous-self.step)
        upper=torch.minimum(self.high,previous+self.step)
        safe=torch.maximum(lower,torch.minimum(upper,clean))
        return safe,finite

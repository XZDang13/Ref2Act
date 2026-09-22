"""Versioned benchmark settings. All distances are metres, times seconds."""
from copy import deepcopy
from .pose import RESET_POSE


def default_config():
    return dict(
        version=3, initial=deepcopy(RESET_POSE),
        reset_noise=dict(joint_position=.04, joint_velocity=.20, roll_pitch=.06981317,
                         linear_velocity=.08, angular_velocity=.15),
        actor_noise=dict(enabled=True, orientation_6d=.05, angular_velocity=.2,
                         joint_position=.01, joint_velocity=.5),
        push=dict(time_range=[1.,4.5], delta_v_range=[0.,.30], duration_s=.10,
                  body='torso_link', evaluation_delta_v=None, evaluation_direction=None),
        contact=dict(threshold_n=1., support_load=.20, grace_s=.18, bad_grace_s=.06),
        reward=dict(balance_sigma=.08, swing_range=[.08,.18], swing_sigma=.05,
                    height_sigma=.10, pose_sigma=.6, velocity_scale=5.,
                    action_rate_scale=1., penalty_clip=False, weights=dict(balance=3.,upright=1.5,swing=1.,
                    height=.5,pose=.25,velocity=-.20,action_rate=-.20,bad_contact=-.5,
                    joint_limit=-10.,joint_acc=-5e-7,torque=-1e-5,target_clip=-1.,target_rate=-.1,target_acc=-.1)),
        termination=dict(height_ratio=.625,upright=.5),
        recovery=dict(balance_error=.05,com_speed=.15,hold_s=.30),
    )

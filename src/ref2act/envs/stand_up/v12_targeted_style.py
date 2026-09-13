"""Targeted V12 waist yaw and hand load imbalance costs."""
import math
import torch
from ref2act.isaac_compat import to_torch


def validate_targeted_style(c):
    keys = {'waist_weight', 'waist_tolerance', 'hand_weight', 'hand_tolerance', 'hand_epsilon'}
    if 'pose' in c:
        pc = c['pose']
        if set(pc) != {'waist_weight','waist_tolerance','arm_weight','arm_tolerance','arm_unload_scale','leg_weight'} or any(not math.isfinite(float(v)) or v < 0 for v in pc.values()) or pc['arm_unload_scale'] <= 0:
            raise ValueError('Invalid unified pose settings')
        if any(k in c for k in ('waist_weight','waist_tolerance','arm_weight')) or c.get('hip_yaw_weight',0) != 0:
            raise ValueError('Unified pose replaces separate waist, arm and hip yaw pose costs')
        c = {k:v for k,v in c.items() if k != 'pose'}
        c.update(waist_weight=0.,waist_tolerance=pc['waist_tolerance'])
    if 'hip_yaw_weight' in c:
        keys |= {'hip_yaw_weight','hip_yaw_tolerance','foot_heading_weight','foot_heading_tolerance','waist_speed_weight','waist_speed_tolerance'}
    if 'arm_weight' in c:
        keys |= {'arm_weight', 'arm_tolerance', 'arm_unload_scale'}
        if c.get('arm_unload_scale', 0) <= 0:
            raise ValueError('Arm unload scale must be positive')
    if set(c) != keys or any(not math.isfinite(float(v)) or v < 0 for v in c.values()) or c['hand_epsilon'] <= 0:
        raise ValueError('Invalid V12 targeted style settings')


def targeted_style(waist, default, hands, c, dt):
    waist_error = (waist - default).abs()
    waist_cost = (waist_error - c['waist_tolerance']).clamp_min(0).square()
    loads = hands.clamp_min(0)
    total = loads.sum(-1)
    difference = (loads[:, 0] - loads[:, 1]).abs()
    # Ignore small sensor/load differences. Equal loads and complete unloading
    # both have zero cost; there is no bonus for increasing total hand force.
    hand_cost = (difference - c['hand_tolerance']).clamp_min(0).square() / (total + c['hand_epsilon'])
    terms = dict(waist_yaw=-dt*c['waist_weight']*waist_cost,
                 hand_imbalance=-dt*c['hand_weight']*hand_cost)
    return terms, dict(waist_error_rad=waist_error, waist_cost=waist_cost,
                      hand_load_difference=difference, hand_total_load=total,
                      hand_imbalance_cost=hand_cost)


def environment_targeted_style(env, staged=None):
    if not hasattr(env, '_targeted_waist_index'):
        names = list(env.robot.joint_names)
        if names.count('waist_yaw_joint') != 1:
            raise ValueError('Expected exactly one waist_yaw_joint')
        env._targeted_waist_index = names.index('waist_yaw_joint')
    i = env._targeted_waist_index
    data = env.robot.data
    settings = env.cfg.pair_standup_reward_cfg['v12_targeted_style']
    base_settings = settings if 'pose' not in settings else {**settings, 'waist_weight':0., 'waist_tolerance':settings['pose']['waist_tolerance']}
    terms, diagnostics = targeted_style(to_torch(data.joint_pos)[:, i], to_torch(data.default_joint_pos)[:, i],
                          env._hand_state['load'], base_settings, env.step_dt)

    c = env.cfg.pair_standup_reward_cfg['v12_targeted_style']
    if 'arm_weight' in c:
        if not hasattr(env, '_targeted_arm_indices'):
            names = list(env.robot.joint_names)
            arm_names = [f'{side}_{joint}_joint' for side in ('left', 'right')
                         for joint in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll')]
            env._targeted_arm_indices = [names.index(n) for n in arm_names]
        indices = env._targeted_arm_indices
        error = (to_torch(data.joint_pos)[:, indices] - to_torch(data.default_joint_pos)[:, indices]).abs()
        cost = (error - c['arm_tolerance']).clamp_min(0).square().sum(-1)
        ready = staged.diagnostics['loaded_rise'].clamp(0, 1)
        unload = (1-env._hand_state['load'].clamp_min(0).sum(-1)/c['arm_unload_scale']).clamp(0, 1)
        gate = ready*unload
        terms['arm_pose'] = -env.step_dt*c['arm_weight']*gate*cost
        diagnostics.update(arm_pose_cost=cost, arm_pose_gate=gate)
    if 'hip_yaw_weight' in c:
        from .turn_alignment import rotation_costs
        if not hasattr(env, '_targeted_hip_yaw_indices'):
            names = list(env.robot.joint_names)
            env._targeted_hip_yaw_indices = [names.index(side+'_hip_yaw_joint') for side in ('left','right')]
        indices = env._targeted_hip_yaw_indices
        rotation_terms, rotation_diagnostics = rotation_costs(
            to_torch(data.joint_pos)[:,indices], to_torch(data.default_joint_pos)[:,indices],
            to_torch(data.joint_vel)[:,i], env._stance_sole_heading_xy,
            staged.diagnostics['body_ready'], c, env.step_dt)
        terms.update(rotation_terms)
        diagnostics.update(rotation_diagnostics)
    if 'pose' in c:
        from .joint_pose import environment_joint_pose
        pose_reward, pose_diagnostics = environment_joint_pose(env, staged, c['pose'])
        terms.pop('waist_yaw', None)
        terms.pop('hip_yaw', None)
        terms['pose'] = pose_reward
        diagnostics.update(pose_diagnostics)
    return terms, diagnostics

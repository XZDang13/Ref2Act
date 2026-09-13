"""One default-pose objective with named waist, arm and leg groups."""
import torch


def pose_arrays(names, c):
    groups = {'waist': ['waist_yaw_joint'],
              'arm': [f'{s}_{j}_joint' for s in ('left','right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll')],
              'leg': [f'{s}_{j}_joint' for s in ('left','right') for j in ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]}
    if len(names) != 23 or set(names) != set(sum(groups.values(), [])):
        raise ValueError('Pose objective requires the named G1 23 joints')
    weights=[]; tolerances=[]; arm=[]
    for n in names:
        g=next(g for g in groups if n in groups[g]);weights.append(c[g+'_weight']);arm.append(g=='arm')
        if g=='leg':
            tol=next(v for suffix,v in [('hip_pitch_joint',.35),('hip_yaw_joint',.4363323129985824),('knee_joint',.5),('ankle_pitch_joint',.25),('hip_roll_joint',.1),('ankle_roll_joint',.1)] if n.endswith(suffix))
        else:tol=c[g+'_tolerance']
        tolerances.append(tol)
    return weights,tolerances,arm


def joint_pose_cost(q,default,weights,tolerances,arm_mask,arm_gate):
    excess=((q-default).abs()-tolerances).clamp_min(0)
    gate=torch.where(arm_mask[None],arm_gate[:,None],torch.ones_like(q))
    return (excess.square()*weights*gate).sum(-1)


def environment_joint_pose(env,staged,c):
    from ref2act.isaac_compat import to_torch
    q=to_torch(env.robot.data.joint_pos)
    if not hasattr(env,'_unified_pose_arrays'):
        w,t,a=pose_arrays(list(env.robot.joint_names),c)
        env._unified_pose_arrays=(torch.tensor(w,device=q.device,dtype=q.dtype),torch.tensor(t,device=q.device,dtype=q.dtype),torch.tensor(a,device=q.device,dtype=torch.bool))
    unload=(1-env._hand_state['load'].clamp_min(0).sum(-1)/c['arm_unload_scale']).clamp(0,1)
    gate=staged.diagnostics['loaded_rise'].clamp(0,1)*unload
    cost=joint_pose_cost(q,to_torch(env.robot.data.default_joint_pos),*env._unified_pose_arrays,gate)
    return -env.step_dt*cost, dict(pose_cost=cost,pose_arm_gate=gate)

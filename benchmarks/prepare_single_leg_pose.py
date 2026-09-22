"""Select a motion seed, then solve a static left-support pose with G1 MJCF.

Run with --motion one or more retargeted NPZ files and --output pose.json.
MuJoCo and SciPy are offline preparation dependencies only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares


def prepare(model_path, motions):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    names = [model.joint(i).name for i in range(1, model.njnt)]
    left = model.body('left_ankle_roll_link').id
    right = model.body('right_ankle_roll_link').id
    pelvis = model.body('pelvis').id
    points = np.array([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
    def geometry(q):
        data.qpos[:] = np.r_[0.,0.,.8,1.,0.,0.,0.,q]
        mujoco.mj_forward(model, data)
        soles = np.stack([points @ data.xmat[i].reshape(3,3).T + data.xpos[i] for i in (left,right)])
        soles[:,:,2] -= .005
        com = data.subtree_com[pelvis].copy()
        return soles, com
    best = None
    for path in motions:
        with np.load(path, allow_pickle=False) as clip:
            order = [list(clip['joint_names']).index(n) for n in names]
            for frame, q in enumerate(clip['joint_pos'][:,order]):
                soles, com = geometry(q)
                clearance = soles[1,:,2].min()-soles[0,:,2].min()
                score = (abs(clearance-.12)*3 + np.ptp(soles[0,:,2])*4
                         + np.linalg.norm(com[:2]-soles[0,:,:2].mean(0))*5
                         + max(0., np.abs(q).max()-1.5))
                if best is None or score < best[0]:
                    best = (score, q.copy(), str(path), frame)
    if best is None:
        raise ValueError('Provide at least one retargeted motion with named joints')
    seed = best[1]
    lo, hi = model.jnt_range[1:].T
    lo, hi = lo.copy(), hi.copy()
    for i, name in enumerate(names):
        if 'hip_yaw' in name or 'waist' in name: lo[i],hi[i]=-.25,.25
        if 'hip_roll' in name: lo[i],hi[i]=max(lo[i],-.4),min(hi[i],.4)
        if 'shoulder_pitch' in name: lo[i],hi[i]=-.5,.5
        if 'shoulder_roll' in name: lo[i],hi[i]=max(lo[i],-.7),min(hi[i],.7)
        if 'shoulder_yaw' in name: lo[i],hi[i]=-.5,.5
    seed = np.clip(seed,lo+.01,hi-.01)
    def residual(q):
        soles, com = geometry(q)
        normal = data.xmat[left].reshape(3,3)[:,2]
        return np.r_[(com[:2]-soles[0,:,:2].mean(0))*30,
                     normal[:2]*10, (soles[1,:,2].min()-soles[0,:,2].min()-.12)*20,
                     (q[names.index('left_knee_joint')]-.35)*.5,
                     (q[names.index('right_knee_joint')]-.85)*.5,
                     (soles[1,:,:2].mean(0)-soles[0,:,:2].mean(0)-[.02,-.20])*3,
                     (q-seed)*.06]
    fit = least_squares(residual,seed,bounds=(lo+.005,hi-.005),max_nfev=1500,
                        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    soles, com = geometry(fit.x)
    root_z = .8-soles[0,:,2].min()+.001
    report = dict(support_sole_height_spread_m=float(np.ptp(soles[0,:,2])),
                  com_support_distance_m=float(np.linalg.norm(com[:2]-soles[0,:,:2].mean(0))),
                  swing_clearance_m=float(soles[1,:,2].min()-soles[0,:,2].min()+.001))
    if report['support_sole_height_spread_m']>.005 or report['com_support_distance_m']>.01 or not .10<report['swing_clearance_m']<.15:
        raise RuntimeError(f'IK validation failed: {report}')
    return dict(version=1, support_leg='left', root_position=[0.,0.,root_z],root_quaternion=[0.,0.,0.,1.],quaternion_convention="xyzw",
                joint_positions=dict(zip(names,fit.x.tolist())),
                provenance=dict(method='motion_seed_then_static_mujoco_IK',motion=best[2],frame=best[3],
                                motion_sha256=hashlib.sha256(Path(best[2]).read_bytes()).hexdigest(),
                                model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest()),
                kinematic_validation=report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,nargs='+',required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'src/ref2act/assets/robots/g1/g1_23dof_rubber_hand.xml')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=prepare(a.model,a.motion)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

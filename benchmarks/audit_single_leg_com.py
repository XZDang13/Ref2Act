"""Independent live CoM reconstruction and ground-projection footprint audit."""
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--nominal',action='store_true');p.add_argument('--output',required=True,type=Path)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from scipy.spatial import ConvexHull
    import gymnasium as gym,ref2act
    from ref2act.robots.g1 import G1SingleLegBalanceEnvCfg
    from ref2act.isaac_compat import to_torch as T
    from ref2act.common.math import quat_apply
    from ref2act.envs.single_leg_balance.logic import capture_point
    cfg=G1SingleLegBalanceEnvCfg();cfg.scene.num_envs=64;cfg.seed=101
    if args.nominal:
        cfg.events=None
        for k in cfg.task['reset_noise']:cfg.task['reset_noise'][k]=0.
    cfg.task['actor_noise']['enabled']=False
    env=gym.make('G1SingleLegBalance-v0',cfg=cfg);c=env.unwrapped
    def reconstruct():
        d=c.robot.data;m=T(d.body_mass)
        offset=quat_apply(T(d.body_link_quat_w).reshape(-1,4),T(d.body_com_pos_b).reshape(-1,3)).reshape(c.num_envs,-1,3)
        pos=T(d.body_link_pos_w)+offset
        vel=T(d.body_link_lin_vel_w)+torch.cross(T(d.body_link_ang_vel_w),offset,dim=-1)
        s=c._geometry()
        return float((s['com']-(m[...,None]*pos).sum(1)/m.sum(-1)[:,None]).abs().max()),float((s['com_vel']-(m[...,None]*vel).sum(1)/m.sum(-1)[:,None]).abs().max())
    with torch.inference_mode():
        env.reset();s=c._geometry();d=c.robot.data
        com_error,vel_error=reconstruct()
        quat=T(d.body_link_quat_w)[:,c._feet[0]]
        corner=s['com'].new_tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
        points=quat_apply(quat[:,None].expand(-1,4,-1).reshape(-1,4),corner[None].expand(c.num_envs,-1,-1).reshape(-1,3)).reshape(c.num_envs,4,3)+s['feet'][:,0,None]
        points[...,2]-=.005
        cp=capture_point(s['com'],s['com_vel'],T(c.scene.env_origins)[:,2])
        axis=quat_apply(quat,s['com'].new_tensor([1.,0.,0.]).expand(c.num_envs,-1))
        yaw=torch.atan2(axis[:,1],axis[:,0])
        rows=[]
        for i in range(c.num_envs):
            center=s['support'][i,:2].cpu().numpy()
            angle=yaw[i].item();r=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
            polygon=(points[i,:,:2].cpu().numpy()-center)@r.T
            proj=(s['com'][i,:2].cpu().numpy()-center)@r.T
            capture=(cp[i].cpu().numpy()-center)@r.T
            hull=ConvexHull(polygon)
            def margin(point):return float(-(hull.equations[:,:2]@point+hull.equations[:,2]).max())
            rows.append(dict(com_xy_foot=proj.tolist(),capture_xy_foot=capture.tolist(),footprint=polygon.tolist(),
                             com_margin_m=margin(proj),capture_margin_m=margin(capture),
                             com_center_error_m=float(np.linalg.norm(proj)),capture_center_error_m=float(np.linalg.norm(capture)),
                             sole_height_spread_m=float(torch.ptp(points[i,:,2])) if hasattr(torch,'ptp') else float(points[i,:,2].max()-points[i,:,2].min()),
                             swing_clearance_m=float(s['clearance'][i,1])))
        def quantiles(key):return dict(zip(['min','median','p95','max'],map(float,np.quantile([r[key] for r in rows],[0,.5,.95,1]))))
        report=dict(nominal=args.nominal,seed=101,num_envs=64,
                    max_com_reconstruction_error_m=com_error,max_velocity_reconstruction_error_m_s=vel_error,
                    com_inside_footprint_fraction=sum(r['com_margin_m']>=0 for r in rows)/len(rows),
                    capture_inside_footprint_fraction=sum(r['capture_margin_m']>=0 for r in rows)/len(rows),
                    com_error_m=quantiles('com_center_error_m'),capture_error_m=quantiles('capture_center_error_m'),
                    com_margin_m=quantiles('com_margin_m'),capture_margin_m=quantiles('capture_margin_m'),
                    sole_height_spread_m=quantiles('sole_height_spread_m'),rows=rows)
        action=c._sim_to_policy_order(c.action_processor.applied_action).clone()
        trace=[]
        for k in range(25):
            _,_,term,trunc,_=env.step(action)
            state=c._state
            a,b=reconstruct();report['max_com_reconstruction_error_m']=max(report['max_com_reconstruction_error_m'],a)
            report['max_velocity_reconstruction_error_m_s']=max(report['max_velocity_reconstruction_error_m_s'],b)
            trace.append(dict(time_s=(k+1)*c.step_dt,com_error_m=float((state['com'][0,:2]-state['support'][0,:2]).norm()),
                              capture_error_m=float(state['balance_error'][0]),height_m=float(state['height'][0]),
                              left_load=float(state['load'][0,0]),right_load=float(state['load'][0,1]),
                              failed=bool(term[0])))
            if term[0] or trunc[0]:break
        report['open_loop_trace_env0']=trace
        assert report['max_com_reconstruction_error_m']<1e-5
        assert report['max_velocity_reconstruction_error_m_s']<1e-5
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('COM_AUDIT',json.dumps({k:v for k,v in report.items() if k not in ('rows','open_loop_trace_env0')}),flush=True)
    env.close()
finally:app.close()

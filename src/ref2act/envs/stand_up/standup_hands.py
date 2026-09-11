"""Ground-only hand sensing and bounded support quality for staged stand-up."""
import math
import re
import torch
from .standup_support import ground_normal_force


def validate_hands(c):
    expected={'enabled','body_names','contact_load','full_load','persistence_s','slip_sigma',
        'approach_height','height_sigma','release_height','preparation_fraction','release_fraction'}
    if set(c)!=expected or type(c['enabled']) is not bool:
        raise ValueError('Invalid hand support configuration')
    if len(c['body_names'])!=2 or len(set(c['body_names']))!=2 or not all(isinstance(v,str) and v for v in c['body_names']):
        raise ValueError('Specify distinct left and right hand rigid bodies')
    for k in expected-{'enabled','body_names'}:
        if not math.isfinite(float(c[k])) or c[k]<=0:
            raise ValueError(f'Invalid hand setting {k}')
    if not c['contact_load']<c['full_load']<=1 or max(c['preparation_fraction'],c['release_fraction'])>=1:
        raise ValueError('Invalid hand loads or reward fractions')


def hand_quality(force, weight, slip, height, contact_steps, *, step_dt, cfg):
    """A brief force spike cannot earn persistent support credit.

    height is hand rigid-body origin height, not measured contact clearance;
    slip is a rigid-body ground-projection speed estimate, not exact patch slip.
    """
    load=(force[...,2].clamp_min(0)/weight[:,None]).clamp(0,1)
    contact=load>=cfg['contact_load']
    hold=max(1,math.ceil(cfg['persistence_s']/step_dt))
    steps=torch.where(contact,contact_steps+1,0).clamp_max(hold)
    persistent=contact & (steps>=hold)
    support=(load/cfg['full_load']).clamp(0,1)*persistent
    support*=1/(1+(slip/cfg['slip_sigma']).square())
    approach=1/(1+((height.abs()-cfg['approach_height']).clamp_min(0)/cfg['height_sigma']).square())
    release=(1-load/cfg['contact_load']).clamp(0,1)*(height/cfg['release_height']).clamp(0,1)
    return dict(load=load,contact=contact,persistent=persistent,steps=steps,
        support=support,approach=approach,release=release,slip=slip,height=height)


class HandContactReader:
    """One view per hand; robot contacts subtracted from net normal force.

    Valid for this flat-ground scene without other external objects. No fallback
    to wrist bodies: a sensor must not silently include a different collision set.
    """
    def __init__(self,env,names):
        from pxr import UsdPhysics
        from ref2act.isaac_compat import to_torch as _to_torch
        paths={p.GetName():str(p.GetPath()) for p in env.sim.stage.Traverse()
            if str(p.GetPath()).startswith('/World/envs/env_0/Robot/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.views=[];self.dt=env.physics_dt;self.convert=_to_torch
        self.body_ids=[env.robot.body_names.index(n) for n in names]
        self.paths=[paths[n] for n in names]
        for name in names:
            patterns=[paths[name].replace('/env_0/',f'/env_{i}/') for i in range(env.num_envs)]
            filters=[[p.replace('/env_0/',f'/env_{i}/') for p in sorted(paths.values())] for i in range(env.num_envs)]
            bodies=env.sim.physics_sim_view.create_rigid_body_view(patterns)
            ids=[int(re.search(r'/env_(\d+)/',p).group(1)) for p in bodies.prim_paths]
            if sorted(ids)!=list(range(env.num_envs)):raise RuntimeError('Incomplete hand contact view')
            order=torch.tensor(ids,device=env.device).argsort()
            view=env.sim.physics_sim_view.create_rigid_contact_view(patterns,filter_patterns=filters,
                max_contact_data_count=env.num_envs*128)
            if view.filter_count!=len(paths):raise RuntimeError('Incorrect hand self-contact filters')
            self.views.append((view,order))

    def read(self):
        forces=[]; net_forces=[]; self_forces=[]
        for view,order in self.views:
            net=self.convert(view.get_net_contact_forces(dt=self.dt))[order]
            own=self.convert(view.get_contact_force_matrix(dt=self.dt))[order]
            net_forces.append(net); self_forces.append(own.sum(-2))
            forces.append(ground_normal_force(net,own))
        self.last_net_force = torch.stack(net_forces,1)
        self.last_robot_force = torch.stack(self_forces,1)
        return torch.stack(forces,1)

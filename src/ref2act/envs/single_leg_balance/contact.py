"""Ground-only contacts for a flat scene without external objects."""
import re
import torch

class GroundContactReader:
    """Per-body ground forces: subtract explicitly filtered self contacts.

    Valid for this flat-ground scene without other external objects. No fallback
    to different bodies: a sensor must not silently include a different collision set.
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
            if sorted(ids)!=list(range(env.num_envs)):raise RuntimeError('Incomplete body contact view')
            order=torch.tensor(ids,device=env.device).argsort()
            view=env.sim.physics_sim_view.create_rigid_contact_view(patterns,filter_patterns=filters,
                max_contact_data_count=env.num_envs*128)
            if view.filter_count!=len(paths):raise RuntimeError('Incorrect body self-contact filters')
            self.views.append((view,order))

    def read(self):
        forces=[]; net_forces=[]; self_forces=[]
        for view,order in self.views:
            net=self.convert(view.get_net_contact_forces(dt=self.dt))[order]
            own=self.convert(view.get_contact_force_matrix(dt=self.dt))[order]
            net_forces.append(net); self_forces.append(own.sum(-2))
            forces.append(net-own.sum(-2))
        self.last_net_force = torch.stack(net_forces,1)
        self.last_robot_force = torch.stack(self_forces,1)
        return torch.stack(forces,1)

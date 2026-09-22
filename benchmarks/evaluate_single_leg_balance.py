"""Evaluate a TorchScript deterministic actor: raw [N,312] -> normalized [N,23].

Export any required robot-input normalization with the actor. This is not a
loader for training checkpoints. Reset-hold mode is an environment diagnostic.
"""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
g=p.add_mutually_exclusive_group(required=True)
g.add_argument('--policy',type=Path);g.add_argument('--reset-hold',action='store_true')
p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--seed',type=int,default=101)
p.add_argument('--strengths',type=float,nargs='+',default=[0.,.1,.2,.3,.4,.5])
p.add_argument('--directions',type=int,default=8)
p.add_argument('--threshold',type=float,default=.8)
p.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
if args.num_envs<1 or args.directions<1:p.error('Counts must be positive')
app=AppLauncher(args).app
try:
    import math
    import torch
    import gymnasium as gym
    import ref2act
    from ref2act.robots.g1 import G1SingleLegBalanceEnvCfg
    from ref2act.envs.single_leg_balance.evaluation import summarize
    from ref2act.envs.single_leg_balance.logic import EpisodeRandom
    cfg=G1SingleLegBalanceEnvCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    cfg.task['actor_noise']['enabled']=False
    cfg.task['push']['time_range']=[2.,2.]
    env=gym.make('G1SingleLegBalance-v0',cfg=cfg);core=env.unwrapped
    actor=torch.jit.load(str(args.policy),map_location=core.device).eval() if args.policy else None
    rows=[];directions=[i*2*math.pi/args.directions for i in range(args.directions)]
    with torch.inference_mode():
        for dv in args.strengths:
            for theta in directions:
                core.task['push']['evaluation_delta_v']=dv;core.task['push']['evaluation_direction']=theta
                # Same per-env reset perturbations in each cell and across actors.
                core._rng=EpisodeRandom(core.num_envs,core.device,args.seed)
                obs,_=env.reset();seen=set()
                hold=core._sim_to_policy_order(core.action_processor.applied_action).clone()
                for _ in range(core.max_episode_length+1):
                    action=actor(obs['policy']) if actor else hold
                    obs,_,_,_,extras=env.step(action)
                    outcome=extras['single_leg_outcomes']
                    for i,eid in enumerate(outcome['env_ids'].tolist()):
                        if eid in seen:continue
                        seen.add(eid)
                        row={k:v[i].tolist() for k,v in outcome.items()};rows.append(row)
                    if len(seen)==core.num_envs:break
                if len(seen)!=core.num_envs:raise RuntimeError('Incomplete evaluation episodes')
                print(f'delta_v={dv:.2f}, direction={theta:.3f}: {len(seen)} episodes',flush=True)
    result=summarize(rows,args.threshold,tuple(args.strengths),tuple(directions))
    result.update(seed=args.seed,controller=str(args.policy) if args.policy else 'reset_hold_diagnostic',
                  task=core.task,episodes=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    env.close()
finally:
    app.close()

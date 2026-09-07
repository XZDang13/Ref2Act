"""GPU integration: command changes, phase/reward agreement and subset reset."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import copy
import gymnasium as gym
import torch
import ref2act
from ref2act.robots.g1 import G1FlatLocomotionEnvCfg
from ref2act.envs.locomotion.rewards import compute_locomotion_phase_features
import ref2act.envs.locomotion.env as module
cfg=G1FlatLocomotionEnvCfg();cfg.scene.num_envs=64
if cfg.sim.physics is None:
    from isaaclab_physx.physics import PhysxCfg
    cfg.sim.physics=PhysxCfg()
env=gym.make('G1FlatLocomotion-v0',cfg=cfg);core=env.unwrapped
original=module.compute_flat_locomotion_reward_terms
calls=0

def checked(inputs,reward_cfg):
    global calls
    assert torch.equal(inputs.gait_phase,core._adaptive_gait.phase())
    calls+=1
    return original(inputs,reward_cfg)

module.compute_flat_locomotion_reward_terms=checked
try:
    obs,_=env.reset();clock=core._adaptive_gait
    for i in range(80):
        cmd=core.command_generator.commands
        cmd.zero_()
        if i<20:cmd[:,0]=.3
        elif i<40:cmd[:,0]=.6;cmd[:,2]=.25
        elif i>=65:cmd[:,1]=-.3
        core.command_generator.steps_left.fill_(1000)
        expected=copy.deepcopy(clock);expected.advance(cmd)
        obs,reward,term,trunc,_=env.step(torch.zeros((64,23),device=core.device))
        valid=~(term|trunc)
        assert torch.allclose(clock.angle[valid],expected.angle[valid],atol=1e-6)
        assert torch.allclose(clock.frequency[valid],expected.frequency[valid],atol=1e-6)
        assert torch.allclose(obs['command'][:,3:7],compute_locomotion_phase_features(clock.phase()),atol=1e-6)
        assert all(torch.isfinite(v).all() for v in [*obs.values(),reward])
        angle=clock.angle.clone();frequency=clock.frequency.clone()
        core._reset_idx(torch.arange(0,8,device=core.device))
        assert torch.equal(clock.angle[8:],angle[8:])
        assert torch.equal(clock.frequency[8:],frequency[8:])
    assert calls==80
    print('adaptive gait GPU smoke passed: 64 environments, 80 steps, phase/reward/observation agreement, command changes and subset resets',flush=True)
finally:
    module.compute_flat_locomotion_reward_terms=original
    env.close();app.close()

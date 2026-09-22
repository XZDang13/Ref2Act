"""Native Isaac smoke: applied-action history, partial reset, pulse and timeout."""
import argparse
import json
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
try:
    import gymnasium as gym
    import torch
    import ref2act
    from ref2act.robots.g1 import G1SingleLegBalanceEnvCfg
    cfg=G1SingleLegBalanceEnvCfg();cfg.scene.num_envs=8;cfg.sim.device='cuda:0'
    cfg.events=None
    cfg.task['actor_noise']['enabled']=False
    for key in cfg.task['reset_noise']:cfg.task['reset_noise'][key]=0.
    env=gym.make('G1SingleLegBalance-v0',cfg=cfg);c=env.unwrapped
    with torch.inference_mode():
        obs,_=env.reset();assert obs['policy'].shape==(8,312) and obs['critic'].shape==(8,140)
        assert torch.isfinite(obs['policy']).all()
        initial=c._geometry()
        print('INITIAL_GEOMETRY', {k:v[0].tolist() for k,v in initial.items()},flush=True)
        action=c._sim_to_policy_order(c.action_processor.applied_action).clone()
        for _ in range(5):
            obs,reward,term,trunc,extras=env.step(action)
            assert torch.isfinite(reward).all() and torch.isfinite(obs['critic']).all()
        assert not term.any(), 'Nominal pose failed in the first 100ms'
        print('CONTACTS_100MS',c._state['load'].tolist(),flush=True)
        # Force pulse at the next interval for integration verification, leaving
        # normal task sampling unchanged. It must clear on a partial reset.
        c._start[:]=c.episode_length_buf*c.step_dt;c._dv[:]=.1;c._direction[:]=0
        before=c._impulse.clone()
        obs,reward,term,trunc,extras=env.step(action)
        assert (c._impulse[:,0]>before[:,0]).all()
        c._reset_idx(torch.tensor([0,2],device=c.device))
        assert (c._force[[0,2]]==0).all() and (c._impulse[[0,2]]==0).all()
        assert (c._timers[[0,2]]==0).all()
        # Full fresh reset, then a clean alternating-row time limit.
        env.reset();c._dv.zero_()
        c.episode_length_buf[::2]=c.max_episode_length-1
        obs,reward,term,trunc,extras=env.step(action)
        assert trunc[::2].all() and not trunc[1::2].any() and not term.any()
        assert extras['final_critic_mask'].tolist()==[True,False]*4
        assert extras['final_critic_observation'].shape==(4,140)
        torch.testing.assert_close(c.history.proprio[::2],c.history.proprio[::2,:1].expand(-1,4,-1))
        torch.testing.assert_close(c.history.actions[::2],c._sim_to_policy_order(c.action_processor.applied_action)[::2,None].expand(-1,4,-1))
        assert extras['single_leg_outcomes']['survived'].all()
        # A failure coincident with the time limit must not count as success or
        # supply a timeout bootstrap row. Exercise the actual task done path.
        s=c._state
        s['swing_contact'][0]=True
        c._timers[0,0]=cfg.task['contact']['grace_s']
        c.episode_length_buf[0]=c.max_episode_length
        terminated,timed_out=c._get_dones()
        assert terminated[0] and timed_out[0]
        assert not c.extras['final_critic_mask'][0]
        outcome=c.extras['single_leg_outcomes']
        assert not outcome['survived'][outcome['env_ids']==0].any()
        # Measure open-loop pose hold; this is not a trained policy evaluation.
        env.reset();lengths=[]
        for _ in range(320):
            obs,reward,term,trunc,extras=env.step(action)
            assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
            lengths.extend(extras['single_leg_outcomes']['duration_s'].tolist())
        print('SINGLE_LEG_SMOKE_PASS',json.dumps(dict(policy=312,critic=140,partial_timeout=True,
              pulse=True,open_loop_episode_lengths_s=lengths[:24])),flush=True)
    env.close()
finally:
    app.close()

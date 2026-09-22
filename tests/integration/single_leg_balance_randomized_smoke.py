from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
try:
    import torch,gymnasium as gym,ref2act,json
    from ref2act.robots.g1 import G1SingleLegBalanceEnvCfg
    cfg=G1SingleLegBalanceEnvCfg();cfg.scene.num_envs=64
    env=gym.make('G1SingleLegBalance-v0',cfg=cfg);c=env.unwrapped
    lengths=[]
    with torch.inference_mode():
        obs,_=env.reset()
        assert torch.isfinite(obs['policy']).all()
        for _ in range(100):
            obs,r,t,x,e=env.step(torch.randn(64,23,device=c.device)*.2)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(obs['critic']).all() and torch.isfinite(r).all()
            lengths.extend(e['single_leg_outcomes']['duration_s'].tolist())
        print('RANDOMIZED_PASS',json.dumps(dict(envs=64,steps=100,episodes=len(lengths),min_duration=min(lengths),mean_duration=sum(lengths)/len(lengths))),flush=True)
    env.close()
finally:app.close()

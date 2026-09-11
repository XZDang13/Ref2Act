"""Exercise native Gym task while explicitly forbidding IRASP imports."""
import argparse
import importlib.abc
import sys
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
try:
    class NoIRASP(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'irasp' or fullname.startswith('irasp.'):
                raise ImportError('Native Ref2Act stand-up must not import IRASP: '+fullname)
    sys.meta_path.insert(0, NoIRASP())
    import gymnasium as gym
    import torch
    import ref2act
    from ref2act.robots.g1 import G1StandUpEnvCfg
    from ref2act.envs.base import LeggedRobotEnv
    from ref2act.envs.stand_up.env import StandUpEnv
    assert StandUpEnv.__bases__ == (LeggedRobotEnv,)
    cfg = G1StandUpEnvCfg()
    cfg.scene.num_envs = 8
    cfg.sim.device = 'cuda:0'
    assert not hasattr(cfg, 'command') and not hasattr(cfg, 'rewards')
    assert cfg.events.rand_robot_mass is not None
    env = gym.make('G1StandUp-v0', cfg=cfg)
    core = env.unwrapped
    with torch.inference_mode():
        obs, extras = env.reset()
        assert obs['policy'].shape == (8, 312)
        assert obs['critic'].shape == (8, 181)
        assert core._episode_sampler.best.shape == (8, 16)
        assert not hasattr(core, 'command_generator')
        assert not hasattr(core, 'observation_model')
        assert not hasattr(core, '_adaptive_gait')
        for _ in range(40):
            obs, reward, terminated, truncated, extras = env.step(torch.zeros(8, 23, device=core.device))
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs['critic']).all()
        # Partial timeout: critic must describe pre-reset state; histories only
        # reset selected rows and contain the post-reset physical joint target.
        core._episode_sampler.deadline[::2] = core.episode_length_buf[::2]+1
        before = core._native_observation.proprio.clone()
        obs, reward, terminated, truncated, extras = env.step(torch.zeros(8, 23, device=core.device))
        assert truncated[::2].all() and not truncated[1::2].any()
        assert not terminated.any()
        assert extras['final_critic_mask'].tolist() == [True,False]*4
        final = extras['final_critic_observation']
        assert final.shape == (4,181) and torch.isfinite(final).all()
        assert not torch.allclose(final,obs['critic'][::2])
        history = core._native_observation
        torch.testing.assert_close(history.proprio[1::2,:-1],before[1::2,1:])
        torch.testing.assert_close(history.proprio[::2],history.proprio[::2,:1].expand(-1,4,-1))
        torch.testing.assert_close(history.actions[::2],history.extractor.action_history_value()[::2,None].expand(-1,4,-1))
        env.reset()
        assert (core._transfer_ground_fraction_filter==0).all()
        assert torch.isfinite(obs['policy']).all()
    assert not any(n == 'irasp' or n.startswith('irasp.') for n in sys.modules)
    import json
    result = {'task':'G1StandUp-v0','base':'LeggedRobotEnv','num_envs':8,
        'policy':312,'critic':181,'milestones':16,'partial_timeout':True,
        'irasp_imported':False,'root_target':core._pair_recovery_target_root_height,
        'shoulder_target':core._pair_recovery_target_shoulder_height}
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
    print('NATIVE_STANDUP_PASS',result,flush=True)
    env.close()
except BaseException:
    import traceback
    traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
    raise
finally:
    app.close()

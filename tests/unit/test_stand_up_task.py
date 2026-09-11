"""Native task contracts; no dependency on IRASP or its training fixtures."""
import ast
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import torch
import ref2act
from ref2act.envs.stand_up.defaults import default_config
from ref2act.envs.stand_up.standup_stages import validate_stages
from ref2act.envs.stand_up import observation as observation_module


def test_registration_and_dependency_direction():
    spec = gym.spec('G1StandUp-v0')
    assert spec.entry_point == 'ref2act.envs.stand_up.env:StandUpEnv'
    assert spec.kwargs['cfg_factory'] == 'ref2act.robots.g1:G1StandUpEnvCfg'
    package = Path(observation_module.__file__).parent
    for path in package.glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('irasp')
    tree = ast.parse((package/'env.py').read_text())
    env = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [ast.unparse(n) for n in env.bases] == ['LeggedRobotEnv']


def test_v12_defaults_are_valid_and_isolated():
    first, second = default_config(), default_config()
    stages=first['standup']['support']['stages']
    assert stages['weights']==dict(preparation=2.5,rise=2.5,stand=2.0)
    assert not any(k in stages for k in ('rise_hold_v2','rise_height_v3','foot_discovery_v2','foot_retraction_v3','foot_ground_retraction_v4','foot_stance_v5'))
    assert first['standup']['reward'].get('mode')!='simple_v13'
    assert first['standup']['initial_state']['mode']=='fixed_supine'
    validate_stages(stages)
    first['standup']['support']['stages']['hands']['enabled'] = False
    assert second['standup']['support']['stages']['hands']['enabled']
    assert second['standup']['actor_observation_noise']['enabled']
    assert second['ref2act']['action_delay_range'] == [0,0]
    assert not second['ref2act']['disable_startup_randomization']


def test_actor_history_partial_reset_and_current_critic(monkeypatch):
    frame = torch.arange(110,dtype=torch.float).reshape(2,55)
    target = torch.arange(46,dtype=torch.float).reshape(2,23)
    extractor = SimpleNamespace(extract=lambda:frame.clone(), action_history_value=lambda:target.clone())
    monkeypatch.setattr(observation_module,'StandUpStateExtractor',lambda env:extractor)
    monkeypatch.setattr(observation_module,'critic_terms',lambda env, extractor, mask: {'current':frame[:,:3].clone()})
    env=SimpleNamespace(cfg=SimpleNamespace(actor_observation_noise={'enabled':False}))
    history=observation_module.StandUpObservation(env)
    obs=history.update(torch.ones(2,dtype=torch.bool))
    assert obs['policy'].shape == (2,312)
    torch.testing.assert_close(obs['policy'][:,:220],frame[:,None].repeat(1,4,1).flatten(1))
    first=frame.clone()
    frame.add_(100);target.add_(10)
    obs=history.update(torch.tensor([False,True]))
    torch.testing.assert_close(history.proprio[0,:3],first[0,None].expand(3,55))
    torch.testing.assert_close(history.proprio[0,-1],frame[0])
    torch.testing.assert_close(history.proprio[1],frame[1,None].expand(4,55))
    torch.testing.assert_close(history.actions[1],target[1,None].expand(4,23))
    torch.testing.assert_close(obs['critic'],frame[:,:3])

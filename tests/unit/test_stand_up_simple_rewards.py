from dataclasses import replace
import pytest
import torch
import json
from pathlib import Path
def default_config():
    # Explicit historical V13 fixture; native task defaults are V12.
    return json.loads((Path(__file__).parent/'fixtures/stand_up_v13.json').read_text())
from ref2act.envs.stand_up.simple_rewards import compose_rewards, validate_simple_rewards
from ref2act.envs.stand_up.standup_stages import StageInputs, stage_rewards


def inputs():
    return dict(stage_terms={k:torch.tensor([.01,.02,.03]) for k in ('preparation','rise','stand')},
        action=torch.ones(3,23),previous_action=torch.zeros(3,23),
        joint_position=torch.zeros(3,23),joint_limits=torch.tensor([-1.,1.]).repeat(3,23,1),
        finite=torch.ones(3,dtype=torch.bool),failed=torch.zeros(3,dtype=torch.bool),
        settling=torch.zeros(3,dtype=torch.bool),cfg={k:v for k,v in default_config()['standup']['reward'].items() if k not in ('posture','hip_collision_weight')},dt=.02)


def test_exact_six_terms_and_independent_action_rate():
    kw=inputs();r=compose_rewards(**kw)
    assert set(r)=={'preparation','rise','stand','action_rate','joint_limit','failure'}
    torch.testing.assert_close(r['action_rate'],torch.full((3,),-.001))
    kw['previous_action']=kw['action'].clone()
    assert (compose_rewards(**kw)['action_rate']==0).all()
    assert (r['joint_limit']==0).all()
    kw['joint_position'].fill_(1.)
    torch.testing.assert_close(compose_rewards(**kw)['joint_limit'],torch.full((3,),-.002))


def test_invalid_failure_and_settling_do_not_collect_positive_credit():
    kw=inputs();kw['finite'][0]=False;kw['action'][0]=float('nan')
    kw['failed'][1]=True;kw['settling'][2]=True
    r=compose_rewards(**kw)
    torch.testing.assert_close(sum(r.values()),torch.tensor([-100.,-100.,0.]))
    assert all(torch.isfinite(v).all() for v in r.values())


def test_continuous_costs_scale_with_dt_but_failure_does_not():
    kw=inputs();kw['failed'][0]=True
    a=compose_rewards(**kw);kw['dt']/=2
    kw['stage_terms']={k:v/2 for k,v in kw['stage_terms'].items()}
    b=compose_rewards(**kw)
    for k in a:torch.testing.assert_close(b[k],a[k] if k=='failure' else a[k]/2)


def test_positive_stages_preserve_support_gate_and_reward_order():
    c=default_config()['standup']['support']['stages']
    for k in ('hands','rise_transfer','stance','rise_hold_v2','foot_discovery_v2','foot_retraction_v3','foot_ground_retraction_v4','foot_stance_v5','rise_height_v3'):c.pop(k,None)
    t=lambda v:torch.tensor(v,dtype=torch.float32)
    x=StageInputs(t([.4,.76]),t([.7,1.05]),.76,1.05,t([.6,1.]),
        t([[.15,.15],[.15,.15]]),torch.ones(2,2),torch.full((2,2),.5),
        torch.ones(2,2,dtype=torch.bool),torch.ones(2),torch.zeros(2),torch.zeros(2),torch.zeros(2),torch.zeros(2))
    positive=stage_rewards(x,c,step_dt=.02,reward_form='positive')
    old=stage_rewards(x,c,step_dt=.02)
    for k,v in positive.rewards.items():
        assert (v>=0).all() and (v<=.02*c['weights'][k]+1.e-6).all()
        torch.testing.assert_close(v-old.rewards[k],torch.full((2,),.02*c['weights'][k]))
    assert sum(positive.rewards.values())[1] > sum(positive.rewards.values())[0]
    unsupported=stage_rewards(replace(x,persistent_contact=torch.tensor([[True,False],[False,True]])),c,step_dt=.02,reward_form='positive')
    assert (unsupported.rewards['rise']==0).all() and (unsupported.rewards['stand']==0).all()


def test_simple_config_rejects_legacy_regularizers():
    cfg=default_config()['standup']['reward'];validate_simple_rewards(cfg)
    cfg['regularization_weight']=.1
    with pytest.raises(ValueError):validate_simple_rewards(cfg)

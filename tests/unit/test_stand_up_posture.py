import pytest
import torch
from ref2act.envs.stand_up.posture import JOINTS, posture_credit, validate_posture
import json
from pathlib import Path
def default_config():
    # Explicit historical V13 fixture; native task defaults are V12.
    return json.loads((Path(__file__).parent/'fixtures/stand_up_v13.json').read_text())
from ref2act.envs.stand_up.simple_rewards import compose_rewards


def args():
    support=default_config()['standup']['support']
    return dict(position=torch.zeros(1,9),target=torch.zeros(1,9),sigma=torch.ones(9),
        support_ready=torch.ones(1),height=torch.ones(1),upright=torch.ones(1),
        foot_load=torch.ones(1),hand_load=torch.zeros(1),
        stage_cfg=support['stages'],support_cfg=support)


def test_only_waist_shoulders_elbows_selected():
    assert len(JOINTS)==9 and len(set(JOINTS))==9
    assert not any(any(s in n for s in ('hip','knee','ankle','wrist')) for n in JOINTS)


def test_waist_can_improve_while_arms_are_still_free():
    a=args();a['height'].fill_(.7);a['hand_load'].fill_(.3)
    a['position'][:,0]=1
    first,d=posture_credit(**a)
    assert d['waist_gate']==1 and d['arms_gate']==0
    a['position'][:,1:]=4
    same,_=posture_credit(**a)
    torch.testing.assert_close(first,same)
    a['position'][:,0]=0
    better,_=posture_credit(**a)
    assert better>first


@pytest.mark.parametrize('name,value',[('support_ready',0.),('height',.8),('upright',.6),('foot_load',.6),('hand_load',.03)])
def test_arms_gate_requires_all_readiness_conditions(name,value):
    a=args();a[name].fill_(value)
    _,d=posture_credit(**a)
    assert d['arms_gate']==0
    if name=='support_ready':assert d['waist_gate']==0


def test_unloading_and_arm_alignment_add_credit_without_removing_waist_credit():
    a=args();a['position'][:,1:]=2
    a['hand_load'].fill_(.03);base,_=posture_credit(**a)
    a['hand_load'].zero_();opened,_=posture_credit(**a)
    assert opened>=base
    a['position'][:,1:]=0;aligned,d=posture_credit(**a)
    assert aligned>opened and aligned==1 and d['arms_gate']==1
    a['hand_load'].fill_(.01);mid,d=posture_credit(**a)
    assert 0<d['arms_gate']<1 and base<mid<aligned


def test_posture_is_bounded_and_masks_follow_existing_failure_rules():
    a=args();credit,_=posture_credit(**a)
    cfg=default_config()['standup']['reward'];cfg.pop('hip_collision_weight',None);dt=.02
    pose=(credit*dt*cfg['posture']['weight']).repeat(3)
    pose[1]=float('nan')
    kw=dict(stage_terms={k:torch.zeros(3) for k in ('preparation','rise','stand')},
        action=torch.zeros(3,23),previous_action=torch.zeros(3,23),joint_position=torch.zeros(3,23),
        joint_limits=torch.tensor([-1.,1.]).repeat(3,23,1),finite=torch.tensor([True,False,True]),
        failed=torch.tensor([False,True,False]),settling=torch.tensor([False,False,True]),cfg=cfg,dt=dt)
    r=compose_rewards(**kw,posture_reward=pose)
    assert len(r)==7
    torch.testing.assert_close(r['posture'],torch.tensor([.003,0.,0.]))
    torch.testing.assert_close(sum(r.values()),torch.tensor([.003,-100.,0.]))
    with pytest.raises(ValueError):compose_rewards(**kw)


@pytest.mark.parametrize('key,value',[('weight',-1.),('waist_sigma',0.),('elbow_sigma',float('nan'))])
def test_bad_configuration_rejected(key,value):
    cfg=default_config()['standup']['reward']['posture'];cfg[key]=value
    with pytest.raises(ValueError):validate_posture(cfg)

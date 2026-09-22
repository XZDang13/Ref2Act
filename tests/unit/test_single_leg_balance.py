import math
from pathlib import Path
import numpy as np
import pytest
import torch
from ref2act.envs.single_leg_balance.defaults import default_config
from ref2act.envs.single_leg_balance.logic import capture_point,persistence,pulse_force,EpisodeRandom,reward_terms,validate_config
from ref2act.envs.single_leg_balance.observation import RobotHistory
from ref2act.envs.single_leg_balance.kinematics import left_sole_min_height
from ref2act.common.math import quat_from_euler_xyz


def test_registration_and_independent_defaults():
    import gymnasium as gym
    assert gym.spec('G1SingleLegBalance-v0').entry_point.endswith(':SingleLegBalanceEnv')
    a=default_config();a['initial']['joint_positions']['left_knee_joint']=99
    assert default_config()['initial']['joint_positions']['left_knee_joint']<1
    validate_config(default_config(),6,.02)


def test_capture_point_velocity_and_floor_translation():
    com=torch.tensor([[0.,0.,1.],[0.,0.,2.]])
    vel=torch.tensor([[.2,0.,0.],[.2,0.,0.]])
    cp=capture_point(com,vel,torch.tensor([0.,1.]))
    torch.testing.assert_close(cp[0],cp[1])
    assert cp[0,0]>0
    assert torch.isfinite(capture_point(com*0,vel,torch.zeros(2))).all()


def test_pulse_impulse_off_grid_and_mass_scaling():
    m=torch.tensor([20.,40.]);dv=torch.tensor([.3,.3]);theta=torch.tensor([0.,math.pi/2])
    start=torch.tensor([1.013,1.073]);imp=torch.zeros(2,3)
    for i in range(100):
        imp+=pulse_force(m,dv,theta,torch.full((2,),i*.02),start,.103,.02)*.02
    torch.testing.assert_close(imp,torch.tensor([[6.,0.,0.],[0.,12.,0.]]),atol=2e-5,rtol=1e-5)


def test_persistence_is_consecutive():
    timer=torch.zeros(2)
    for _ in range(8):timer=persistence(timer,torch.tensor([True,False]),.02)
    assert timer[0]<.18
    timer=persistence(timer,torch.tensor([True,True]),.02)
    assert timer[0]>=.18-1e-7
    timer=persistence(timer,torch.tensor([False,True]),.02)
    assert timer[0]==0 and timer[1]==.04


def test_history_partial_reset_and_applied_action():
    h=RobotHistory();mask=torch.ones(2,dtype=torch.bool)
    h.update(torch.zeros(2,55),torch.ones(2,23)*7,mask)
    h.update(torch.ones(2,55),torch.ones(2,23)*9,torch.zeros(2,dtype=torch.bool))
    out=h.update(torch.ones(2,55)*2,torch.ones(2,23)*11,torch.tensor([True,False]))
    assert out.shape==(2,312)
    assert (out[0,:220]==2).all() and (out[0,220:]==11).all()
    assert h.actions[1,:,0].tolist()==[7,7,9,11]


def test_rng_partial_reset_order_and_stream_independence():
    a=EpisodeRandom(4,'cpu',17);b=EpisodeRandom(4,'cpu',17)
    ids=torch.arange(4);a.advance(ids);b.advance(ids)
    a.advance(torch.tensor([1]));a.advance(torch.tensor([3]))
    b.advance(torch.tensor([3,1]))
    a.uniform(ids,55,1)
    torch.testing.assert_close(a.uniform(ids,3,2),b.uniform(ids,3,2))
    assert not torch.equal(a.uniform(ids,3,2),a.uniform(ids,3,1))


def test_balance_cannot_reward_hopping_or_second_support():
    c=default_config();cfg=dict(c['reward'],support_load=.2,target_height=.77)
    s=dict(load=torch.tensor([[1.,0.],[0.,0.],[.5,.5]]),swing_contact=torch.tensor([False,False,True]),
           bad_contact=torch.zeros(3,dtype=torch.bool),support_contact=torch.tensor([True,False,True]),
           clearance=torch.tensor([[0.,.12]]*3),balance_error=torch.zeros(3),upright=torch.ones(3),height=torch.ones(3)*.77)
    q=torch.zeros(3,23);terms=reward_terms(s,q,q,q,q,q,cfg)
    assert terms['balance'].tolist()==[1.,0.,0.]
    assert terms['swing'].tolist()==[1.,0.,0.]
    s['balance_error'][0]=.2
    assert reward_terms(s,q,q,q,q,q,cfg)['balance'][0]<.01


@pytest.mark.parametrize('bad',[-1,float('nan'),float('inf')])
def test_invalid_push(bad):
    c=default_config();c['push']['duration_s']=bad
    with pytest.raises(ValueError):validate_config(c,6,.02)


def test_reset_fk_matches_mujoco():
    mujoco=pytest.importorskip('mujoco')
    path=Path(__file__).resolve().parents[2]/'src/ref2act/assets/robots/g1/g1_23dof_rubber_hand.xml'
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    names=[model.joint(i).name for i in range(1,model.njnt)]
    pose=default_config()['initial']
    q=torch.tensor([[pose['joint_positions'][n] for n in names]])
    g=torch.Generator().manual_seed(7)
    q=q.expand(12,-1)+(torch.rand(12,23,generator=g)-.5)*.08
    angles=(torch.rand(12,3,generator=g)-.5)*.14
    quat=quat_from_euler_xyz(*angles.unbind(-1))
    heights=torch.ones(12)*pose['root_position'][2]
    actual=left_sole_min_height(q,names,quat,heights)
    points=np.array([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
    expected=[]
    for i in range(12):
        data.qpos[:]=np.r_[0,0,heights[i].item(),quat[i,[3,0,1,2]].numpy(),q[i].numpy()]
        mujoco.mj_forward(model,data)
        foot=model.body('left_ankle_roll_link').id
        world=points@data.xmat[foot].reshape(3,3).T+data.xpos[foot]
        expected.append(world[:,2].min()-.005)
    torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float32),atol=2e-6,rtol=1e-5)


def test_evaluation_censoring_and_no_isolated_maximum():
    from ref2act.envs.single_leg_balance.evaluation import summarize
    rows=[dict(delta_v=dv,direction=0.,survived=success,duration_s=6. if success else 1.,
               touchdowns=0,recovery_time_s=-1.) for dv,success in [(0.,True),(.1,False),(.2,True)]]
    result=summarize(rows,strengths=(0.,.1,.2),directions=(0.,))
    assert result['maximum_recoverable_delta_v_m_s']==0.
    assert result['cells'][0]['restricted_mean_survival_s']==6.
    assert result['cells'][0]['mean_recovery_time_s'] is None
    with pytest.raises(ValueError):summarize(rows,strengths=(0.,.1,.2),directions=(0.,1.))

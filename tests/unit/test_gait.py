import math
from dataclasses import replace
import pytest
import torch
from ref2act.envs.locomotion.gait import CommandGaitCfg, CommandGaitClock
from ref2act.envs.locomotion.task_rewards import FlatLocomotionRewardCfg, phase_gait_targets


def test_schedule_limits_and_sagittal_reflection():
    cfg=CommandGaitCfg()
    cmds=torch.tensor([[.3,0,0],[.6,0,0],[0,.3,0],[0,0,.5],[.6,.15,.25],[100.,100.,100.]])
    expected=torch.tensor([1.,1.25,1.25,1.25,1.25,1.25])
    assert torch.allclose(cfg.target_frequency(cmds),expected)
    assert torch.equal(cfg.target_frequency(cmds),cfg.target_frequency(cmds*torch.tensor([1.,-1.,-1.])))
    assert torch.equal(cfg.target_frequency(cmds),cfg.target_frequency(-cmds))


def test_clock_tracks_integrated_frequency_without_read_side_effects():
    clock=CommandGaitClock(CommandGaitCfg(),2,'cpu',step_dt=.02)
    ids=torch.arange(2);cmd=torch.tensor([[.3,0,0],[.6,0,0]])
    clock.reset(ids,cmd,torch.tensor([.1,.3]))
    accumulated=clock.angle.clone()
    for i in range(150):
        if i==37:cmd=cmd.flip(0)
        previous=clock.frequency.clone()
        clock.advance(cmd)
        accumulated += 2*math.pi*.02*clock.frequency
        assert ((clock.frequency-previous).abs() <= .25*(-math.expm1(-.02/.2))+1e-6).all()
        assert torch.allclose(clock.phase(),clock.phase())
        assert torch.allclose(torch.sin(clock.angle),torch.sin(accumulated),atol=3e-5)
    assert (clock.frequency>=1.).all() and (clock.frequency<=1.25).all()


def test_stop_finishes_swing_and_holds_grounded_phase_then_restarts():
    clock=CommandGaitClock(CommandGaitCfg(),40,'cpu',step_dt=.02)
    ids=torch.arange(40);cmd=torch.tensor([[.6,0,0]]).repeat(40,1)
    clock.reset(ids,cmd,torch.linspace(-math.pi,math.pi,40))
    cmd.zero_()
    for _ in range(60):clock.advance(cmd)
    phase=clock.phase().clone()
    height,contact=phase_gait_targets(phase,stance_ratio=.55,swing_height=.09)
    assert torch.allclose(height,torch.zeros_like(height),atol=1e-6)
    assert contact.all()
    for _ in range(30):clock.advance(cmd)
    assert torch.equal(clock.phase(),phase)
    cmd[:,0]=.6;clock.advance(cmd)
    delta=(clock.phase()-phase+math.pi).remainder(2*math.pi)-math.pi
    assert (delta>0).all() and (delta<=2*math.pi*1.25*.02+1e-6).all()


def test_subset_reset_preserves_other_clocks_and_standing_reset_is_grounded():
    clock=CommandGaitClock(CommandGaitCfg(),3,'cpu',step_dt=.02)
    cmds=torch.tensor([[.6,0,0],[.3,0,0],[0,0,0.]])
    clock.reset(torch.arange(3),cmds,torch.tensor([.3,.7,.9]))
    clock.advance(cmds);angle=clock.angle.clone();freq=clock.frequency.clone()
    clock.reset(torch.tensor([1]),cmds,torch.tensor([1.,2.,3.]))
    assert torch.equal(clock.angle[[0,2]],angle[[0,2]])
    assert torch.equal(clock.frequency[[0,2]],freq[[0,2]])
    assert torch.allclose(clock.angle[1],torch.tensor(2.))
    h,_=phase_gait_targets(clock.phase()[2:3],stance_ratio=.55,swing_height=.09)
    assert torch.equal(h,torch.zeros_like(h))


def test_contract_restores_fixed_and_adaptive_clocks():
    for cfg in [FlatLocomotionRewardCfg(gait_period=1.),FlatLocomotionRewardCfg(gait_period=.8),FlatLocomotionRewardCfg(gait_period=.8,gait_schedule=CommandGaitCfg())]:
        restored=FlatLocomotionRewardCfg.from_contract(cfg.contract())
        assert restored==cfg
        assert restored.contract()==cfg.contract()
    assert 'gait_schedule' not in FlatLocomotionRewardCfg().contract()


@pytest.mark.parametrize('kwargs',[{'min_frequency':0},{'max_frequency':float('nan')},{'forward_fast':.3},{'smoothing_time':0},{'stand_threshold':-1}])
def test_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):CommandGaitCfg(**kwargs)


def test_stop_reward_tracks_clock_until_both_feet_are_grounded():
    from types import SimpleNamespace
    from ref2act.envs.locomotion.task_rewards import phase_gait_signals
    cfg=FlatLocomotionRewardCfg(gait_schedule=CommandGaitCfg())
    phase=torch.tensor([[0.,-math.pi],[math.pi/2,-math.pi/2]])
    height,contact=phase_gait_targets(phase,stance_ratio=.55,swing_height=.09)
    inputs=SimpleNamespace(commands=torch.zeros(2,3),gait_phase=phase,feet_height=height,feet_contact=contact)
    signals=phase_gait_signals(inputs,cfg)
    assert torch.equal(signals['phase_contact_match'],torch.ones(2))
    assert torch.equal(signals['swing_foot_height_l2'],torch.zeros(2))
    assert torch.equal(signals['target_single_stance'],torch.tensor([1.,0.]))
    fixed=phase_gait_signals(inputs,FlatLocomotionRewardCfg())
    assert fixed['phase_contact_match'][0]==.5  # Legacy immediate stand contract.

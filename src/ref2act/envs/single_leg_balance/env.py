"""Native fixed-left balance. No command, motion runtime, phase or assistance."""
import math
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul
from ref2act.envs.base import LeggedRobotEnv, resolve_cfg_factory
from ref2act.isaac_compat import to_torch as T
from .contact import GroundContactReader
from .logic import EpisodeRandom, capture_point, persistence, pulse_force, reward_terms, safety_reward_terms, target_smoothness_terms, validate_config
from .observation import RobotHistory, proprioception, noisy_frame


class SingleLegBalanceEnv(LeggedRobotEnv):
    def __init__(self,cfg=None,render_mode=None,cfg_factory=None,**kwargs):
        cfg=cfg or resolve_cfg_factory(cfg_factory or 'ref2act.robots.g1:G1SingleLegBalanceEnvCfg')
        validate_config(cfg.task,cfg.episode_length_s,cfg.sim.dt*cfg.decimation)
        if cfg.action.mode!='offset' or cfg.pair_action_history!='applied_action':
            raise ValueError('V1 requires offset actions and applied-action history')
        super().__init__(cfg,render_mode,**kwargs)
        self.task=cfg.task
        self.history=RobotHistory()
        self._reset_mask=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self._rng=EpisodeRandom(self.num_envs,self.device,cfg.seed or 0)
        self._ids=torch.arange(self.num_envs,device=self.device)
        names=self.robot.body_names
        self._feet=[names.index(s+'_ankle_roll_link') for s in ('left','right')]
        self._torso=names.index('torso_link')
        self._push_body=torch.tensor([names.index(self.task['push']['body'])],device=self.device)
        # Every non-foot articulation body is monitored; wrist/hip contacts cannot hide.
        self._contact_names=[names[i] for i in self._feet]+[n for n in names if n not in [names[i] for i in self._feet]]
        self._contacts=GroundContactReader(self,self._contact_names)
        self._force_sum=torch.zeros(self.num_envs,len(names),3,device=self.device)
        self._force_peak=torch.zeros(self.num_envs,len(names),device=self.device)
        self._force_last=torch.zeros_like(self._force_sum)
        self._contact_samples=0
        self._timers=torch.zeros(self.num_envs,3,device=self.device)
        self._start=torch.zeros(self.num_envs,device=self.device)
        self._dv=torch.zeros_like(self._start);self._direction=torch.zeros_like(self._start)
        self._impulse=torch.zeros(self.num_envs,3,device=self.device)
        self._force=torch.zeros_like(self._impulse)
        self._recovery_hold=torch.zeros_like(self._start)
        self._recovery_time=torch.full_like(self._start,-1.)
        self._touchdowns=torch.zeros_like(self._start)
        self._previous_touch=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self._target=T(self.robot.data.default_joint_pos).clone()
        pose=self.task['initial']
        if set(pose['joint_positions'])!=set(self.robot.joint_names):
            raise ValueError('Reset pose must name exactly the 23 robot joints')
        for name,value in pose['joint_positions'].items(): self._target[:,self.robot.joint_names.index(name)]=value
        limits=T(self.robot.data.joint_pos_limits)
        if not torch.isfinite(self._target).all() or ((self._target<limits[...,0])|(self._target>limits[...,1])).any():
            raise ValueError('Reset pose exceeds live joint limits')
        self._state=None
        self._target_previous=self._target.clone()
        self._target_previous2=self._target.clone()
        self._smooth_steps=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self._acc_sq_sum=torch.zeros_like(self._target)

    def _setup_scene(self):
        super()._setup_scene()
        if self.device!='cpu': self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

    def _pre_physics_step(self,actions):
        if not torch.isfinite(actions).all(): raise RuntimeError('Nonfinite policy action')
        for key in ('final_critic_mask','final_critic_observation','single_leg_outcomes'):
            self.extras.pop(key,None)
        if self.task.get('version',1)>=3:
            self._target_previous2.copy_(self._target_previous)
            self._target_previous.copy_(self.action_processor.target_joint_position)
            self._smooth_steps+=1
            self._acc_sq_sum.zero_()
        self.action_processor.pre_process_action(self._policy_to_sim_order(actions))
        self._force_sum.zero_();self._force_peak.zero_();self._contact_samples=0
        self._state=None
        p=self.task['push']
        self._force=pulse_force(T(self.robot.data.body_mass).sum(-1),self._dv,self._direction,
                                self.episode_length_buf.float()*self.step_dt,self._start,p['duration_s'],self.step_dt)
        self._impulse+=self._force*self.step_dt
        self._write_force(self._force,self._ids)

    def _write_force(self,force,ids):
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force[:,None],torques=torch.zeros_like(force[:,None]),
            body_ids=self._push_body,env_ids=ids,is_global=True)

    def _sample_contacts(self):
        if self.task.get('version',1)>=3:
            self._acc_sq_sum+=T(self.robot.data.joint_acc).square()
        f=self._contacts.read()
        self._force_sum+=f;self._force_last.copy_(f)
        self._force_peak=torch.maximum(self._force_peak,f.norm(dim=-1))

    def _apply_action(self):
        if self._contact_samples: self._sample_contacts()
        self._contact_samples+=1
        super()._apply_action()

    def _geometry(self):
        d=self.robot.data
        mass=T(d.body_mass);total=mass.sum(-1)
        com=(T(d.body_com_pos_w)*mass[...,None]).sum(1)/total[:,None]
        vel=(T(d.body_com_lin_vel_w)*mass[...,None]).sum(1)/total[:,None]
        feet=T(d.body_link_pos_w)[:,self._feet]
        quats=T(d.body_link_quat_w)[:,self._feet]
        corners=feet.new_tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])
        points=quat_apply(quats[:,:,None].expand(-1,-1,4,-1).reshape(-1,4),
                          corners[None,None].expand(self.num_envs,2,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,4,3)+feet[:,:,None]
        points[...,2]-=.005
        origin=T(self.scene.env_origins)
        clearance=(points[...,2]-origin[:,None,None,2]).amin(-1)
        support=points[:,0].mean(1)
        cp=capture_point(com,vel,origin[:,2])
        upright=quat_apply(T(d.body_link_quat_w)[:,self._torso],feet.new_tensor([0.,0.,1.]).expand(self.num_envs,-1))[:,2]
        return dict(com=com,com_vel=vel,feet=feet,clearance=clearance,support=support,
                    balance_error=(cp-support[:,:2]).norm(dim=-1),upright=upright,
                    height=T(d.body_link_pos_w)[:,self.root_body_index,2]-origin[:,2],weight=total*9.81)

    def _measure(self):
        if self._state is not None: return self._state
        if self._contact_samples!=self.cfg.decimation: raise RuntimeError('Missing contact substep samples')
        self._sample_contacts()
        s=self._geometry();c=self.task['contact']
        # Average load plus final sample detect support loss; peak detects brief touches.
        mean=self._force_sum/self.cfg.decimation
        s['load']=mean[:,:2,2].clamp_min(0)/s['weight'][:,None]
        s['support_contact']=(self._force_last[:,0,2]>c['threshold_n'])&(s['load'][:,0]>=c['support_load'])
        s['swing_contact']=self._force_peak[:,1]>c['threshold_n']
        s['bad_contact']=(self._force_peak[:,2:]>c['threshold_n']).any(-1)
        self._state=s
        return s

    def _critic(self,s):
        d=self.robot.data;origin=T(self.scene.env_origins);root=T(d.body_link_pos_w)[:,self.root_body_index]
        terms=[proprioception(self),self._sim_to_policy_order(self.action_processor.applied_action),
               self._sim_to_policy_order(self.action_processor.previous_applied_action),
               T(d.body_link_lin_vel_w)[:,self.root_body_index],(s['feet']-root[:,None]).flatten(1),
               T(d.body_link_lin_vel_w)[:,self._feet].flatten(1),s['com']-root,s['com_vel'],
               s['clearance'],s['load'],s['height'][:,None],s['upright'][:,None],
               self._timers,self._force/s['weight'][:,None],
               (self._start-self.episode_length_buf*self.step_dt)[:,None],self._dv[:,None],
               torch.stack((self._direction.cos(),self._direction.sin()),-1),
               (1-self.episode_length_buf.float()/self.max_episode_length)[:,None],
               self._recovery_hold[:,None]]
        value=torch.cat(terms,-1).float()
        if value.shape[1]!=self.cfg.critic_observation_space: raise RuntimeError(f'Critic layout changed: {value.shape}')
        return value

    def _get_observations(self):
        s=self._geometry()
        # Reset rows must not inherit old-episode contact forces or derived critic state.
        s['load']=self._force_sum[:,:2,2].clamp_min(0)/self.cfg.decimation/s['weight'][:,None]
        actor=self.history.update(noisy_frame(proprioception(self),self.task['actor_noise']),
                                  self._sim_to_policy_order(self.action_processor.applied_action),self._reset_mask)
        critic=self._critic(s)
        self._reset_mask.zero_()
        if not torch.isfinite(critic).all(): raise RuntimeError('Nonfinite single-leg critic observation')
        return {'policy':actor,'critic':critic}

    def _get_dones(self):
        s=self._measure();c=self.task['contact'];t=self.task['termination']
        active=torch.stack((s['swing_contact'],~s['support_contact'],s['bad_contact']),-1)
        self._timers=persistence(self._timers,active,self.step_dt)
        self._touchdowns+=(s['swing_contact']&~self._previous_touch).float()
        self._previous_touch.copy_(s['swing_contact'])
        finite=torch.isfinite(self._critic(s)).all(-1)
        terminated=(~finite)|(s['height']<t['height_ratio']*self.task['initial']['root_position'][2])|(s['upright']<t['upright'])
        terminated|=(self._timers>=self._timers.new_tensor([c['grace_s'],c['grace_s'],c['bad_grace_s']])-1e-7).any(-1)
        timeout=self.episode_length_buf>=self.max_episode_length
        elapsed=self.episode_length_buf*self.step_dt
        after=elapsed>=self._start+self.task['push']['duration_s']
        r=self.task['recovery']
        stable=after&s['support_contact']&~s['swing_contact']&~s['bad_contact']
        stable&=(s['balance_error']<=r['balance_error'])&(s['com_vel'][:,:2].norm(dim=-1)<=r['com_speed'])
        stable&=(s['clearance'][:,1]>=self.task['reward']['swing_range'][0])&(s['upright']>=.9)&~terminated
        self._recovery_hold=persistence(self._recovery_hold,stable,self.step_dt)
        recovered=(self._recovery_time<0)&(self._recovery_hold>=r['hold_s']-1e-7)
        self._recovery_time[recovered]=(elapsed-self._start-self.task['push']['duration_s'])[recovered]
        boot=timeout&~terminated
        self.extras['final_critic_mask']=boot.clone()
        self.extras['final_critic_observation']=self._critic(s)[boot].clone()
        done=terminated|timeout
        self.extras['single_leg_outcomes']={
            'env_ids':self._ids[done].clone(),'survived':(timeout&~terminated)[done].clone(),
            'episode_id':self._rng.episode[done].clone(),'push_start_s':self._start[done].clone(),
            'push_completed':after[done].clone(),'failed':terminated[done].clone(),
            'duration_s':elapsed[done].clone(),'touchdowns':self._touchdowns[done].clone(),
            'recovery_time_s':self._recovery_time[done].clone(),'delta_v':self._dv[done].clone(),
            'direction':self._direction[done].clone(),'applied_impulse_ns':self._impulse[done].clone()}
        return terminated,timeout

    def _get_rewards(self):
        s=self._measure();cfg=dict(self.task['reward'],support_load=self.task['contact']['support_load'],
                                  target_height=self.task['initial']['root_position'][2])
        terms=reward_terms(s,T(self.robot.data.joint_pos),self._target,T(self.robot.data.joint_vel),
                           self.action_processor.applied_action,self.action_processor.previous_applied_action,cfg)
        if self.task.get('version',1)>=2:
            p=self.action_processor;d=self.robot.data
            raw=p.applied_action*p.scale+p.offset+p.offset_noise
            acceleration=T(d.joint_acc)
            if self.task.get('version',1)>=3:
                acceleration=(self._acc_sq_sum/self.cfg.decimation).sqrt()
            terms.update(safety_reward_terms(T(d.joint_pos),T(d.soft_joint_pos_limits),
                acceleration,T(d.applied_torque),raw,p.target_joint_position))
        if self.task.get('version',1)>=3:
            terms.update(target_smoothness_terms(self.action_processor.target_joint_position,
                self._target_previous,self._target_previous2,self._smooth_steps>=2))
        reward=sum(cfg['weights'][k]*v for k,v in terms.items())*self.step_dt
        # Nonfinite states terminate; no NaNs reach PPO.
        reward=torch.where(torch.isfinite(reward),reward,torch.zeros_like(reward))
        log=self.extras.setdefault('log',{})
        for k,v in terms.items():
            log['SingleLeg/reward_'+k]=torch.nan_to_num(v).mean().detach()
            log['SingleLeg/weighted_'+k]=torch.nan_to_num(v*cfg['weights'][k]*self.step_dt).mean().detach()
        log['SingleLeg/balance_error_m']=torch.nan_to_num(s['balance_error']).mean().detach()
        log['SingleLeg/reward_total']=reward.mean().detach()
        return reward

    def _reset_idx(self,env_ids):
        ids=self._normalize_env_ids(env_ids)
        self.robot.reset(ids)
        DirectRLEnv._reset_idx(self,ids)
        self.action_processor.reset_action_buffer(ids)
        self._rng.advance(ids)
        noise=self.task['reset_noise'];draw=self._rng.uniform(ids,55,stream=1)*2-1
        q=self._target[ids]+draw[:,:23]*noise['joint_position']
        limits=T(self.robot.data.joint_pos_limits)[ids]
        q=q.clamp(limits[...,0],limits[...,1]);v=draw[:,23:46]*noise['joint_velocity']
        pose=self.task['initial'];root=T(self.robot.data.default_root_state)[ids].clone()
        root[:,:3]=root.new_tensor(pose['root_position'])+T(self.scene.env_origins)[ids]
        angles=draw[:,46:48]*noise['roll_pitch']
        root[:,3:7]=quat_mul(quat_from_euler_xyz(angles[:,0],angles[:,1],torch.zeros_like(angles[:,0])),
                             root.new_tensor(pose['root_quaternion']).expand(len(ids),-1))
        root[:,7:]=0.;root[:,7:9]=draw[:,48:50]*noise['linear_velocity']
        root[:,10:13]=draw[:,50:53]*noise['angular_velocity']
        # Align the perturbed sole using an analytic left-leg FK, without stepping
        # other environments or inserting a settling/controller phase.
        from .kinematics import left_sole_min_height
        root[:,2]+=.001-left_sole_min_height(q,self.robot.joint_names,root[:,3:7],root[:,2]-T(self.scene.env_origins)[ids,2])
        self.robot.write_root_link_pose_to_sim_index(root_pose=root[:,:7],env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root[:,7:],env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=v,env_ids=ids)
        if self.cfg.action.latency_range not in (None,(0,0),[0,0]):
            self.action_processor.set_random_delays(ids,self.cfg.action.latency_range)
        else:
            self.action_processor.delays[ids]=0
        # Seed the physical target with q0 and record its actual normalized action.
        p=self.action_processor
        initial=(q-p.offset[ids]-p.offset_noise[ids])/p.scale
        p.applied_action[ids]=initial;p.previous_applied_action[ids]=initial
        p._update_target_joint_position(ids)
        self._target_previous[ids]=p.target_joint_position[ids]
        self._target_previous2[ids]=p.target_joint_position[ids]
        self._smooth_steps[ids]=0
        self._acc_sq_sum[ids]=0
        # DequeBuffer reset zeros delayed commands; fill all selected history slots.
        p.action_buffer.reset(ids,values=initial)
        push=self.task['push'];u=self._rng.uniform(ids,3,stream=2)
        self._start[ids]=push['time_range'][0]+u[:,0]*(push['time_range'][1]-push['time_range'][0])
        self._dv[ids]=push['delta_v_range'][0]+u[:,1]*(push['delta_v_range'][1]-push['delta_v_range'][0])
        self._direction[ids]=u[:,2]*(2*math.pi)
        if push['evaluation_delta_v'] is not None:self._dv[ids]=push['evaluation_delta_v']
        if push['evaluation_direction'] is not None:self._direction[ids]=push['evaluation_direction']
        for value in (self._timers,self._force_sum,self._force_peak,self._force_last,self._force,
                      self._impulse,self._recovery_hold,self._touchdowns,self._previous_touch):value[ids]=0
        self._recovery_time[ids]=-1.;self._reset_mask[ids]=True
        self._write_force(self._force[ids],ids)

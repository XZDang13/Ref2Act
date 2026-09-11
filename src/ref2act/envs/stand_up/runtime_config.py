"""Stand-up simulator settings, independent of PAIR and locomotion."""
from copy import deepcopy


def configure_runtime(cfg, config, *, device, num_envs, seed, **_compat):
    from ref2act.robots._articulation_shared import G1_CFG
    cfg.seed = int(seed)
    cfg.sim.device = str(device)
    cfg.scene.num_envs = int(num_envs)
    runtime = config.get('ref2act', {})
    profile = runtime.get('robot_profile', 'legacy')
    if profile not in ('legacy', 'current_locomotion'):
        raise ValueError('Unknown robot_profile')
    cfg.pair_robot_profile = profile
    cfg.robot.init_state.joint_pos = deepcopy(G1_CFG.init_state.joint_pos)
    cfg.robot.init_state.pos = tuple(G1_CFG.init_state.pos)
    if profile == 'current_locomotion':
        cfg.robot.init_state.joint_pos.update({'.*_hip_pitch_joint': -.15,
            '.*_knee_joint': .35, '.*_ankle_pitch_joint': -.20})
    cfg.pair_action_history = config.get('pair_collection', {}).get('action_history', 'q_target')
    if cfg.pair_action_history not in ('q_target', 'applied_action'):
        raise ValueError('Unknown action history contract')
    for name in ('gpu_max_rigid_patch_count', 'gpu_max_rigid_contact_count'):
        if name in runtime:
            from isaaclab_physx.physics import PhysxCfg
            value = int(runtime[name])
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            if cfg.sim.physics is None:
                cfg.sim.physics = PhysxCfg()
            setattr(cfg.sim.physics, name, value)
    for name in ('push_robot', 'rand_leg_joint_friction', 'rand_arm_joint_friction', 'rand_torso_joint_friction'):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    if runtime.get('disable_startup_randomization', False):
        for name in ('physics_material', 'rand_robot_mass', 'rand_base_com', 'rand_contact_offsets',
                     'rand_leg_joint_armature', 'rand_arm_joint_armature', 'rand_torso_joint_armature',
                     'rand_robot_joint_stiffness_and_damping'):
            if hasattr(cfg.events, name):
                setattr(cfg.events, name, None)
    lower, upper = map(int, runtime.get('action_delay_range', [0, 0]))
    if lower < 0 or upper < lower:
        raise ValueError('action_delay_range must be ordered and non-negative')
    cfg.action.buffer_length = upper + 1
    cfg.action.latency_range = (lower, upper)
    recovery = config.get('pair_collection', {}).get('fall_recovery', {})
    reward = recovery.get('reward', {})
    cfg.pair_recovery_shoulder_body_names = tuple(recovery.get('shoulder_body_names',
        ('left_shoulder_roll_link', 'right_shoulder_roll_link')))
    cfg.pair_recovery_minimum_upright = float(reward.get('minimum_upright_projection', 0.))
    cfg.pair_recovery_target_upright = float(reward.get('target_upright_projection', .95))
    if len(cfg.pair_recovery_shoulder_body_names) != 2 or cfg.pair_recovery_target_upright <= cfg.pair_recovery_minimum_upright:
        raise ValueError('Invalid standing reference geometry')
    cfg.pair_fall_recovery_upright = float(recovery.get('recovered_upright_projection', .85))
    cfg.pair_recovery_height_sensor_enabled = False
    cfg.actor_observation_noise = deepcopy(config['standup'].get('actor_observation_noise', {}))
    cfg.observation_space = cfg.policy_observation_space = 312
    cfg.critic_observation_space = 181
    return cfg


def configure_standup_cfg(cfg, config, *, device, num_envs, seed):
    from .config import configure_task
    configure_runtime(cfg, config, device=device, num_envs=num_envs, seed=seed)
    return configure_task(cfg, config, device=device, num_envs=num_envs, seed=seed)

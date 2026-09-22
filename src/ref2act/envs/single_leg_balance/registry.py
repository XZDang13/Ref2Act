"""Simulator-independent Gym registration."""
ENV_ID='G1SingleLegBalance-v0'


def register_envs():
    import gymnasium as gym
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID,entry_point='ref2act.envs.single_leg_balance.env:SingleLegBalanceEnv',
                     kwargs={'cfg_factory':'ref2act.robots.g1:G1SingleLegBalanceEnvCfg'},
                     disable_env_checker=True)

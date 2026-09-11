"""Gym entry point; registration does not import the simulator."""
ENV_ID = "G1StandUp-v0"

def register_envs():
    import gymnasium as gym
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="ref2act.envs.stand_up.env:StandUpEnv",
                     kwargs={"cfg_factory": "ref2act.robots.g1:G1StandUpEnvCfg"},
                     disable_env_checker=True)

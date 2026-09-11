from .registry import ENV_ID, register_envs

__all__ = ["StandUpEnv", "ENV_ID", "register_envs"]

def __getattr__(name):
    if name == "StandUpEnv":
        from .env import StandUpEnv
        return StandUpEnv
    raise AttributeError(name)

from __future__ import annotations


def register_envs() -> None:
    import gymnasium

    env_id = "G1FlatStandUp-v0"
    if env_id in gymnasium.registry:
        return
    gymnasium.register(
        id=env_id,
        entry_point="ref2act.envs.stand_up.env:StandUpEnv",
        kwargs={"cfg_factory": "ref2act.robots.g1:G1FlatStandUpEnvCfg"},
        disable_env_checker=True,
    )


__all__ = ["register_envs"]

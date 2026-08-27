from __future__ import annotations


def register_envs() -> None:
    import gymnasium

    specs = (
        ("G1FlatLocomotion-v0", "ref2act.robots.g1:G1FlatLocomotionEnvCfg"),
        ("G1SlopeLocomotion-v0", "ref2act.robots.g1:G1SlopeLocomotionEnvCfg"),
        ("G1UnevenLocomotion-v0", "ref2act.robots.g1:G1UnevenLocomotionEnvCfg"),
        ("G1MixedTerrainLocomotion-v0", "ref2act.robots.g1:G1MixedTerrainLocomotionEnvCfg"),
    )
    for env_id, cfg_factory in specs:
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=env_id,
            entry_point="ref2act.envs.locomotion.env:LocomotionEnv",
            kwargs={"cfg_factory": cfg_factory},
            disable_env_checker=True,
        )


__all__ = ["register_envs"]

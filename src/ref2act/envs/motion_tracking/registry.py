from __future__ import annotations


def register_envs() -> None:
    import gymnasium

    specs = (
        ("G1MotionTracking-v0", "ref2act.robots.g1:G1MotionTrackingEnvCfg"),
        ("G1MotionTrackingRough-v0", "ref2act.robots.g1:G1MotionTrackingRoughEnvCfg"),
    )

    for env_id, cfg_factory in specs:
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=env_id,
            entry_point="ref2act.envs.motion_tracking.env:MotionTrackingEnv",
            kwargs={"cfg_factory": cfg_factory},
        )

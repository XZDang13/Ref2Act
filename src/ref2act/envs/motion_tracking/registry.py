from __future__ import annotations


def register_envs() -> None:
    import gymnasium

    specs = (
        ("G1MotionTracking-v0", "ref2act.robots.g1:G1MotionTrackingEnvCfg"),
        ("PiPlusMotionTracking-v0", "ref2act.robots.pi_plus:PiPlusMotionTrackingEnvCfg"),
        ("G1MotionTrackingRough-v0", "ref2act.robots.g1:G1MotionTrackingRoughEnvCfg"),
        ("PiPlusMotionTrackingRough-v0", "ref2act.robots.pi_plus:PiPlusMotionTrackingRoughEnvCfg"),
    )

    for env_id, cfg_factory in specs:
        if env_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=env_id,
            entry_point="ref2act.envs.motion_tracking.env:MotionTrackingEnv",
            kwargs={"cfg_factory": cfg_factory},
        )

import gymnasium

gymnasium.register(
    id="G1MotionTracking-v0",
    entry_point=f"{__name__}.env:G1MotionTrackingEnv"
)
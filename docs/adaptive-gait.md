# Command-adaptive G1 locomotion gait

The fixed 0.8 s baseline is Ref2Act commit `7e9df21`. The new G1 default
uses `CommandGaitCfg` with periods limited to 0.8–1.0 s. This is a candidate
schedule, not a claim that adaptive cadence has already outperformed 0.8 s.
The 9 cm swing target, 0.55 stance ratio, tracking weights and actor dimensions
remain unchanged. Motion tracking is unaffected.

## Schedule

Frequency is full cycles per foot (alternating steps/s = 2 × frequency):

```
demand = clip(max((abs(vx_cmd)-0.3)/0.3,
                  abs(vy_cmd)/0.3,
                  abs(wz_cmd)/0.5), 0, 1)
f_target = 1.0 + 0.25 * demand
f += (1-exp(-dt/0.2)) * (f_target-f)
phase = wrap(phase + 2*pi*dt*f)
```

Examples at steady command: vx=0.3 → 1.0 s; vx=0.45 → 0.889 s;
vx=0.6 → 0.8 s; pure |wz|=0.5 → 0.8 s; (0.6,0.15,0.25) → 0.8 s.
The function is symmetric under left/right command reflection. It uses only
commanded velocities, never measured or estimated linear velocity.

Zero-command norm below 0.01 requests a stop. The clock finishes the current
swing and stops at the next double-support region. It does not snap phases to
another angle. The gait reward honors this settling swing; fixed-clock rewards
retain their original immediate standing gate. A standing reset starts with
both feet grounded; moving resets retain randomized phase. Partial resets
only reset the selected environments.

## Training and deployment order

`ref2act.envs.locomotion.gait.CommandGaitClock` is a Torch-only clock with no
Isaac simulator dependencies in the module. Use the same parameters, dt, and
update order in a deployment runner. This repository change does not install
or validate a hardware runner.

1. On reset, call `clock.reset(ids, commands, initial_phase)`; initialize history.
2. Read `clock.phase()` and encode `sinL, cosL, sinR, cosR` for the actor.
3. At each control tick, accept the next command, then call `clock.advance`
   exactly once, as training does in `_pre_physics_step`; execute the action.
4. The reward and next observation read the same updated phase. Reads never
   advance time. Preserve frequency and phase between ticks; command changes
   do not reset either state.

A deployment runner with different command/observation timing must reproduce
this ordering, including the one-step command handoff. Stop request is not an
emergency-stop function. IMU 6D orientation and applied-action history keep
their existing definitions. Left/right phase offsets remain half a cycle.

The reward contract records the schedule. `gait_period=0.8` is only the fixed
fallback when `gait_schedule=None`. `FlatLocomotionRewardCfg.from_contract`
restores old fixed 1.0/0.8 s contracts or adaptive contracts. IRASP viewer and
resume entry points restore these recorded semantics before creating an env.
Do not replay an old fixed-period policy using current adaptive defaults.
Fresh training uses the current adaptive configuration; no full training run
is started by this implementation change.

Logs expose `Gait/frequency_hz_mean`, `Gait/target_frequency_hz_mean` and
`Gait/clock_advancing_fraction`. Frequency logs describe the oscillator's
cadence while active; a parked clock advances at zero despite its stored
frequency remaining within bounds.

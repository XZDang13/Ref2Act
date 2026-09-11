# G1 stand-up

`G1StandUp-v0` is a native Ref2Act task, alongside locomotion and motion tracking.
`StandUpEnv` inherits `LeggedRobotEnv` directly. It has no IRASP dependency,
velocity command generator, gait clock, or locomotion observation/reward/reset path.

After launching Isaac Lab:

```python
import gymnasium as gym
import ref2act
from ref2act.robots.g1 import G1StandUpEnvCfg

cfg = G1StandUpEnvCfg()
cfg.scene.num_envs = 4096
cfg.sim.device = "cuda:0"
env = gym.make("G1StandUp-v0", cfg=cfg)
obs, extras = env.reset()
# obs["policy"]: [N, 312]; obs["critic"]: [N, 181]
```

The configuration factory also permits `gym.make("G1StandUp-v0")`. Defaults
are independent Python data in `envs/stand_up/defaults.py`; no downstream YAML
or trainer package is required. To customize task rewards before constructing
an environment, modify a copy returned by `default_config()` and apply it using
`runtime_config.configure_standup_cfg(cfg, settings, device=..., num_envs=..., seed=...)`.

## Runtime and observation behavior

- Fixed supine reset; 0.6 s settling; asynchronous episode deadlines and stagnation cutoffs.
- Bilateral foot placement, stance separation, hand support, weight transfer and standing rewards.
- Ground-only foot/hand contact measurements, self-collision enabled.
- Actor: four 55-dimensional proprioception frames followed by four 23-dimensional
  absolute joint-target frames, oldest first (312 values). Only actor proprioception receives noise.
- Critic: current privileged state (181 values). Partial resets repeat the new actor
  frame and current physical joint target; old-episode history is removed.
- Startup domain randomization enabled; action delay remains `[0, 0]`.
- Assistance targets `torso_link`. Its initial ratio is zero; a trainer explicitly
  updates the curriculum through `set_standup_assistance_ratio`.
- Timeout metadata: `final_critic_mask` has N rows; `final_critic_observation`
  contains only the selected pre-reset rows, in mask order.

For compatibility, internal `pair_standup_*` / `pair_recovery_*` field names and
historical reward modes are retained. They do not introduce a PAIR dependency.
The task's discount/bootstrap settings are consumed only by potential shaping;
Ref2Act does not select an optimizer or run PPO.

## Downstream compatibility

IRASP's old `G1FlatPAIRStandUp-v0` registration points at this environment.
Its observation wrapper and trainer retain historical normalization, actor,
checkpoint and RLAlg GAE contracts. The wrapper disables native observation
assembly to avoid computing and drawing noise twice. Training and viewing scripts
now instantiate `G1StandUpEnvCfg`.

Reset now performs simulator/event/action-buffer housekeeping followed by one
supine state write. There is no intermediate locomotion pose write. The original migration preserved V12 parameters. V13 changes the reward aggregation as described below. The subsequent runtime optimization is
documented below; measured rollout throughput is separate from full-training throughput.

## Static parameter cache

Stand-up enables `cfg.cache_static_physics` by default. The cache belongs to one
articulation view and is installed after startup domain randomization. It stores
only native mass and **body-local** CoM buffers. World CoM, velocities, collision
and force data continue updating at the normal simulation cadence.

`set_masses`, `set_coms` and `set_inertias` invalidate the cache, including partial
or failed writes. Invalidation also clears assistance's total mass and relevant
Isaac Lab derived-state timestamps, so runtime randomization cannot use stale
mass or CoM values. Root-view replacement rebuilds the cache; closing the
environment restores original view methods. Edits made directly through USD or
an independent backend handle must call `env.unwrapped.invalidate_static_physics()`.
Public articulation setters are covered automatically. Borrowed getter buffers
must not be modified without completing the corresponding setter operation.

Set `cfg.cache_static_physics = False` before construction for a diagnostic
uncached run. At runtime the cache's `enabled` property can be toggled for an A/B
check. Toggling invalidates stored parameters and their derived task state.

## V13 reward aggregation

The optional V13 configuration uses `simple_rewards.py`: positive preparation/rise/stand
credits with weights 1/3/5, action-rate cost 0.05, joint-limit cost 10 (soft
boundary 0.9), and a failure event cost of 100. Continuous terms are multiplied
by control dt. These are six base terms, each logged under `Rewards/`.
The V13 configuration also enables an optional seventh `Rewards/posture` term
with weight 0.15. It rewards default waist-yaw alignment during supported rising
and default shoulder/elbow alignment only near standing, with sufficient actual
foot load and unloaded hands. Legs and wrists are excluded. The two contributions
are additive and bounded; at dt 0.02 the posture term is at most 0.003.
`Posture/` diagnostics expose each gate and alignment quality. Historical
configurations without `posture` retain the six base terms.
Stage credit geometry and bilateral support gates retain the V12 implementation.

There are no additional completion/hold bonuses, torque/power/velocity costs,
action-acceleration costs, or separate hip-collision reward in this path.
`legacy_rewards.py` preserves historical reward configurations for replay.
V13 settings are experimental; use an explicit V13 configuration to
replay its original reward function. Simplifying the positive reward baseline
does not alone resolve posture or alter the support-stage discovery conditions.


### Supported rise and hold revision

The experimental V13 configuration enables `stages.rise_hold_v2`. Saved configurations omitting it
retain the historical stage formulas. The seven additive terms are unchanged.
Torso lean is measured from the torso up axis along average sole heel-to-toe
heading. A rational angular score supplies feedback even for backward lean.
The target moves from 15 degrees forward to upright with height, foot takeover,
and CoM support quality; no height threshold converts it into constant credit.

Rise credit mixes height (0.50), torso alignment (0.25), support margin quality
(0.15), and foot takeover (0.10), behind the existing bilateral readiness gate.
Stand credit combines pose completion (60%) with autonomous quiet hold (40%).
Motion reduces only the hold contribution. Hold requires both feet loaded,
actual total foot load, hands unloaded, and CoM margin. Margin is the signed
inward distance to the measured near-ground sole-point convex hull; 25 mm earns
full balance quality. This geometric estimate is not a measured CoP.

Success additionally checks nonnegative margin, each foot carrying at least
0.2 mg, unloaded hands, torso lean within 10 degrees, torso uprightness, and
valid stance. Existing height, velocity and 0.5-second persistence checks remain.
Diagnostics distinguish torso target/alignment, signed margin, standing pose,
and autonomous hold. The added computation is batched on device.


### Foot discovery revision

Defaults enable `stages.foot_discovery_v2`. Preparation's internal foot score is
0.40 hip-distance quality + 0.30 foot-lowering quality + 0.20 signed sole
orientation + 0.10 persistent usable contact. Each component uses the existing
bilateral mean/min aggregation; stance mixing and the total preparation budget
remain unchanged. The lowering score uses mean absolute collision-sole-point
height with the existing 1 cm tolerance and 15 cm scale. Orientation uses
(1 + world sole normal z) / 2 independently of contact and height.
Strict whole-sole planting still controls placement/support readiness.
`Stages/foot_lowering_credit` and `Stages/foot_orientation_credit` expose the
new components. Historical configurations omitting the flag retain the old
formula; saved running configurations are not changed.


## Native V12 baseline (2026-09-11)

`G1StandUpEnvCfg()` and `G1StandUp-v0` now default to V12, including its
complete staged reward configuration, fixed-supine reset, actor noise, single-frame
privileged critic, domain randomization and torso assistance. Stage weights are
2.5/2.5/2.0. V13 reward flags and mixed standing resets are not enabled. Native
performance optimizations remain. No IRASP import or YAML dependency is added.

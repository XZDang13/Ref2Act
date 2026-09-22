# G1 Single-Leg Balance under Disturbance

`G1SingleLegBalance-v0` is a native `LeggedRobotEnv` subclass. The fixed task is
left-foot support with a raised right foot, starting near a single-leg equilibrium.
It has no goal, command, reference trajectory, phase, support-leg ID, assistance,
transition stages, or dependency on IRASP. `q*` defines reset and weak regularization
only. It is never an actor observation.

## Usage

Launch Isaac Lab before importing the environment configuration:

```python
import gymnasium as gym
import ref2act
from ref2act.robots.g1 import G1SingleLegBalanceEnvCfg

cfg = G1SingleLegBalanceEnvCfg()
cfg.scene.num_envs = 64
cfg.sim.device = "cuda:0"
env = gym.make("G1SingleLegBalance-v0", cfg=cfg)
obs, extras = env.reset()
# obs['policy']: [64, 312], obs['critic']: [64, 140]
# env.step(action): action [64, 23], canonical G1 policy joint order
```

`cfg.task` is an independent dictionary per configuration. Change it before
constructing the environment. Defaults use a flat plane, 200 Hz simulation,
50 Hz control, startup G1 domain randomization, seed 0 and six-second episodes.
The task does not launch training or load any learned controller.

## Actor/action contract

The raw actor vector concatenates four 55-D proprioception frames, followed by
four 23-D action frames, oldest first: `4*55 + 4*23 = 312`. Proprioception is
pelvis rotation-6D, pelvis-local angular velocity, absolute joint positions and
joint velocities in canonical G1 policy order. The action history contract is
`post_delay_normalized_applied_action_policy_order_v1`, taken directly from
`ActionProcessor.applied_action` after delay selection, before physical-target
clamping. It is **not** absolute joint targets or the newly requested pre-delay action.

Actions use existing Ref2Act `offset` mode and its default pose/gain-derived
scale; the reset pose does not redefine the action offset. Delay defaults to zero.
Partial resets seed the physical target to the perturbed reset joint position,
fill the delay buffer with its equivalent normalized action, and repeat the new
proprioception/action frame. No old-episode history remains. Actor proprioception
gets the usual per-component uniform noise; action records and critic stay clean.
Noise is sampled once per new frame.

IRASP can reuse `PAIRRobotOnlyActor`/`PAIRRobotOnlyCore` with this native policy
vector. Apply the chosen checkpoint's input normalization in the external actor
adapter and validate its action mode, joint order and history contract. The old
stand-up wrapper's default q-target history is not compatible. This change does
not add an IRASP PPO training/checkpoint loader.

## Reset pose and perturbations

`envs/single_leg_balance/pose.py` embeds a reproducible motion-seeded IK result,
including source frame and file/model SHA256 hashes. The seed is frame 254 of
`CMU__55__55_01_stageii_001.npz`; named joints project its 29-DoF motion onto the
23-DoF robot. Static IK then aligns left sole and whole-body CoM, raises the right
sole, and constrains the limbs to a moderate configuration. Motion data and
MuJoCo/SciPy are not runtime dependencies.

Reproduce the preparation with `benchmarks/prepare_single_leg_pose.py --motion
<retargeted-npz> ... --output <pose.json>`. The output quaternion convention is
explicitly **xyzw**, matching Ref2Act/Isaac Lab in this repository.

Defaults: joint position ±0.04 rad; joint velocity ±0.20 rad/s; root roll/pitch
±4 degrees; horizontal base velocity ±0.08 m/s; angular velocity ±0.15 rad/s.
Batched left-leg FK lifts/lowers the root so the lowest perturbed support-sole
collision point starts 1 mm above the plane. This prevents floor penetration;
it does not enforce balance or insert simulator settling steps. Joint positions
are clamped to live joint limits. Startup mass/CoM randomization is retained.

Reset and push draws use separate counter-based streams keyed by seed,
environment ID and episode number. Resetting one environment cannot shift another
one's scenario sequence. Compare equal environment counts and episode indices.
Exact training resume must additionally preserve simulator/controller/history and
`_rng.episode` state; a model-only checkpoint is not an exact environment resume.

## Reward and contact contract

Whole-body mass-weighted CoM and CoM velocity define
`xi_xy = c_xy + v_xy / sqrt(9.81 / max(c_z - ground_z, 0.10))`.
Balance quality is `exp(-(distance(xi_xy, left_sole_center)/0.08)^2)`.
This capture-point approximation is shaping, not a physical guarantee or measured
CoP. Support center is the mean of four transformed collision-sphere centers;
clearance uses the lowest sphere surface, including its 5 mm radius.

Reward rates are:

| Component | Weight | Definition |
|---|---:|---|
| Balance | 3.0 | Capture-point alignment, gated by loaded left foot, no right/non-foot contact |
| Upright | 1.5 | Squared torso up projection, clamped to [0,1] |
| Swing clearance | 1.0 | Full credit in [0.08,0.18] m; Gaussian decay outside, same contact gate |
| Height | 0.5 | Gaussian pelvis-height quality about reset height, sigma 0.10 m |
| Pose | 0.25 | Weak mean joint-error quality, sigma 0.6 rad |
| Joint velocity | -0.10 | Mean squared velocity / (5 rad/s)^2, capped at one |
| Applied-action rate | -0.05 | Mean squared action difference, capped at one |
| Bad contact | -0.5 | Any swing/non-foot contact or missing support |

Multiply the weighted sum by control dt. All component qualities/costs lie in
[0,1]. No early completion bonus, external lifting force, or pose curriculum is
used. `SingleLeg/reward_*` logs unweighted qualities; `reward_total` is the actual
returned per-step reward. Nonfinite rewards become zero and nonfinite physical
states terminate; nonfinite post-reset observations raise an error before PPO.

Ground forces are net body contact forces minus explicitly filtered robot
self-contact forces, sampled over **every physics substep**. This subtraction is
valid for the configured plane-only scene; adding external objects requires a
new filter contract. All non-foot rigid bodies are monitored. Contact uses a 1 N
threshold. Left support also requires at least 0.20 body weight in mean vertical
load and current final-substep contact. Peak substep force detects brief touches.

Continuous right-foot contact or left-support loss for 0.18 s fails. Continuous
non-foot contact for 0.06 s fails. The timers reset when their condition clears.
Pelvis height below 0.625 of nominal or torso up projection below 0.5 also fails.
Reaching six seconds without failure is success; a simultaneous failure and time
limit is a failure. There is no early-success termination.

## Disturbances and critic

Each episode draws one push start in [1.0,4.5] s, a horizontal direction in
[0,2*pi), and delta-v in [0,0.30] m/s. A 0.10 s **world-frame force pulse at the
`torso_link` body CoM**, with zero explicit torque, delivers `J = total_mass*delta_v`.
The application point follows the body. Force uses the live randomized total mass.
Interval-overlap integration preserves requested impulse for off-grid start/end
times. Actual applied impulse is accumulated; early termination may truncate it.
Reset explicitly clears the permanent wrench. Push strength has no automatic
curriculum; training can explicitly schedule `delta_v_range` if required.

The 140-D privileged critic contains, in order:

| Terms | Dimensions |
|---|---:|
| Clean current proprioception | 55 |
| Current / previous applied action | 23 + 23 |
| Root world linear velocity | 3 |
| Root-relative feet positions / world feet velocities | 6 + 6 |
| Root-relative CoM position / world CoM velocity | 3 + 3 |
| Foot clearance / normalized foot loads | 2 + 2 |
| Pelvis height / torso up projection | 1 + 1 |
| Three contact violation timers | 3 |
| Applied world force / body weight | 3 |
| Signed time to push / delta-v / direction cosine-sine | 1 + 1 + 2 |
| Episode remaining fraction / recovery hold timer | 1 + 1 |

Push schedule is privileged; none of it appears in actor input. Time-limit
bootstrap uses `extras['final_critic_mask']` (N bool rows) and
`extras['final_critic_observation']` (only those pre-reset critic rows, in mask
order). Terminations, including failure at timeout, do not bootstrap.

## Evaluation

`extras['single_leg_outcomes']` reports completed episode IDs, survival, failure,
duration, swing touchdown events, push timing/strength/direction, applied impulse,
and recovery time. A recovery requires 0.30 continuous seconds of valid single
support, swing clearance >=8 cm, torso projection >=0.9, capture-point error <=5 cm
and horizontal CoM speed <=0.15 m/s after the pulse ends. Recovery time is measured
to completion of this hold window. `-1` means not recovered by episode end. For a
zero-strength push it measures nominal stability after the scheduled pulse window.

`benchmarks/evaluate_single_leg_balance.py` evaluates a TorchScript deterministic
actor accepting raw 312-D observations and returning 23-D actions. Export any
input normalization with the actor. Example:

```bash
python benchmarks/evaluate_single_leg_balance.py --headless \
  --policy /absolute/path/actor.pt --num-envs 64 --seed 101 \
  --output /absolute/path/robustness.json
```

Default grid: delta-v `{0,.1,.2,.3,.4,.5}` m/s, eight directions, a fixed 2 s push.
Each cell uses the same per-env reset scenarios; one episode per environment is
retained to prevent faster-failing policies from oversampling resets. Repeat with
multiple evaluation seeds. Actor observation noise is disabled; startup physics
randomization and reset perturbations remain. Failures before the push still count.

Summaries include survival curve, touchdown-episode rate, recovery fraction and
conditional recovery time. Duration is a **restricted mean survival time**;
surviving episodes are right-censored at 6 s, not recorded as falls. Maximum
recoverable delta-v is the largest contiguous tested strength meeting 80% survival
in every direction; null means even zero strength fails. Convert to impulse with
`J=m*delta_v` for a specified mass. It is a grid/threshold estimate, not a certified
physical maximum. `--reset-hold` is only an open-loop environment diagnostic.

For Random/FDM/PAIR comparisons, keep architecture, critic, normalization protocol,
PPO, budgets, randomization and evaluation scenarios identical. Only encoder
initialization differs. Learning AUC and iterations to 80% survival belong in the
external training analysis; they are not inferred from raw return.

## Verification

```bash
python -m pytest -q tests/unit/test_single_leg_balance.py
python tests/integration/single_leg_balance_smoke.py --headless
```

Validation on 2026-09-14: all 161 Ref2Act unit tests passed, including 12
Single-Leg Balance cases. The native eight-environment smoke also passed. Actual Isaac reset had
11.945 cm right-sole clearance and 0.0039 cm CoM projection distance from left sole
center. At 100 ms, left foot carried ~0.874 body weight and right foot zero.
The open-loop reset-target controller failed after ~0.38 s; this is evidence of
valid initial transitions, **not** a trained balance/recovery result. Partial
history reset, sparse timeout bootstrap, and force clearing were exercised.

A 64-environment default-randomization smoke ran 100 control steps with random
actions: 387 completed episodes, minimum duration 0.26 s, mean duration 0.296 s,
and all observations/rewards finite. The evaluation CLI completed a reduced
two-strength/two-direction grid (32 episodes) and wrote valid JSON summaries.

See [the projected-CoM audit](SINGLE_LEG_BALANCE_COM_AUDIT.md) for independent coordinate checks and the default reset distribution; footprint containment is distinct from capture-point error and actual contact support.

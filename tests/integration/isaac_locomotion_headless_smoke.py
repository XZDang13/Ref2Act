"""Manual GPU smoke test for G1 flat locomotion without a motion file."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--terrain",
        choices=("flat", "slope", "uneven", "mixed"),
        default="flat",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    launcher = AppLauncher(args)

    import gymnasium as gym
    import torch

    import ref2act  # noqa: F401
    from ref2act.robots.g1 import (
        G1FlatLocomotionEnvCfg,
        G1MixedTerrainLocomotionEnvCfg,
        G1SlopeLocomotionEnvCfg,
        G1UnevenLocomotionEnvCfg,
    )

    cfg_classes = {
        "flat": G1FlatLocomotionEnvCfg,
        "slope": G1SlopeLocomotionEnvCfg,
        "uneven": G1UnevenLocomotionEnvCfg,
        "mixed": G1MixedTerrainLocomotionEnvCfg,
    }
    cfg = cfg_classes[args.terrain]()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("G1FlatLocomotion-v0", cfg=cfg)
    failure: BaseException | None = None
    try:
        observation, _ = env.reset()
        expected_shapes = {
            "command": (args.num_envs, 7),
            "robot": (args.num_envs, 78),
            "privilege": (args.num_envs, 88),
        }
        actual_shapes = {name: tuple(value.shape) for name, value in observation.items()}
        if actual_shapes != expected_shapes:
            raise RuntimeError(f"Unexpected observations: {actual_shapes}")
        from pxr import PhysxSchema, Usd
        core = env.unwrapped
        articulations = [PhysxSchema.PhysxArticulationAPI(prim) for prim in
                        Usd.PrimRange(core.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot'), Usd.TraverseInstanceProxies())
                        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI)]
        assert articulations and all(api.GetEnabledSelfCollisionsAttr().Get() for api in articulations)

        assert len(core._reward_effort_joint_indices) == 12
        assert len(core._reward_ankle_joint_indices) == 4
        assert len(core._reward_hip_joint_indices) == 4
        assert len(core._reward_arm_joint_indices) == 10
        assert len(core._reward_torso_joint_indices) == 1
        for _ in range(args.steps):
            observation, reward, terminated, truncated, _ = env.step(
                torch.zeros((args.num_envs, 23), device=env.unwrapped.device)
            )
            tensors = list(observation.values()) + [reward, terminated, truncated]
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise RuntimeError("Locomotion smoke produced a non-finite tensor.")
            terms = core._flat_locomotion_reward_terms()
            from ref2act.envs.locomotion.task_rewards import LOCOMOTION_TASK_REWARD_WEIGHTS
            assert set(terms) == set(LOCOMOTION_TASK_REWARD_WEIGHTS)
            assert all(torch.isfinite(value).all() for value in terms.values())
            assert (terms["feet_air_time"] >= 0).all() and (terms["feet_air_time"] <= 0.600001).all()
            assert (terms["action_rate_l2"] <= 0).all()
            from ref2act.isaac_compat import to_torch
            air = to_torch(core.contact_sensor.data.current_air_time)[:, core._foot_contact_sensor_indices]
            expected_flight = ((air > 0).sum(-1) == 2).float()
            torch.testing.assert_close(terms["both_feet_air"], -0.5 * expected_flight)
            assert "Gait/both_feet_air_fraction" in core.extras["log"]
            assert "Gait/single_stance_fraction" in core.extras["log"]
            assert "Curriculum/reward_penalty_scale" not in core.extras["log"]
        print(
            f"locomotion smoke passed: terrain={args.terrain}, "
            f"num_envs={args.num_envs}, steps={args.steps}",
            flush=True,
        )
    except BaseException as exc:
        failure = exc
        traceback.print_exc()
    finally:
        env.close()
        launcher.app.close()
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

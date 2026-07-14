"""Manual MuJoCo smoke for the current Retargeter motion contract."""

from __future__ import annotations

import argparse

import torch

from ref2act.bridges.mujoco.env import MujocoEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion")
    args = parser.parse_args()

    joint_count = 23
    limits = torch.tensor([[-10.0, 10.0]]).repeat(joint_count, 1)
    env = MujocoEnv(
        simulation_dt=0.005,
        decimation=4,
        kp=torch.full((joint_count,), 20.0),
        kd=torch.full((joint_count,), 0.5),
        effort_limits=torch.full((joint_count,), 100.0),
        joint_pos_limits=limits,
        action_offset=torch.zeros(joint_count),
        action_scale=torch.ones(joint_count),
        expert_motion_file=args.motion,
        root_link_name="pelvis",
        anchor_body_name="torso_link",
        render=False,
    )
    try:
        observation = env.reset()
        if observation.shape != (107,) or not torch.isfinite(observation).all():
            raise RuntimeError(f"Invalid MuJoCo reset observation: shape={tuple(observation.shape)}")
        for _ in range(10):
            observation = env.step(torch.zeros(joint_count))
            if observation.shape != (107,) or not torch.isfinite(observation).all():
                raise RuntimeError(f"Invalid MuJoCo step observation: shape={tuple(observation.shape)}")
        print("mujoco smoke passed: steps=10, policy_dim=107")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

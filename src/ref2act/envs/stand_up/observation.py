from __future__ import annotations
import torch
from ref2act.isaac_compat import to_torch as _to_torch
from .standup_observation import critic_terms, pack_critic_terms, noisy_actor_frame, actor_noise_settings

class StandUpStateExtractor:
    def __init__(self, env):
        self.env = env
        self.asset = env.robot
        self.proprio_dim = 55

    def _anchor_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from ref2act.common.math import quat_apply_inverse, quaternion_to_rotation_6d

        quat = _to_torch(self.asset.data.body_link_quat_w)[:, self.env.anchor_body_index]
        lin = quat_apply_inverse(
            quat,
            _to_torch(self.asset.data.body_link_lin_vel_w)[:, self.env.anchor_body_index],
        )
        ang = quat_apply_inverse(
            quat,
            _to_torch(self.asset.data.body_link_ang_vel_w)[:, self.env.anchor_body_index],
        )
        return quat, quaternion_to_rotation_6d(quat), lin, ang

    def extract(self) -> torch.Tensor:
        _, rotation, _, angular_velocity = self._anchor_state()
        joint_position = self.env._sim_to_policy_order(_to_torch(self.asset.data.joint_pos))
        joint_velocity = self.env._sim_to_policy_order(_to_torch(self.asset.data.joint_vel))
        value = torch.cat(
            (rotation, angular_velocity, joint_position, joint_velocity), dim=-1
        ).float()
        if value.shape[-1] != self.proprio_dim:
            raise RuntimeError("Ref2Act PAIR proprioception layout changed unexpectedly.")
        return value

    def raw_applied_action(self) -> torch.Tensor:
        """Return the post-delay raw action used by intervention-side models."""
        return self.env._sim_to_policy_order(
            self.env.action_processor.applied_action
        ).float().clone()

    def action_history_value(self) -> torch.Tensor:
        """Return the configured history value in policy order.

        Current locomotion collection opts into normalized applied_action;
        existing collection/standup configs retain the physical target below.

        ``ActionProcessor.applied_action`` is a mode-dependent normalized
        command.  Its identical numeric value denotes different joint targets
        in Ref2Act's ``median`` and ``offset`` modes.  The already clamped
        ``target_joint_position`` is the physical command contract shared by
        both modes and is already expressed in the same absolute-radian
        coordinates as the joint-position observation.
        """

        if getattr(getattr(self.env, "cfg", None), "pair_action_history", "q_target") == "applied_action":
            return self.raw_applied_action()
        return self.env._sim_to_policy_order(
            _to_torch(self.env.action_processor.target_joint_position)
        ).float().clone()

class StandUpObservation:
    """Four actor frames plus physical targets; current privileged critic state."""
    def __init__(self, env):
        self.env = env
        self.extractor = StandUpStateExtractor(env)
        self.noise = actor_noise_settings(env.cfg.actor_observation_noise)
        self.proprio = self.actions = None
        self.layout = {}

    def update(self, reset_mask):
        frame = noisy_actor_frame(self.extractor.extract(), self.noise)
        target = self.extractor.action_history_value()
        if self.proprio is None:
            self.proprio = frame[:, None].repeat(1, 4, 1)
            self.actions = target[:, None].repeat(1, 4, 1)
        else:
            self.proprio[:, :-1] = self.proprio[:, 1:].clone()
            self.actions[:, :-1] = self.actions[:, 1:].clone()
            self.proprio[:, -1] = frame
            self.actions[:, -1] = target
            self.proprio[reset_mask] = frame[reset_mask, None]
            self.actions[reset_mask] = target[reset_mask, None]
        policy = torch.cat((self.proprio.flatten(1), self.actions.flatten(1)), -1)
        if not torch.isfinite(policy).all():
            raise RuntimeError("Nonfinite stand-up actor observation")
        critic, self.layout = pack_critic_terms(critic_terms(self.env, self.extractor, reset_mask))
        return {"policy": policy, "critic": critic}

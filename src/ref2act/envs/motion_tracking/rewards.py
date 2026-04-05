from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol

import torch
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from .action import ActionProcessor
from .types import ReferenceMotions


@dataclass(frozen=True)
class RewardTermCfg:
    id: str
    type: str
    weight: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class JointAccPenaltyTermCfg(RewardTermCfg):
    id: str = "joint_acc_penalty"
    type: str = "joint_acc_penalty"
    weight: float = -2.5e-7


@dataclass(frozen=True)
class JointTorquePenaltyTermCfg(RewardTermCfg):
    id: str = "joint_torque_penalty"
    type: str = "joint_torque_penalty"
    weight: float = -1e-5


@dataclass(frozen=True)
class JointLimitPenaltyTermCfg(RewardTermCfg):
    id: str = "joint_limit_penalty"
    type: str = "joint_limit_penalty"
    weight: float = -10.0


@dataclass(frozen=True)
class SelfCollisionPenaltyTermCfg(RewardTermCfg):
    id: str = "self_collision_penalty"
    type: str = "self_collision_penalty"
    weight: float = -0.1
    body_indices: tuple[int, ...] = ()
    force_threshold: float = 1.0


@dataclass(frozen=True)
class FootSlipPenaltyTermCfg(RewardTermCfg):
    id: str = "foot_slip_penalty"
    type: str = "foot_slip_penalty"
    weight: float = -0.1
    foot_body_indices: tuple[int, ...] = ()
    foot_contact_body_indices: tuple[int, ...] = ()
    force_threshold: float = 1.0


@dataclass(frozen=True)
class ActionRatePenaltyTermCfg(RewardTermCfg):
    id: str = "action_rate_penalty"
    type: str = "action_rate_penalty"
    weight: float = -1e-3


@dataclass(frozen=True)
class AnchorPositionRewardTermCfg(RewardTermCfg):
    id: str = "anchor_position_reward"
    type: str = "anchor_position_reward"
    weight: float = 0.5
    anchor_body_index: int = -1
    std: float = 0.3**2
    height_only: bool = False


@dataclass(frozen=True)
class AnchorQuaternionRewardTermCfg(RewardTermCfg):
    id: str = "anchor_quaternion_reward"
    type: str = "anchor_quaternion_reward"
    weight: float = 0.5
    anchor_body_index: int = -1
    std: float = 0.4**2


@dataclass(frozen=True)
class KeyPositionRewardTermCfg(RewardTermCfg):
    id: str = "key_position_reward"
    type: str = "key_position_reward"
    weight: float = 1.0
    key_body_indices: tuple[int, ...] = ()
    std: float = 0.3**2


@dataclass(frozen=True)
class KeyQuaternionRewardTermCfg(RewardTermCfg):
    id: str = "key_quaternion_reward"
    type: str = "key_quaternion_reward"
    weight: float = 1.0
    key_body_indices: tuple[int, ...] = ()
    std: float = 0.4**2


@dataclass(frozen=True)
class KeyLinearVelocityRewardTermCfg(RewardTermCfg):
    id: str = "key_linear_velocity_reward"
    type: str = "key_linear_velocity_reward"
    weight: float = 1.0
    anchor_body_index: int = -1
    key_body_indices: tuple[int, ...] = ()
    std: float = 1.0**2


@dataclass(frozen=True)
class KeyAngularVelocityRewardTermCfg(RewardTermCfg):
    id: str = "key_angular_velocity_reward"
    type: str = "key_angular_velocity_reward"
    weight: float = 1.0
    anchor_body_index: int = -1
    key_body_indices: tuple[int, ...] = ()
    std: float = 3.14**2


@dataclass(frozen=True)
class RewardSpec:
    terms: tuple[RewardTermCfg, ...]
    dt: float
    output_mode: str = "sum"

    def enabled_terms(self) -> tuple[RewardTermCfg, ...]:
        return tuple(term for term in self.terms if term.enabled)


@dataclass
class RewardTermResult:
    value: torch.Tensor
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class RewardComputation:
    vector: torch.Tensor
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    metrics: dict[str, dict[str, torch.Tensor]]


class RewardTerm(Protocol):
    type_name: str

    def compute(self, context: "RewardContext", spec: RewardTermCfg) -> RewardTermResult:
        ...


@dataclass
class RewardContext:
    robot: Articulation
    reference_motion: ReferenceMotions
    contact_sensor: ContactSensor
    action_model: ActionProcessor
    _cache: dict[tuple, tuple[torch.Tensor, ...] | torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def anchor_body_pose_error(
        self,
        anchor_body_index: int,
        *,
        height_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = ("anchor_body_pose_error", anchor_body_index, height_only)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        position_slice = slice(2, 3) if height_only else slice(None)
        robot_anchor_body_positions = self.robot.data.body_pos_w[:, anchor_body_index, position_slice]
        robot_anchor_body_quaternions = self.robot.data.body_quat_w[:, anchor_body_index]

        reference_anchor_body_positions = self.reference_motion.body_positions[:, anchor_body_index, position_slice]
        reference_anchor_body_quaternions = self.reference_motion.body_quaternions[:, anchor_body_index]

        position_error = (robot_anchor_body_positions - reference_anchor_body_positions).square().sum(-1)
        quaternion_error = quat_error_magnitude(robot_anchor_body_quaternions, reference_anchor_body_quaternions).square()
        result = (position_error, quaternion_error)
        self._cache[cache_key] = result
        return result

    def key_body_pose_error(self, key_body_indices: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = ("key_body_pose_error", key_body_indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        robot_key_body_positions = self.robot.data.body_pos_w[:, key_body_indices]
        robot_key_body_quaternions = self.robot.data.body_quat_w[:, key_body_indices]

        reference_relative_key_body_positions = self.reference_motion.body_pos_relative[:, key_body_indices]
        reference_relative_key_body_quaternions = self.reference_motion.body_quat_relative[:, key_body_indices]

        position_error = (robot_key_body_positions - reference_relative_key_body_positions).square().sum(-1).mean(-1)
        quaternion_error = (
            quat_error_magnitude(robot_key_body_quaternions, reference_relative_key_body_quaternions).square().mean(-1)
        )
        result = (position_error, quaternion_error)
        self._cache[cache_key] = result
        return result

    def reference_alignment_quaternion(self, anchor_body_index: int) -> torch.Tensor:
        cache_key = ("reference_alignment_quaternion", anchor_body_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        robot_anchor_quaternion = self.robot.data.body_quat_w[:, anchor_body_index]
        reference_anchor_quaternion = self.reference_motion.body_quaternions[:, anchor_body_index]
        result = yaw_quat(quat_mul(robot_anchor_quaternion, quat_inv(reference_anchor_quaternion)))
        self._cache[cache_key] = result
        return result

    def key_body_state_error(
        self,
        anchor_body_index: int,
        key_body_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = ("key_body_state_error", anchor_body_index, key_body_indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        robot_key_body_lin_vel_w = self.robot.data.body_lin_vel_w[:, key_body_indices]
        robot_key_body_ang_vel_w = self.robot.data.body_ang_vel_w[:, key_body_indices]

        alignment_quaternion_w = self.reference_alignment_quaternion(anchor_body_index)
        alignment_quaternion_w = alignment_quaternion_w[:, None, :].expand(-1, len(key_body_indices), -1)

        reference_key_body_lin_vel_w = quat_apply(
            alignment_quaternion_w,
            self.reference_motion.body_linear_velocities[:, key_body_indices],
        )
        reference_key_body_ang_vel_w = quat_apply(
            alignment_quaternion_w,
            self.reference_motion.body_angular_velocities[:, key_body_indices],
        )

        lin_vel_error = (robot_key_body_lin_vel_w - reference_key_body_lin_vel_w).square().sum(-1).mean(-1)
        ang_vel_error = (robot_key_body_ang_vel_w - reference_key_body_ang_vel_w).square().sum(-1).mean(-1)
        result = (lin_vel_error, ang_vel_error)
        self._cache[cache_key] = result
        return result

    def joint_state_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = ("joint_state_error",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        pos_error = (self.robot.data.joint_pos - self.reference_motion.joint_pos).square().sum(-1)
        vel_error = (self.robot.data.joint_vel - self.reference_motion.joint_vel).square().sum(-1)
        result = (pos_error, vel_error)
        self._cache[cache_key] = result
        return result


def _weighted_result(raw: torch.Tensor, weight: float, *, metrics: dict[str, torch.Tensor] | None = None) -> RewardTermResult:
    value = raw * weight
    term_metrics = {"raw": raw, "weighted": value}
    if metrics:
        term_metrics.update(metrics)
    return RewardTermResult(value=value, metrics=term_metrics)


class JointAccPenaltyTerm:
    type_name = "joint_acc_penalty"

    def compute(self, context: RewardContext, spec: JointAccPenaltyTermCfg) -> RewardTermResult:
        raw = torch.sum(torch.square(context.robot.data.joint_acc), dim=1)
        return _weighted_result(raw, spec.weight)


class JointTorquePenaltyTerm:
    type_name = "joint_torque_penalty"

    def compute(self, context: RewardContext, spec: JointTorquePenaltyTermCfg) -> RewardTermResult:
        raw = torch.sum(torch.square(context.robot.data.applied_torque), dim=1)
        return _weighted_result(raw, spec.weight)


class JointLimitPenaltyTerm:
    type_name = "joint_limit_penalty"

    def compute(self, context: RewardContext, spec: JointLimitPenaltyTermCfg) -> RewardTermResult:
        out_of_limits = -(context.robot.data.joint_pos - context.robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (context.robot.data.joint_pos - context.robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        raw = torch.sum(out_of_limits, dim=1)
        return _weighted_result(raw, spec.weight)


class SelfCollisionPenaltyTerm:
    type_name = "self_collision_penalty"

    def compute(self, context: RewardContext, spec: SelfCollisionPenaltyTermCfg) -> RewardTermResult:
        net_contact_forces = context.contact_sensor.data.net_forces_w_history
        if len(spec.body_indices) == 0:
            if net_contact_forces is not None:
                raw = torch.zeros(net_contact_forces.shape[0], device=net_contact_forces.device)
            else:
                net_contact_forces = context.contact_sensor.data.net_forces_w
                num_envs = 0 if net_contact_forces is None else net_contact_forces.shape[0]
                device = context.contact_sensor.device if net_contact_forces is None else net_contact_forces.device
                raw = torch.zeros(num_envs, device=device)
            return _weighted_result(raw, spec.weight)

        if net_contact_forces is None:
            net_contact_forces = context.contact_sensor.data.net_forces_w
            num_envs = 0 if net_contact_forces is None else net_contact_forces.shape[0]
            device = context.contact_sensor.device if net_contact_forces is None else net_contact_forces.device
            return _weighted_result(torch.zeros(num_envs, device=device), spec.weight)

        filtered_contact_forces = context.contact_sensor.data.force_matrix_w_history
        if filtered_contact_forces is None:
            raw = torch.zeros(net_contact_forces.shape[0], device=net_contact_forces.device)
            return _weighted_result(raw, spec.weight)

        contact_magnitudes = torch.norm(filtered_contact_forces[:, :, spec.body_indices], dim=-1)
        is_contact = contact_magnitudes.amax(dim=1).amax(dim=-1) > spec.force_threshold
        raw = torch.sum(is_contact, dim=1)
        return _weighted_result(raw, spec.weight)


class FootSlipPenaltyTerm:
    type_name = "foot_slip_penalty"

    def compute(self, context: RewardContext, spec: FootSlipPenaltyTermCfg) -> RewardTermResult:
        num_envs = context.robot.data.body_lin_vel_w.shape[0]
        device = context.robot.data.body_lin_vel_w.device
        if len(spec.foot_body_indices) == 0 or len(spec.foot_contact_body_indices) == 0:
            return _weighted_result(torch.zeros(num_envs, device=device), spec.weight)

        contact_history = context.contact_sensor.data.net_forces_w_history
        if contact_history is None:
            net_contact_forces = context.contact_sensor.data.net_forces_w
            if net_contact_forces is None:
                return _weighted_result(torch.zeros(num_envs, device=device), spec.weight)
            contact_history = net_contact_forces.unsqueeze(1)

        is_contact = (
            torch.norm(contact_history[:, :, spec.foot_contact_body_indices], dim=-1).amax(dim=1) > spec.force_threshold
        ).to(context.robot.data.body_lin_vel_w.dtype)
        foot_planar_vel = torch.linalg.norm(context.robot.data.body_lin_vel_w[:, spec.foot_body_indices, :2], dim=-1)
        raw = torch.sum(foot_planar_vel * is_contact, dim=1)
        return _weighted_result(raw, spec.weight)


class ActionRatePenaltyTerm:
    type_name = "action_rate_penalty"

    def compute(self, context: RewardContext, spec: ActionRatePenaltyTermCfg) -> RewardTermResult:
        action_delta = context.action_model.applied_action - context.action_model.previous_applied_action
        raw = torch.sum(torch.square(action_delta), dim=1)
        return _weighted_result(raw, spec.weight)


class AnchorPositionRewardTerm:
    type_name = "anchor_position_reward"

    def compute(self, context: RewardContext, spec: AnchorPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.anchor_body_pose_error(spec.anchor_body_index, height_only=spec.height_only)
        raw = torch.exp(-position_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": position_error})


class AnchorQuaternionRewardTerm:
    type_name = "anchor_quaternion_reward"

    def compute(self, context: RewardContext, spec: AnchorQuaternionRewardTermCfg) -> RewardTermResult:
        _, quaternion_error = context.anchor_body_pose_error(spec.anchor_body_index, height_only=False)
        raw = torch.exp(-quaternion_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": quaternion_error})


class KeyPositionRewardTerm:
    type_name = "key_position_reward"

    def compute(self, context: RewardContext, spec: KeyPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.key_body_pose_error(spec.key_body_indices)
        raw = torch.exp(-position_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": position_error})


class KeyQuaternionRewardTerm:
    type_name = "key_quaternion_reward"

    def compute(self, context: RewardContext, spec: KeyQuaternionRewardTermCfg) -> RewardTermResult:
        _, quaternion_error = context.key_body_pose_error(spec.key_body_indices)
        raw = torch.exp(-quaternion_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": quaternion_error})


class KeyLinearVelocityRewardTerm:
    type_name = "key_linear_velocity_reward"

    def compute(self, context: RewardContext, spec: KeyLinearVelocityRewardTermCfg) -> RewardTermResult:
        lin_vel_error, _ = context.key_body_state_error(spec.anchor_body_index, spec.key_body_indices)
        raw = torch.exp(-lin_vel_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": lin_vel_error})


class KeyAngularVelocityRewardTerm:
    type_name = "key_angular_velocity_reward"

    def compute(self, context: RewardContext, spec: KeyAngularVelocityRewardTermCfg) -> RewardTermResult:
        _, ang_vel_error = context.key_body_state_error(spec.anchor_body_index, spec.key_body_indices)
        raw = torch.exp(-ang_vel_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": ang_vel_error})


REWARD_TERM_REGISTRY: dict[str, RewardTerm] = {
    term.type_name: term
    for term in (
        JointAccPenaltyTerm(),
        JointTorquePenaltyTerm(),
        JointLimitPenaltyTerm(),
        SelfCollisionPenaltyTerm(),
        FootSlipPenaltyTerm(),
        ActionRatePenaltyTerm(),
        AnchorPositionRewardTerm(),
        AnchorQuaternionRewardTerm(),
        KeyPositionRewardTerm(),
        KeyQuaternionRewardTerm(),
        KeyLinearVelocityRewardTerm(),
        KeyAngularVelocityRewardTerm(),
    )
}


def register_reward_term(term: RewardTerm) -> None:
    REWARD_TERM_REGISTRY[term.type_name] = term


def default_reward_spec(*, dt: float, anchor_height_only: bool = False) -> RewardSpec:
    return RewardSpec(
        dt=dt,
        terms=(
            JointAccPenaltyTermCfg(),
            JointTorquePenaltyTermCfg(),
            JointLimitPenaltyTermCfg(),
            SelfCollisionPenaltyTermCfg(),
            FootSlipPenaltyTermCfg(),
            ActionRatePenaltyTermCfg(),
            AnchorPositionRewardTermCfg(height_only=anchor_height_only),
            AnchorQuaternionRewardTermCfg(),
            KeyPositionRewardTermCfg(),
            KeyQuaternionRewardTermCfg(),
            KeyLinearVelocityRewardTermCfg(),
            KeyAngularVelocityRewardTermCfg(),
        ),
    )


class Rewards:
    def __init__(self, spec: RewardSpec):
        self.spec = spec
        self.last_result: RewardComputation | None = None
        self.last_metrics: dict[str, dict[str, float]] = {}

    def get_task_reward(
        self,
        robot: Articulation,
        reference_motion: ReferenceMotions,
        contact_sensor: ContactSensor,
        action_model: ActionProcessor,
    ) -> torch.Tensor:
        context = RewardContext(robot=robot, reference_motion=reference_motion, contact_sensor=contact_sensor, action_model=action_model)
        components: list[torch.Tensor] = []
        component_map: dict[str, torch.Tensor] = {}
        metrics_map: dict[str, dict[str, torch.Tensor]] = {}

        for term_cfg in self.spec.enabled_terms():
            term = REWARD_TERM_REGISTRY[term_cfg.type]
            result = term.compute(context, term_cfg)
            scaled_value = result.value * self.spec.dt
            components.append(scaled_value)
            component_map[term_cfg.id] = scaled_value
            metrics_map[term_cfg.id] = {
                metric_name: metric_value * self.spec.dt if metric_name == "weighted" else metric_value
                for metric_name, metric_value in result.metrics.items()
            }

        if components:
            reward_vector = torch.stack(components, dim=-1)
        else:
            reward_vector = torch.zeros(robot.data.joint_pos.shape[0], 0, device=robot.data.joint_pos.device)
        total_reward = reward_vector.sum(-1)

        self.last_result = RewardComputation(
            vector=reward_vector,
            total=total_reward,
            components=component_map,
            metrics=metrics_map,
        )
        self.last_metrics = {
            term_id: {name: float(value.mean().item()) for name, value in metrics.items()}
            for term_id, metrics in metrics_map.items()
        }

        if self.spec.output_mode == "vector":
            return reward_vector
        if self.spec.output_mode != "sum":
            raise ValueError(f"Unsupported reward output mode: {self.spec.output_mode}")
        return total_reward


@dataclass(frozen=True)
class AMPRewardsCfg:
    discriminator_reward_scale: float = 2.0
    discriminator_reward_weight: float = 1.0


class AMPReward:
    def __init__(self, cfg: AMPRewardsCfg):
        self.cfg = cfg

    def get_rewards(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(logits)

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol

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
class CoMPositionRewardTermCfg(RewardTermCfg):
    id: str = "com_position_reward"
    type: str = "com_position_reward"
    weight: float = 0.25
    std: float = 0.3**2


@dataclass(frozen=True)
class CoMVelocityRewardTermCfg(RewardTermCfg):
    id: str = "com_velocity_reward"
    type: str = "com_velocity_reward"
    weight: float = 0.10
    anchor_body_index: int = -1
    std: float = 1.0**2


@dataclass(frozen=True)
class CoMSupportRewardTermCfg(RewardTermCfg):
    id: str = "com_support_reward"
    type: str = "com_support_reward"
    weight: float = 0.30
    foot_body_indices: tuple[int, ...] = ()
    foot_contact_body_indices: tuple[int, ...] = ()
    force_threshold: float = 10.0
    support_margin: float = 0.05
    std: float = 0.1**2


@dataclass(frozen=True)
class EndEffectorPositionRewardTermCfg(RewardTermCfg):
    id: str = "end_effector_position_reward"
    type: str = "end_effector_position_reward"
    weight: float = 0.35
    end_effector_body_indices: tuple[int, ...] = ()
    std: float = 0.3**2


@dataclass(frozen=True)
class EndEffectorVelocityRewardTermCfg(RewardTermCfg):
    id: str = "end_effector_velocity_reward"
    type: str = "end_effector_velocity_reward"
    weight: float = 0.15
    anchor_body_index: int = -1
    end_effector_body_indices: tuple[int, ...] = ()
    std: float = 1.0**2


@dataclass(frozen=True)
class MultiScaleJointPositionRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_joint_position_reward"
    type: str = "multi_scale_joint_position_reward"
    weight: float = 1.0
    fine_std: float = 0.25**2
    medium_std: float = 0.75**2
    coarse_std: float = 1.50**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleJointVelocityRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_joint_velocity_reward"
    type: str = "multi_scale_joint_velocity_reward"
    weight: float = 0.30
    fine_std: float = 1.0**2
    medium_std: float = 3.0**2
    coarse_std: float = 6.0**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleAnchorHeightRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_anchor_height_reward"
    type: str = "multi_scale_anchor_height_reward"
    weight: float = 0.50
    anchor_body_index: int = -1
    fine_std: float = 0.10**2
    medium_std: float = 0.30**2
    coarse_std: float = 0.70**2
    fine_weight: float = 0.45
    medium_weight: float = 0.35
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleProjectedGravityRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_projected_gravity_reward"
    type: str = "multi_scale_projected_gravity_reward"
    weight: float = 0.80
    anchor_body_index: int = -1
    fine_std: float = 0.10**2
    medium_std: float = 0.35**2
    coarse_std: float = 0.80**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleKeyPositionRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_key_position_reward"
    type: str = "multi_scale_key_position_reward"
    weight: float = 1.0
    key_body_indices: tuple[int, ...] = ()
    fine_std: float = 0.30**2
    medium_std: float = 0.75**2
    coarse_std: float = 1.50**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleKeyQuaternionRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_key_quaternion_reward"
    type: str = "multi_scale_key_quaternion_reward"
    weight: float = 1.0
    key_body_indices: tuple[int, ...] = ()
    fine_std: float = 0.40**2
    medium_std: float = 1.0**2
    coarse_std: float = 2.0**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleKeyLinearVelocityRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_key_linear_velocity_reward"
    type: str = "multi_scale_key_linear_velocity_reward"
    weight: float = 1.0
    anchor_body_index: int = -1
    key_body_indices: tuple[int, ...] = ()
    fine_std: float = 1.0**2
    medium_std: float = 3.0**2
    coarse_std: float = 6.0**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleKeyAngularVelocityRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_key_angular_velocity_reward"
    type: str = "multi_scale_key_angular_velocity_reward"
    weight: float = 1.0
    anchor_body_index: int = -1
    key_body_indices: tuple[int, ...] = ()
    fine_std: float = 3.14**2
    medium_std: float = 6.0**2
    coarse_std: float = 12.0**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleEndEffectorPositionRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_end_effector_position_reward"
    type: str = "multi_scale_end_effector_position_reward"
    weight: float = 0.35
    end_effector_body_indices: tuple[int, ...] = ()
    fine_std: float = 0.30**2
    medium_std: float = 0.75**2
    coarse_std: float = 1.50**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class MultiScaleEndEffectorVelocityRewardTermCfg(RewardTermCfg):
    id: str = "multi_scale_end_effector_velocity_reward"
    type: str = "multi_scale_end_effector_velocity_reward"
    weight: float = 0.15
    anchor_body_index: int = -1
    end_effector_body_indices: tuple[int, ...] = ()
    fine_std: float = 1.0**2
    medium_std: float = 3.0**2
    coarse_std: float = 6.0**2
    fine_weight: float = 0.50
    medium_weight: float = 0.30
    coarse_weight: float = 0.20
    coarse_kernel: str = "rational"


@dataclass(frozen=True)
class TrackingProgressRewardTermCfg(RewardTermCfg):
    id: str = "tracking_progress_reward"
    type: str = "tracking_progress_reward"
    weight: float = 0.50
    soft_threshold: float = 1.0
    recovery_enter_threshold: float = 1.8
    clip: float = 0.20
    potential: str = "log1p"


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
    tracking_quality: Any | None = None
    _cache: dict[tuple, tuple[torch.Tensor, ...] | torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    def _zeros(self) -> torch.Tensor:
        return torch.zeros(self.robot.data.joint_pos.shape[0], device=self.robot.data.joint_pos.device)

    def body_masses(self) -> torch.Tensor:
        cache_key = ("body_masses",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        masses = self.robot.root_physx_view.get_masses().to(self.robot.data.body_pos_w.device)
        self._cache[cache_key] = masses
        return masses

    def _body_pose_error(
        self,
        cache_prefix: str,
        body_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (cache_prefix, body_indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if len(body_indices) == 0:
            result = (self._zeros(), self._zeros())
            self._cache[cache_key] = result
            return result

        robot_body_positions = self.robot.data.body_pos_w[:, body_indices]
        robot_body_quaternions = self.robot.data.body_quat_w[:, body_indices]

        reference_relative_body_positions = self.reference_motion.body_pos_relative[:, body_indices]
        reference_relative_body_quaternions = self.reference_motion.body_quat_relative[:, body_indices]

        position_error = (robot_body_positions - reference_relative_body_positions).square().sum(-1).mean(-1)
        quaternion_error = (
            quat_error_magnitude(robot_body_quaternions, reference_relative_body_quaternions).square().mean(-1)
        )
        result = (position_error, quaternion_error)
        self._cache[cache_key] = result
        return result

    def _body_state_error(
        self,
        cache_prefix: str,
        anchor_body_index: int,
        body_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (cache_prefix, anchor_body_index, body_indices)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if len(body_indices) == 0:
            result = (self._zeros(), self._zeros())
            self._cache[cache_key] = result
            return result

        robot_body_lin_vel_w = self.robot.data.body_lin_vel_w[:, body_indices]
        robot_body_ang_vel_w = self.robot.data.body_ang_vel_w[:, body_indices]

        alignment_quaternion_w = self.reference_alignment_quaternion(anchor_body_index)
        alignment_quaternion_w = alignment_quaternion_w[:, None, :].expand(-1, len(body_indices), -1)

        reference_body_lin_vel_w = quat_apply(
            alignment_quaternion_w,
            self.reference_motion.body_linear_velocities[:, body_indices],
        )
        reference_body_ang_vel_w = quat_apply(
            alignment_quaternion_w,
            self.reference_motion.body_angular_velocities[:, body_indices],
        )

        lin_vel_error = (robot_body_lin_vel_w - reference_body_lin_vel_w).square().sum(-1).mean(-1)
        ang_vel_error = (robot_body_ang_vel_w - reference_body_ang_vel_w).square().sum(-1).mean(-1)
        result = (lin_vel_error, ang_vel_error)
        self._cache[cache_key] = result
        return result

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

    def projected_gravity_error(self, anchor_body_index: int) -> torch.Tensor:
        cache_key = ("projected_gravity_error", anchor_body_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        robot_anchor_quaternion = self.robot.data.body_quat_w[:, anchor_body_index]
        reference_anchor_quaternion = self.reference_motion.body_quaternions[:, anchor_body_index]
        gravity_w = robot_anchor_quaternion.new_tensor((0.0, 0.0, -1.0)).expand(robot_anchor_quaternion.shape[0], -1)
        robot_projected_gravity = quat_apply(quat_inv(robot_anchor_quaternion), gravity_w)
        reference_projected_gravity = quat_apply(quat_inv(reference_anchor_quaternion), gravity_w)
        result = (robot_projected_gravity - reference_projected_gravity).square().sum(-1)
        self._cache[cache_key] = result
        return result

    def key_body_pose_error(self, key_body_indices: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        return self._body_pose_error("key_body_pose_error", key_body_indices)

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
        return self._body_state_error("key_body_state_error", anchor_body_index, key_body_indices)

    def end_effector_pose_error(
        self,
        end_effector_body_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._body_pose_error("end_effector_pose_error", end_effector_body_indices)

    def end_effector_state_error(
        self,
        anchor_body_index: int,
        end_effector_body_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._body_state_error("end_effector_state_error", anchor_body_index, end_effector_body_indices)

    def body_com_positions(self) -> torch.Tensor:
        cache_key = ("body_com_positions",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        result = self.robot.data.body_com_pos_w
        self._cache[cache_key] = result
        return result

    def body_com_velocities(self) -> torch.Tensor:
        cache_key = ("body_com_velocities",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        body_com_offset_w = quat_apply(self.robot.data.body_quat_w, self.robot.data.body_com_pos_b)
        result = self.robot.data.body_link_lin_vel_w + torch.linalg.cross(
            self.robot.data.body_ang_vel_w,
            body_com_offset_w,
            dim=-1,
        )
        self._cache[cache_key] = result
        return result

    def reference_body_com_positions(self) -> torch.Tensor:
        cache_key = ("reference_body_com_positions",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        reference_body_com_offset_w = quat_apply(
            self.reference_motion.body_quat_relative,
            self.robot.data.body_com_pos_b,
        )
        result = self.reference_motion.body_pos_relative + reference_body_com_offset_w
        self._cache[cache_key] = result
        return result

    def reference_body_com_velocities(self, anchor_body_index: int) -> torch.Tensor:
        cache_key = ("reference_body_com_velocities", anchor_body_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        reference_body_com_offset_w = quat_apply(
            self.reference_motion.body_quaternions,
            self.robot.data.body_com_pos_b,
        )
        reference_body_com_vel_w = self.reference_motion.body_linear_velocities + torch.linalg.cross(
            self.reference_motion.body_angular_velocities,
            reference_body_com_offset_w,
            dim=-1,
        )
        alignment_quaternion_w = self.reference_alignment_quaternion(anchor_body_index)
        alignment_quaternion_w = alignment_quaternion_w[:, None, :].expand_as(self.reference_motion.body_quaternions)
        result = quat_apply(alignment_quaternion_w, reference_body_com_vel_w)
        self._cache[cache_key] = result
        return result

    def com_position(self) -> torch.Tensor:
        cache_key = ("com_position",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        body_masses = self.body_masses()
        total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        result = torch.sum(self.body_com_positions() * body_masses[..., None], dim=1) / total_mass
        self._cache[cache_key] = result
        return result

    def reference_com_position(self) -> torch.Tensor:
        cache_key = ("reference_com_position",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        body_masses = self.body_masses()
        total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        result = torch.sum(self.reference_body_com_positions() * body_masses[..., None], dim=1) / total_mass
        self._cache[cache_key] = result
        return result

    def com_velocity(self) -> torch.Tensor:
        cache_key = ("com_velocity",)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        body_masses = self.body_masses()
        total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        result = torch.sum(self.body_com_velocities() * body_masses[..., None], dim=1) / total_mass
        self._cache[cache_key] = result
        return result

    def reference_com_velocity(self, anchor_body_index: int) -> torch.Tensor:
        cache_key = ("reference_com_velocity", anchor_body_index)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        body_masses = self.body_masses()
        total_mass = body_masses.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        result = torch.sum(self.reference_body_com_velocities(anchor_body_index) * body_masses[..., None], dim=1) / total_mass
        self._cache[cache_key] = result
        return result

    def com_support_error(
        self,
        foot_body_indices: tuple[int, ...],
        foot_contact_body_indices: tuple[int, ...],
        force_threshold: float,
        support_margin: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = ("com_support_error", foot_body_indices, foot_contact_body_indices, force_threshold, support_margin)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        zeros = self._zeros()
        num_feet = min(len(foot_body_indices), len(foot_contact_body_indices))
        if num_feet == 0:
            result = (zeros, zeros, zeros)
            self._cache[cache_key] = result
            return result

        contact_history = context_contact_history(self.contact_sensor)
        if contact_history is None:
            result = (zeros, zeros, zeros)
            self._cache[cache_key] = result
            return result

        foot_body_indices = foot_body_indices[:num_feet]
        foot_contact_body_indices = foot_contact_body_indices[:num_feet]
        foot_positions_xy = self.robot.data.body_pos_w[:, foot_body_indices, :2]
        is_contact = (
            torch.norm(contact_history[:, :, foot_contact_body_indices], dim=-1).amax(dim=1) > force_threshold
        )

        dtype = foot_positions_xy.dtype
        contact_count = is_contact.sum(dim=1).to(dtype)
        distance = torch.zeros_like(contact_count)
        if torch.any(is_contact):
            contact_rank = torch.cumsum(is_contact.to(torch.int64), dim=1)
            first_mask = ((contact_rank == 1) & is_contact).to(dtype)
            second_mask = ((contact_rank == 2) & is_contact).to(dtype)
            first_support_xy = torch.sum(foot_positions_xy * first_mask.unsqueeze(-1), dim=1)
            second_support_xy = torch.sum(foot_positions_xy * second_mask.unsqueeze(-1), dim=1)
            com_xy = self.com_position()[:, :2]

            single_contact = contact_count == 1
            if torch.any(single_contact):
                distance[single_contact] = torch.linalg.norm(
                    com_xy[single_contact] - first_support_xy[single_contact],
                    dim=-1,
                )

            multi_contact = contact_count >= 2
            if torch.any(multi_contact):
                distance[multi_contact] = _point_to_segment_distance_2d(
                    com_xy[multi_contact],
                    first_support_xy[multi_contact],
                    second_support_xy[multi_contact],
                )

        error = torch.clamp(distance - support_margin, min=0.0).square()
        result = (error, distance, contact_count)
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


def context_contact_history(contact_sensor: ContactSensor) -> torch.Tensor | None:
    contact_history = contact_sensor.data.net_forces_w_history
    if contact_history is not None:
        return contact_history

    net_contact_forces = contact_sensor.data.net_forces_w
    if net_contact_forces is None:
        return None
    return net_contact_forces.unsqueeze(1)


def _point_to_segment_distance_2d(
    point_xy: torch.Tensor,
    segment_start_xy: torch.Tensor,
    segment_end_xy: torch.Tensor,
) -> torch.Tensor:
    segment = segment_end_xy - segment_start_xy
    segment_length_sq = segment.square().sum(dim=-1, keepdim=True)
    projection = torch.where(
        segment_length_sq > 1.0e-8,
        ((point_xy - segment_start_xy) * segment).sum(dim=-1, keepdim=True) / segment_length_sq,
        torch.zeros_like(segment_length_sq),
    ).clamp_(0.0, 1.0)
    closest = segment_start_xy + projection * segment
    return torch.linalg.norm(point_xy - closest, dim=-1)


def _multi_scale_tracking_kernel(
    error: torch.Tensor,
    *,
    fine_std: float,
    medium_std: float,
    coarse_std: float,
    fine_weight: float,
    medium_weight: float,
    coarse_weight: float,
    coarse_kernel: str = "rational",
) -> torch.Tensor:
    fine = torch.exp(-error / fine_std)
    medium = torch.exp(-error / medium_std)

    if coarse_kernel == "rational":
        coarse = 1.0 / (1.0 + error / coarse_std)
    elif coarse_kernel == "exp":
        coarse = torch.exp(-error / coarse_std)
    else:
        raise ValueError(f"Unknown coarse kernel: {coarse_kernel}")

    return fine_weight * fine + medium_weight * medium + coarse_weight * coarse


def _smoothstep(lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    if hi <= lo:
        return torch.zeros_like(x)
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _multi_scale_result(error: torch.Tensor, spec: Any) -> RewardTermResult:
    raw = _multi_scale_tracking_kernel(
        error,
        fine_std=spec.fine_std,
        medium_std=spec.medium_std,
        coarse_std=spec.coarse_std,
        fine_weight=spec.fine_weight,
        medium_weight=spec.medium_weight,
        coarse_weight=spec.coarse_weight,
        coarse_kernel=spec.coarse_kernel,
    )
    return _weighted_result(raw, spec.weight, metrics={"error": error})


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


class CoMPositionRewardTerm:
    type_name = "com_position_reward"

    def compute(self, context: RewardContext, spec: CoMPositionRewardTermCfg) -> RewardTermResult:
        position_error = (context.com_position()[:, :2] - context.reference_com_position()[:, :2]).square().sum(-1)
        raw = torch.exp(-position_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": position_error})


class CoMVelocityRewardTerm:
    type_name = "com_velocity_reward"

    def compute(self, context: RewardContext, spec: CoMVelocityRewardTermCfg) -> RewardTermResult:
        velocity_error = (
            context.com_velocity()[:, :2] - context.reference_com_velocity(spec.anchor_body_index)[:, :2]
        ).square().sum(-1)
        raw = torch.exp(-velocity_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": velocity_error})


class CoMSupportRewardTerm:
    type_name = "com_support_reward"

    def compute(self, context: RewardContext, spec: CoMSupportRewardTermCfg) -> RewardTermResult:
        support_error, distance, contact_count = context.com_support_error(
            spec.foot_body_indices,
            spec.foot_contact_body_indices,
            spec.force_threshold,
            spec.support_margin,
        )
        raw = torch.exp(-support_error / spec.std)
        raw = torch.where(contact_count > 0.0, raw, torch.zeros_like(raw))
        return _weighted_result(
            raw,
            spec.weight,
            metrics={"error": support_error, "distance": distance, "contact_count": contact_count},
        )


class EndEffectorPositionRewardTerm:
    type_name = "end_effector_position_reward"

    def compute(self, context: RewardContext, spec: EndEffectorPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.end_effector_pose_error(spec.end_effector_body_indices)
        raw = torch.exp(-position_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": position_error})


class EndEffectorVelocityRewardTerm:
    type_name = "end_effector_velocity_reward"

    def compute(self, context: RewardContext, spec: EndEffectorVelocityRewardTermCfg) -> RewardTermResult:
        lin_vel_error, _ = context.end_effector_state_error(spec.anchor_body_index, spec.end_effector_body_indices)
        raw = torch.exp(-lin_vel_error / spec.std)
        return _weighted_result(raw, spec.weight, metrics={"error": lin_vel_error})


class MultiScaleJointPositionRewardTerm:
    type_name = "multi_scale_joint_position_reward"

    def compute(self, context: RewardContext, spec: MultiScaleJointPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.joint_state_error()
        return _multi_scale_result(position_error, spec)


class MultiScaleJointVelocityRewardTerm:
    type_name = "multi_scale_joint_velocity_reward"

    def compute(self, context: RewardContext, spec: MultiScaleJointVelocityRewardTermCfg) -> RewardTermResult:
        _, velocity_error = context.joint_state_error()
        return _multi_scale_result(velocity_error, spec)


class MultiScaleAnchorHeightRewardTerm:
    type_name = "multi_scale_anchor_height_reward"

    def compute(self, context: RewardContext, spec: MultiScaleAnchorHeightRewardTermCfg) -> RewardTermResult:
        height_error, _ = context.anchor_body_pose_error(spec.anchor_body_index, height_only=True)
        return _multi_scale_result(height_error, spec)


class MultiScaleProjectedGravityRewardTerm:
    type_name = "multi_scale_projected_gravity_reward"

    def compute(self, context: RewardContext, spec: MultiScaleProjectedGravityRewardTermCfg) -> RewardTermResult:
        gravity_error = context.projected_gravity_error(spec.anchor_body_index)
        return _multi_scale_result(gravity_error, spec)


class MultiScaleKeyPositionRewardTerm:
    type_name = "multi_scale_key_position_reward"

    def compute(self, context: RewardContext, spec: MultiScaleKeyPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.key_body_pose_error(spec.key_body_indices)
        return _multi_scale_result(position_error, spec)


class MultiScaleKeyQuaternionRewardTerm:
    type_name = "multi_scale_key_quaternion_reward"

    def compute(self, context: RewardContext, spec: MultiScaleKeyQuaternionRewardTermCfg) -> RewardTermResult:
        _, quaternion_error = context.key_body_pose_error(spec.key_body_indices)
        return _multi_scale_result(quaternion_error, spec)


class MultiScaleKeyLinearVelocityRewardTerm:
    type_name = "multi_scale_key_linear_velocity_reward"

    def compute(self, context: RewardContext, spec: MultiScaleKeyLinearVelocityRewardTermCfg) -> RewardTermResult:
        lin_vel_error, _ = context.key_body_state_error(spec.anchor_body_index, spec.key_body_indices)
        return _multi_scale_result(lin_vel_error, spec)


class MultiScaleKeyAngularVelocityRewardTerm:
    type_name = "multi_scale_key_angular_velocity_reward"

    def compute(self, context: RewardContext, spec: MultiScaleKeyAngularVelocityRewardTermCfg) -> RewardTermResult:
        _, ang_vel_error = context.key_body_state_error(spec.anchor_body_index, spec.key_body_indices)
        return _multi_scale_result(ang_vel_error, spec)


class MultiScaleEndEffectorPositionRewardTerm:
    type_name = "multi_scale_end_effector_position_reward"

    def compute(self, context: RewardContext, spec: MultiScaleEndEffectorPositionRewardTermCfg) -> RewardTermResult:
        position_error, _ = context.end_effector_pose_error(spec.end_effector_body_indices)
        return _multi_scale_result(position_error, spec)


class MultiScaleEndEffectorVelocityRewardTerm:
    type_name = "multi_scale_end_effector_velocity_reward"

    def compute(self, context: RewardContext, spec: MultiScaleEndEffectorVelocityRewardTermCfg) -> RewardTermResult:
        lin_vel_error, _ = context.end_effector_state_error(spec.anchor_body_index, spec.end_effector_body_indices)
        return _multi_scale_result(lin_vel_error, spec)


class TrackingProgressRewardTerm:
    type_name = "tracking_progress_reward"

    def compute(self, context: RewardContext, spec: TrackingProgressRewardTermCfg) -> RewardTermResult:
        if context.tracking_quality is None:
            raw = context._zeros()
            return _weighted_result(
                raw,
                spec.weight,
                metrics={"progress": raw, "gate": raw, "score": raw, "previous_score": raw},
            )

        score = context.tracking_quality.score
        previous_score = context.tracking_quality.previous_score
        if spec.potential == "log1p":
            progress = torch.log1p(previous_score) - torch.log1p(score)
        elif spec.potential == "sqrt":
            progress = torch.sqrt(previous_score + 1.0e-6) - torch.sqrt(score + 1.0e-6)
        else:
            raise ValueError(f"Unsupported tracking progress potential: {spec.potential}")

        progress = progress.clamp(-spec.clip, spec.clip)
        gate = _smoothstep(spec.soft_threshold, spec.recovery_enter_threshold, score)
        raw = gate * progress
        return _weighted_result(
            raw,
            spec.weight,
            metrics={"progress": progress, "gate": gate, "score": score, "previous_score": previous_score},
        )


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
        CoMPositionRewardTerm(),
        CoMVelocityRewardTerm(),
        CoMSupportRewardTerm(),
        EndEffectorPositionRewardTerm(),
        EndEffectorVelocityRewardTerm(),
        MultiScaleJointPositionRewardTerm(),
        MultiScaleJointVelocityRewardTerm(),
        MultiScaleAnchorHeightRewardTerm(),
        MultiScaleProjectedGravityRewardTerm(),
        MultiScaleKeyPositionRewardTerm(),
        MultiScaleKeyQuaternionRewardTerm(),
        MultiScaleKeyLinearVelocityRewardTerm(),
        MultiScaleKeyAngularVelocityRewardTerm(),
        MultiScaleEndEffectorPositionRewardTerm(),
        MultiScaleEndEffectorVelocityRewardTerm(),
        TrackingProgressRewardTerm(),
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


def robust_tracking_reward_spec(*, dt: float, include_com_terms: bool = False) -> RewardSpec:
    terms: tuple[RewardTermCfg, ...] = (
        MultiScaleJointPositionRewardTermCfg(),
        MultiScaleJointVelocityRewardTermCfg(),
        MultiScaleAnchorHeightRewardTermCfg(),
        MultiScaleProjectedGravityRewardTermCfg(),
        MultiScaleKeyPositionRewardTermCfg(),
        MultiScaleKeyQuaternionRewardTermCfg(),
        MultiScaleKeyLinearVelocityRewardTermCfg(),
        MultiScaleKeyAngularVelocityRewardTermCfg(),
        MultiScaleEndEffectorPositionRewardTermCfg(),
        MultiScaleEndEffectorVelocityRewardTermCfg(),
        TrackingProgressRewardTermCfg(),
    )
    if include_com_terms:
        terms += (
            CoMPositionRewardTermCfg(),
            CoMVelocityRewardTermCfg(),
            CoMSupportRewardTermCfg(),
        )
    terms += (
        JointAccPenaltyTermCfg(),
        JointTorquePenaltyTermCfg(),
        JointLimitPenaltyTermCfg(),
        SelfCollisionPenaltyTermCfg(),
        FootSlipPenaltyTermCfg(weight=-0.05),
        ActionRatePenaltyTermCfg(),
    )
    return RewardSpec(dt=dt, terms=terms)


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
        tracking_quality: Any | None = None,
    ) -> torch.Tensor:
        context = RewardContext(
            robot=robot,
            reference_motion=reference_motion,
            contact_sensor=contact_sensor,
            action_model=action_model,
            tracking_quality=tracking_quality,
        )
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

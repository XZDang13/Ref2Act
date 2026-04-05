from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Protocol

import torch

from .buffer import DequeBuffer
from .utils import IndexLike


def ensure_batch_vector(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor.flatten(start_dim=1)


@dataclass(frozen=True)
class ObservationNoiseSpec:
    low: float
    high: float
    kind: str = "uniform"

    def __post_init__(self) -> None:
        if self.kind != "uniform":
            raise ValueError(f"Unsupported observation noise kind: {self.kind}")

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.empty_like(value).uniform_(self.low, self.high)


@dataclass(frozen=True)
class ObservationTermSpec:
    id: str
    type: str
    enabled: bool = True
    window_length: int = 1
    flatten: bool = True
    noise: ObservationNoiseSpec | None = None

    def __post_init__(self) -> None:
        if self.window_length < 1:
            raise ValueError("Observation term window_length must be at least 1.")


@dataclass(frozen=True)
class ObservationGroupSpec:
    name: str
    terms: tuple[ObservationTermSpec, ...]
    enabled: bool = True


@dataclass(frozen=True)
class ObservationSpec:
    groups: tuple[ObservationGroupSpec, ...]

    def enabled_groups(self) -> tuple[ObservationGroupSpec, ...]:
        return tuple(group for group in self.groups if group.enabled)

    def describe(
        self,
        layout: "ObservationLayout",
        registry: Mapping[str, "ObservationTerm"] | None = None,
    ) -> "ObservationDescription":
        registry = DEFAULT_OBSERVATION_TERM_REGISTRY if registry is None else registry
        group_dims: dict[str, int] = {}
        for group in self.enabled_groups():
            group_dim = 0
            for term_spec in group.terms:
                if not term_spec.enabled:
                    continue
                term = registry[term_spec.type]
                term_dim = term.dimension(layout, term_spec)
                if term_spec.flatten:
                    group_dim += term_dim * term_spec.window_length
                elif term_spec.window_length == 1:
                    group_dim += term_dim
                else:
                    raise ValueError(
                        f"Observation description only supports flattened history or single-frame terms: {term_spec.id}"
                    )
            group_dims[group.name] = group_dim
        return ObservationDescription(group_dims=group_dims)


@dataclass(frozen=True)
class ObservationDescription:
    group_dims: Mapping[str, int]

    @property
    def total_dim(self) -> int:
        return sum(self.group_dims.values())


@dataclass(frozen=True)
class ObservationLayout:
    joint_dim: int
    action_dim: int
    key_body_count: int


@dataclass(frozen=True)
class ObservationContext:
    target_projected_gravity: torch.Tensor | None = None
    target_joint_pos: torch.Tensor | None = None
    target_joint_vel: torch.Tensor | None = None
    projected_gravity: torch.Tensor | None = None
    anchor_ang_vel_b: torch.Tensor | None = None
    joint_pos: torch.Tensor | None = None
    joint_vel: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    relative_anchor_pos: torch.Tensor | None = None
    relative_anchor_tangent_and_normal: torch.Tensor | None = None
    relative_key_pos: torch.Tensor | None = None
    relative_key_tangent_and_normal: torch.Tensor | None = None
    anchor_lin_vel: torch.Tensor | None = None
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)


class ObservationTerm(Protocol):
    type_name: str

    def compute(self, context: ObservationContext, spec: ObservationTermSpec) -> torch.Tensor:
        ...

    def dimension(self, layout: ObservationLayout, spec: ObservationTermSpec) -> int:
        ...


class ContextFieldObservationTerm:
    def __init__(self, type_name: str, field_name: str, dimension_getter) -> None:
        self.type_name = type_name
        self._field_name = field_name
        self._dimension_getter = dimension_getter

    def compute(self, context: ObservationContext, spec: ObservationTermSpec) -> torch.Tensor:
        value = getattr(context, self._field_name, None)
        if value is None:
            value = context.extras.get(self._field_name)
        if value is None:
            raise KeyError(f"Observation context field '{self._field_name}' required by term '{spec.id}' is missing.")
        return ensure_batch_vector(value)

    def dimension(self, layout: ObservationLayout, spec: ObservationTermSpec) -> int:
        return int(self._dimension_getter(layout))


DEFAULT_OBSERVATION_TERM_REGISTRY: dict[str, ObservationTerm] = {
    "target_projected_gravity": ContextFieldObservationTerm(
        "target_projected_gravity",
        "target_projected_gravity",
        lambda layout: 3,
    ),
    "target_joint_pos": ContextFieldObservationTerm(
        "target_joint_pos",
        "target_joint_pos",
        lambda layout: layout.joint_dim,
    ),
    "target_joint_vel": ContextFieldObservationTerm(
        "target_joint_vel",
        "target_joint_vel",
        lambda layout: layout.joint_dim,
    ),
    "projected_gravity": ContextFieldObservationTerm(
        "projected_gravity",
        "projected_gravity",
        lambda layout: 3,
    ),
    "anchor_ang_vel_b": ContextFieldObservationTerm(
        "anchor_ang_vel_b",
        "anchor_ang_vel_b",
        lambda layout: 3,
    ),
    "joint_pos": ContextFieldObservationTerm(
        "joint_pos",
        "joint_pos",
        lambda layout: layout.joint_dim,
    ),
    "joint_vel": ContextFieldObservationTerm(
        "joint_vel",
        "joint_vel",
        lambda layout: layout.joint_dim,
    ),
    "previous_action": ContextFieldObservationTerm(
        "previous_action",
        "previous_action",
        lambda layout: layout.action_dim,
    ),
    "relative_anchor_pos": ContextFieldObservationTerm(
        "relative_anchor_pos",
        "relative_anchor_pos",
        lambda layout: 3,
    ),
    "relative_anchor_tangent_and_normal": ContextFieldObservationTerm(
        "relative_anchor_tangent_and_normal",
        "relative_anchor_tangent_and_normal",
        lambda layout: 6,
    ),
    "relative_key_pos": ContextFieldObservationTerm(
        "relative_key_pos",
        "relative_key_pos",
        lambda layout: layout.key_body_count * 3,
    ),
    "relative_key_tangent_and_normal": ContextFieldObservationTerm(
        "relative_key_tangent_and_normal",
        "relative_key_tangent_and_normal",
        lambda layout: layout.key_body_count * 6,
    ),
    "anchor_lin_vel": ContextFieldObservationTerm(
        "anchor_lin_vel",
        "anchor_lin_vel",
        lambda layout: 3,
    ),
}


class ObservationComposer:
    def __init__(
        self,
        *,
        spec: ObservationSpec,
        layout: ObservationLayout,
        num_envs: int,
        device: torch.device | str = "cpu",
        registry: Mapping[str, ObservationTerm] | None = None,
    ) -> None:
        self.spec = spec
        self.layout = layout
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.registry = dict(DEFAULT_OBSERVATION_TERM_REGISTRY if registry is None else registry)
        self._history_buffers = self._build_history_buffers()

    def _build_history_buffers(self) -> dict[str, DequeBuffer]:
        buffers: dict[str, DequeBuffer] = {}
        for group in self.spec.enabled_groups():
            for term_spec in group.terms:
                if not term_spec.enabled or term_spec.window_length <= 1:
                    continue
                term = self.registry[term_spec.type]
                term_dim = term.dimension(self.layout, term_spec)
                buffers[term_spec.id] = DequeBuffer(
                    self.num_envs,
                    term_spec.window_length,
                    (term_dim,),
                    device=self.device,
                )
        return buffers

    def _compute_term_value(
        self,
        term_spec: ObservationTermSpec,
        context: ObservationContext,
    ) -> torch.Tensor:
        term = self.registry[term_spec.type]
        value = ensure_batch_vector(term.compute(context, term_spec))
        if term_spec.noise is not None:
            value = term_spec.noise.apply(value)
        return value

    def reset(self, env_ids: IndexLike, context: ObservationContext) -> None:
        for group in self.spec.enabled_groups():
            for term_spec in group.terms:
                if not term_spec.enabled:
                    continue
                if term_spec.window_length <= 1:
                    continue
                value = self._compute_term_value(term_spec, context)
                self._history_buffers[term_spec.id].reset(env_ids, values=value[env_ids], fill_all=True)

    def compose(self, context: ObservationContext) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for group in self.spec.enabled_groups():
            group_values: list[torch.Tensor] = []
            for term_spec in group.terms:
                if not term_spec.enabled:
                    continue
                value = self._compute_term_value(term_spec, context)
                history_buffer = self._history_buffers.get(term_spec.id)
                if history_buffer is not None:
                    history_buffer.append(value)
                    term_value = history_buffer.get_all()
                    if term_spec.flatten:
                        term_value = term_value.flatten(start_dim=1)
                else:
                    term_value = value
                    if not term_spec.flatten:
                        term_value = term_value.unsqueeze(1)
                group_values.append(term_value)
            if not group_values:
                continue
            if len(group_values) == 1:
                outputs[group.name] = group_values[0]
                continue
            if any(value.ndim != 2 for value in group_values):
                raise ValueError(f"Observation group '{group.name}' mixes non-flattened tensors and cannot be concatenated.")
            outputs[group.name] = torch.cat(group_values, dim=-1)
        return outputs


def register_observation_term(term: ObservationTerm) -> None:
    DEFAULT_OBSERVATION_TERM_REGISTRY[term.type_name] = term


def replace_group_terms(spec: ObservationSpec, *groups: ObservationGroupSpec) -> ObservationSpec:
    replacements = {group.name: group for group in groups}
    return ObservationSpec(
        groups=tuple(replacements.get(group.name, group) for group in spec.groups)
    )


__all__ = [
    "DEFAULT_OBSERVATION_TERM_REGISTRY",
    "ObservationComposer",
    "ObservationContext",
    "ObservationDescription",
    "ObservationGroupSpec",
    "ObservationLayout",
    "ObservationNoiseSpec",
    "ObservationSpec",
    "ObservationTerm",
    "ObservationTermSpec",
    "ensure_batch_vector",
    "register_observation_term",
    "replace_group_terms",
]

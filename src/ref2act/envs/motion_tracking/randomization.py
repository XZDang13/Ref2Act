import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


def _resolve_asset(env, *entity_cfgs: SceneEntityCfg | None) -> Articulation:
    asset_name = None
    for entity_cfg in entity_cfgs:
        if entity_cfg is None:
            continue
        if asset_name is None:
            asset_name = entity_cfg.name
            continue
        if entity_cfg.name != asset_name:
            raise ValueError("All entity configs in a structured randomization term must target the same asset.")
    if asset_name is None:
        raise ValueError("Structured randomization requires at least one SceneEntityCfg.")
    return env.scene[asset_name]


def _resolve_env_ids(env, env_ids: torch.Tensor | None, device: str) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=device)
    return env_ids.to(device=device)


def _resolve_ids(ids: slice | list[int] | torch.Tensor | None, total: int, device: str) -> torch.Tensor:
    if ids == slice(None) or ids is None:
        return torch.arange(total, device=device)
    if isinstance(ids, torch.Tensor):
        return ids.to(device=device)
    return torch.tensor(ids, device=device)


def _iter_group_cfgs(
    group_specs: list[tuple[str, SceneEntityCfg | None, tuple[float, float] | None]]
) -> list[tuple[str, SceneEntityCfg, tuple[float, float]]]:
    groups: list[tuple[str, SceneEntityCfg, tuple[float, float]]] = []
    for group_name, entity_cfg, scale_range in group_specs:
        if entity_cfg is None or scale_range is None:
            continue
        groups.append((group_name, entity_cfg, scale_range))
    return groups


def _get_env_cache(env, attr_name: str) -> dict:
    cache = getattr(env, attr_name, None)
    if cache is None:
        cache = {}
        setattr(env, attr_name, cache)
    return cache


def randomize_rigid_body_com_from_default(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
):
    asset = _resolve_asset(env, asset_cfg)
    env_ids_cpu = _resolve_env_ids(env, env_ids, "cpu")
    body_ids = _resolve_ids(asset_cfg.body_ids, asset.num_bodies, "cpu")

    default_com_cache = _get_env_cache(env, "_ref2act_default_com_cache")
    if asset_cfg.name not in default_com_cache:
        default_com_cache[asset_cfg.name] = asset.root_physx_view.get_coms().clone()

    default_coms = default_com_cache[asset_cfg.name]
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids_cpu[:, None], body_ids] = default_coms[env_ids_cpu[:, None], body_ids]

    range_list = [com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")]
    ranges = torch.tensor(range_list, device="cpu")
    offsets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_cpu), 1, 3), device="cpu")
    coms[env_ids_cpu[:, None], body_ids, :3] += offsets
    asset.root_physx_view.set_coms(coms, env_ids_cpu)


def randomize_group_body_masses(
    env,
    env_ids: torch.Tensor | None,
    base_cfg: SceneEntityCfg | None = None,
    base_scale_range: tuple[float, float] | None = None,
    legs_cfg: SceneEntityCfg | None = None,
    legs_scale_range: tuple[float, float] | None = None,
    arms_cfg: SceneEntityCfg | None = None,
    arms_scale_range: tuple[float, float] | None = None,
    min_mass: float = 1e-6,
):
    group_cfgs = _iter_group_cfgs(
        [
            ("base", base_cfg, base_scale_range),
            ("legs", legs_cfg, legs_scale_range),
            ("arms", arms_cfg, arms_scale_range),
        ]
    )
    asset = _resolve_asset(env, *(cfg for _, cfg, _ in group_cfgs))
    env_ids_cpu = _resolve_env_ids(env, env_ids, "cpu")

    masses = asset.root_physx_view.get_masses()
    inertias = asset.root_physx_view.get_inertias()

    seen_body_ids: set[int] = set()
    for group_name, group_cfg, scale_range in group_cfgs:
        body_ids = _resolve_ids(group_cfg.body_ids, asset.num_bodies, "cpu")
        overlap = seen_body_ids.intersection(body_ids.tolist())
        if overlap:
            raise ValueError(f"Structured mass randomization group '{group_name}' overlaps with another group.")
        seen_body_ids.update(body_ids.tolist())

        scale = math_utils.sample_uniform(*scale_range, (len(env_ids_cpu), 1), device="cpu")
        default_mass = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
        masses[env_ids_cpu[:, None], body_ids] = torch.clamp(default_mass * scale, min=min_mass)

        ratios = masses[env_ids_cpu[:, None], body_ids] / default_mass
        default_inertia = asset.data.default_inertia[env_ids_cpu[:, None], body_ids].clone()
        inertias[env_ids_cpu[:, None], body_ids] = default_inertia * ratios[..., None]

    asset.root_physx_view.set_masses(masses, env_ids_cpu)
    asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def randomize_group_actuator_gains(
    env,
    env_ids: torch.Tensor | None,
    legs_cfg: SceneEntityCfg | None = None,
    legs_scale_range: tuple[float, float] | None = None,
    torso_cfg: SceneEntityCfg | None = None,
    torso_scale_range: tuple[float, float] | None = None,
    arms_cfg: SceneEntityCfg | None = None,
    arms_scale_range: tuple[float, float] | None = None,
):
    group_cfgs = _iter_group_cfgs(
        [
            ("legs", legs_cfg, legs_scale_range),
            ("torso", torso_cfg, torso_scale_range),
            ("arms", arms_cfg, arms_scale_range),
        ]
    )
    asset = _resolve_asset(env, *(cfg for _, cfg, _ in group_cfgs))
    env_ids_device = _resolve_env_ids(env, env_ids, asset.device)

    joint_stiffness = asset.data.default_joint_stiffness[env_ids_device].clone()
    joint_damping = asset.data.default_joint_damping[env_ids_device].clone()

    seen_joint_ids: set[int] = set()
    for group_name, group_cfg, scale_range in group_cfgs:
        joint_ids = _resolve_ids(group_cfg.joint_ids, asset.num_joints, asset.device)
        overlap = seen_joint_ids.intersection(joint_ids.tolist())
        if overlap:
            raise ValueError(f"Structured actuator randomization group '{group_name}' overlaps with another group.")
        seen_joint_ids.update(joint_ids.tolist())

        scale = math_utils.sample_uniform(*scale_range, (len(env_ids_device), 1), device=asset.device)
        joint_stiffness[:, joint_ids] = asset.data.default_joint_stiffness[env_ids_device][:, joint_ids] * scale
        joint_damping[:, joint_ids] = asset.data.default_joint_damping[env_ids_device][:, joint_ids] * scale

    asset.write_joint_stiffness_to_sim(joint_stiffness, env_ids=env_ids_device)
    asset.write_joint_damping_to_sim(joint_damping, env_ids=env_ids_device)

    for actuator in asset.actuators.values():
        actuator_joint_ids = _resolve_ids(actuator.joint_indices, asset.num_joints, asset.device)
        actuator.stiffness[env_ids_device] = joint_stiffness[:, actuator_joint_ids]
        actuator.damping[env_ids_device] = joint_damping[:, actuator_joint_ids]


def randomize_group_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    legs_cfg: SceneEntityCfg | None = None,
    legs_scale_range: tuple[float, float] | None = None,
    torso_cfg: SceneEntityCfg | None = None,
    torso_scale_range: tuple[float, float] | None = None,
    arms_cfg: SceneEntityCfg | None = None,
    arms_scale_range: tuple[float, float] | None = None,
    min_effort_limit: float = 1e-4,
):
    group_cfgs = _iter_group_cfgs(
        [
            ("legs", legs_cfg, legs_scale_range),
            ("torso", torso_cfg, torso_scale_range),
            ("arms", arms_cfg, arms_scale_range),
        ]
    )
    asset = _resolve_asset(env, *(cfg for _, cfg, _ in group_cfgs))
    env_ids_device = _resolve_env_ids(env, env_ids, asset.device)

    effort_limit_cache = _get_env_cache(env, "_ref2act_default_effort_limit_cache")
    if group_cfgs[0][1].name not in effort_limit_cache:
        effort_limit_cache[group_cfgs[0][1].name] = asset.data.joint_effort_limits.clone()

    default_effort_limits = effort_limit_cache[group_cfgs[0][1].name]
    effort_limits = asset.data.joint_effort_limits.clone()

    seen_joint_ids: set[int] = set()
    for group_name, group_cfg, scale_range in group_cfgs:
        joint_ids = _resolve_ids(group_cfg.joint_ids, asset.num_joints, asset.device)
        overlap = seen_joint_ids.intersection(joint_ids.tolist())
        if overlap:
            raise ValueError(f"Structured motor strength group '{group_name}' overlaps with another group.")
        seen_joint_ids.update(joint_ids.tolist())

        scale = math_utils.sample_uniform(*scale_range, (len(env_ids_device), 1), device=asset.device)
        default_limits = default_effort_limits[env_ids_device][:, joint_ids]
        effort_limits[:, joint_ids] = torch.clamp(default_limits * scale, min=min_effort_limit)

    asset.write_joint_effort_limit_to_sim(effort_limits, env_ids=env_ids_device)

    for actuator in asset.actuators.values():
        actuator_joint_ids = _resolve_ids(actuator.joint_indices, asset.num_joints, asset.device)
        actuator.effort_limit[env_ids_device] = effort_limits[:, actuator_joint_ids]
        actuator.effort_limit_sim[env_ids_device] = effort_limits[:, actuator_joint_ids]


def randomize_action_latency(
    env,
    env_ids: torch.Tensor | None,
    latency_range: tuple[int, int],
):
    if not hasattr(env, "action_processer"):
        raise AttributeError("Action latency randomization requires env.action_processer to be initialized.")
    env_ids = _resolve_env_ids(env, env_ids, env.device)
    env.action_processer.set_random_delays(env_ids, latency_range)


class randomize_rigid_body_collider_offsets_by_body(ManagerTermBase):
    """Randomize collider offsets on selected bodies while enforcing contact >= rest."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if isinstance(self.asset, Articulation):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore[attr-defined]
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
        else:
            self.num_shapes_per_body = None

    def _shape_ranges(self) -> list[tuple[int, int]]:
        if self.num_shapes_per_body is None or self.asset_cfg.body_ids == slice(None):
            return [(0, self.asset.root_physx_view.max_shapes)]

        ranges = []
        for body_id in self.asset_cfg.body_ids:
            start_idx = sum(self.num_shapes_per_body[:body_id])
            end_idx = start_idx + self.num_shapes_per_body[body_id]
            ranges.append((start_idx, end_idx))
        return ranges

    def __call__(
        self,
        env,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        rest_offset_range: tuple[float, float] | None = None,
        contact_offset_range: tuple[float, float] | None = None,
        min_contact_gap: float = 1e-4,
    ):
        env_ids_cpu = _resolve_env_ids(env, env_ids, "cpu")
        current_rest_offsets = self.asset.root_physx_view.get_rest_offsets().clone()
        current_contact_offsets = self.asset.root_physx_view.get_contact_offsets().clone()
        target_rest_offsets = current_rest_offsets.clone()
        target_contact_offsets = current_contact_offsets.clone()

        for start_idx, end_idx in self._shape_ranges():
            num_shapes = end_idx - start_idx
            if rest_offset_range is not None:
                samples = math_utils.sample_uniform(*rest_offset_range, (len(env_ids_cpu), num_shapes), device="cpu")
                target_rest_offsets[env_ids_cpu, start_idx:end_idx] = torch.clamp(samples, min=0.0)

            if contact_offset_range is not None:
                samples = math_utils.sample_uniform(*contact_offset_range, (len(env_ids_cpu), num_shapes), device="cpu")
                target_contact_offsets[env_ids_cpu, start_idx:end_idx] = torch.clamp(samples, min=min_contact_gap)

            target_contact_offsets[env_ids_cpu, start_idx:end_idx] = torch.maximum(
                target_contact_offsets[env_ids_cpu, start_idx:end_idx],
                target_rest_offsets[env_ids_cpu, start_idx:end_idx] + min_contact_gap,
            )

        # PhysX validates each setter against the currently active counterpart.
        # Raise contact first so the subsequent rest write is always valid.
        contact_offsets_for_rest_update = current_contact_offsets.clone()
        contact_offsets_for_rest_update[env_ids_cpu] = torch.maximum(
            target_contact_offsets[env_ids_cpu],
            current_rest_offsets[env_ids_cpu] + min_contact_gap,
        )

        self.asset.root_physx_view.set_contact_offsets(contact_offsets_for_rest_update, env_ids_cpu)
        self.asset.root_physx_view.set_rest_offsets(target_rest_offsets, env_ids_cpu)
        self.asset.root_physx_view.set_contact_offsets(target_contact_offsets, env_ids_cpu)

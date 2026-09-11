"""Flat-ground support shaping. Tensor functions are independent of Isaac Sim."""
from __future__ import annotations

import re
import math
from itertools import combinations

import torch

from .recovery_reward import smooth_height_gate


def sphere_support_points(rotated_centers, radius=0.005):
    """Lowest points of spheres in world coordinates, regardless of foot tilt."""
    return rotated_centers - rotated_centers.new_tensor([0., 0., radius])


def bilateral_support_ready(foot_load, contact, *, minimum_per_foot):
    return contact.all(-1) & (foot_load.amin(-1) >= minimum_per_foot)


def foot_preparation_quality(sole_points, foot_normals, shank_vectors, *,
                             clearance_sigma, clearance_tolerance,
                             flat_sigma, shank_min_cos, shank_sigma, flat_mode="exponential"):
    """Contact-independent world geometry; no pelvis-height or knee-angle gate.

    Inspired by HumanUP feet_height/feet_orientation and HoST shank_orientation.
    Signed sole normals distinguish a usable sole from an upside-down foot.
    A broad shank tolerance permits forward lean and bent knees.
    """
    clearance = (sole_points[..., 2].abs().mean(-1) - clearance_tolerance).clamp_min(0)
    low = torch.exp(-clearance / clearance_sigma)
    normal_z = foot_normals[..., 2].clamp(-1, 1)
    strict_flat = torch.exp(-((1 - normal_z) / flat_sigma).square())
    if flat_mode not in ("exponential", "signed_linear"):
        raise ValueError("Unknown foot preparation flat_mode.")
    flat = (1 + normal_z) / 2 if flat_mode == "signed_linear" else strict_flat
    shank_cos = shank_vectors[..., 2] / shank_vectors.norm(dim=-1).clamp_min(1.e-6)
    shank = torch.exp(-((shank_min_cos - shank_cos).clamp_min(0) / shank_sigma).square())
    return {"feet_low_per_foot": low, "feet_signed_flat_per_foot": (1 + normal_z) / 2,
            "feet_low": low.mean(-1), "feet_flat": flat.mean(-1), "feet_flat_per_foot": flat,
            "support_orientation_per_foot": strict_flat,
            "shank_upright": shank.mean(-1), "clearance_m": clearance.mean(-1)}


def foot_placement_quality(sole_points, com_xy, *, xy_tolerance, xy_sigma,
                           height_tolerance, height_sigma):
    """Pre-contact geometry, never evidence of load or a support polygon.

    Each sole center approaches a broad disk around projected whole-body CoM
    and the floor. Add squared errors inside one rational score, rather than
    multiplying contact/orientation gates that vanish in lifted-leg states.
    The mean preserves credit for either foot; the minimum favors the worse one.
    """
    center_xy = sole_points[..., :2].mean(-2)
    distance = (center_xy - com_xy[:, None]).norm(dim=-1)
    clearance = sole_points[..., 2].abs().mean(-1)
    xy_error = (distance - xy_tolerance).clamp_min(0) / xy_sigma
    z_error = (clearance - height_tolerance).clamp_min(0) / height_sigma
    per_foot = 1 / (1 + xy_error.square() + z_error.square())
    return {"foot_placement": .5 * (per_foot.mean(-1) + per_foot.amin(-1)),
            "foot_placement_per_foot": per_foot,
            "foot_com_distance_per_foot": distance,
            "foot_clearance_per_foot": clearance}


def hip_placement_quality(relative_xy, sole_points, *, cfg):
    """Per-foot geometry in the pelvis yaw frame, relative to the same-side hip.

    Horizontal quality is independent of ground contact and sole clearance.
    The preparation reward also retains the existing smooth clearance error.
    """
    x, y = relative_xy.unbind(-1)
    x_error = ((cfg["hip_forward_min"] - x).clamp_min(0)
               + (x - cfg["hip_forward_max"]).clamp_min(0)) / cfg["hip_forward_sigma"]
    y_error = (y.abs() - cfg["hip_lateral_tolerance"]).clamp_min(0) / cfg["hip_lateral_sigma"]
    horizontal_error = x_error.square() + y_error.square()
    if cfg.get("placement_geometry", "directional") == "radial":
        # Discovery should improve actual distance, not prescribe an approach
        # direction while the pelvis turns during an asymmetric transition.
        horizontal_error = ((relative_xy.norm(dim=-1) - cfg["hip_radial_tolerance"])
                            .clamp_min(0) / cfg["hip_radial_sigma"]).square()
    geometry = 1 / (1 + horizontal_error)
    clearance = sole_points[..., 2].abs().mean(-1)
    z_error = (clearance - cfg["placement_height_tolerance"]).clamp_min(0) / cfg["placement_height_sigma"]
    placement = 1 / (1 + horizontal_error + z_error.square())
    def aggregate(value):
        leading = value.amax(-1) if cfg.get("allow_first_foot_preparation", False) else value.amin(-1)
        return .5 * (value.mean(-1) + leading)
    return {"hip_geometry": aggregate(geometry), "hip_geometry_per_foot": geometry,
            "foot_placement": aggregate(placement), "foot_placement_per_foot": placement,
            "foot_hip_distance_per_foot": relative_xy.norm(dim=-1),
            "foot_hip_forward_per_foot": x, "foot_hip_lateral_per_foot": y}


def sole_plant_quality(sole_points, foot_normals, *, tolerance, sigma):
    """Reward the whole collision sole near the floor, not a tilted toe/heel touch."""
    worst_clearance = sole_points[..., 2].abs().amax(-1)
    error = (worst_clearance - tolerance).clamp_min(0) / sigma
    signed_flat = (1 + foot_normals[..., 2].clamp(-1, 1)) / 2
    per_foot = signed_flat / (1 + error.square())
    return {"sole_plant": .5 * (per_foot.mean(-1) + per_foot.amin(-1)),
            "sole_plant_per_foot": per_foot,
            "sole_max_clearance_per_foot": worst_clearance}


def squat_support_quality(quality, cfg):
    """Current bilateral readiness; the weaker requirement limits rise credit.

    Require persistent ground contact in addition to orientation-weighted load.
    No height gate: a supported squat can extend without losing readiness.
    Geometry/planting rewards remain available before any contact.
    """
    # Use actual distance rather than the old aggregate geometry score:
    # a single tucked foot cannot hide the other foot being far forward.
    distance_progress = ((cfg["squat_distance_start"] - quality["foot_hip_distance_per_foot"])
        / (cfg["squat_distance_start"] - cfg["squat_distance_full"]))
    geometry = smooth_height_gate(distance_progress, start=0., full=1.)
    plant = smooth_height_gate(quality["sole_plant_per_foot"],
        start=cfg["squat_plant_start"], full=cfg["squat_plant_full"])
    load = smooth_height_gate(quality["usable_foot_load_per_foot"],
        start=cfg["squat_load_start"], full=cfg["squat_load_full"])
    return torch.minimum(torch.minimum(geometry, plant), load).amin(-1) * quality["persistent_contact_per_foot"].all(-1)


def supported_pelvis_lift_quality(root_score, load, upright_projection, *, cfg, hip_quality=None, squat_quality=None):
    """Extra resolution for seat-off, backed by current usable ground load.

    No velocity bonus or peak payout: unsupported jumps/backward bridges earn
    no credit, and unloading immediately removes credit. Full-height objectives
    remain active after this short height interval saturates.
    """
    lift = ((root_score - cfg["pelvis_lift_start"]) /
            (cfg["pelvis_lift_full"] - cfg["pelvis_lift_start"])).clamp(0, 1)
    support = (load / cfg["pelvis_lift_load_full"]).clamp(0, 1)
    forward_righting = (upright_projection / cfg["righting_target_projection"]).clamp(0, 1)
    geometry_multiplier = 1.
    if "pelvis_lift_hip_floor" in cfg:
        if hip_quality is None:
            raise ValueError("Hip-coupled pelvis lift requires horizontal hip geometry.")
        floor = cfg["pelvis_lift_hip_floor"]
        geometry_multiplier = floor + (1 - floor) * hip_quality.clamp(0, 1)
    if cfg.get("bilateral_squat", False):
        if squat_quality is None:
            raise ValueError("Bilateral lift requires squat support quality.")
        geometry_multiplier = squat_quality
    return lift * support * forward_righting * geometry_multiplier


def filtered_support_load(current, previous, *, step_dt, time_constant):
    """Slow load acquisition, immediate release: no airborne filter-tail credit."""
    current = current.clamp(0, 1)
    alpha = -math.expm1(-step_dt / time_constant)
    return torch.minimum(previous + alpha * (current - previous), current).clamp(0, 1)


def com_support_quality(distance, foot_load, persistent_contact, filtered_load, *, sigma, cfg):
    """Soft double-support alignment; any point inside the region scores equally.

    foot_load is per-foot ground load weighted by that foot's sole orientation.
    No support yields distance=inf and zero alignment, without NaN arithmetic.
    The rational distance score retains feedback outside the immediate foot area.
    """
    region_quality = 1 / (1 + (distance.clamp_min(0) / sigma).square())
    both = persistent_contact.all(-1)
    per_foot_gate = smooth_height_gate(foot_load.amin(-1),
        start=cfg["com_min_foot_load_start"], full=cfg["com_min_foot_load_full"])
    load_gate = smooth_height_gate(filtered_load,
        start=cfg["com_load_start"], full=cfg["com_load_full"])
    gate = both.to(filtered_load.dtype) * per_foot_gate * load_gate
    return region_quality, gate, gate * region_quality


def com_preparation_quality(distance, persistent_contact, sole_orientation, *, sigma):
    """Approach the real contact region before full load; inverted/airborne feet give no credit."""
    region = 1 / (1 + (distance.clamp_min(0) / sigma).square())
    gate = persistent_contact.all(-1).to(region.dtype) * sole_orientation.amin(-1).clamp(0, 1)
    return gate * region


def bilateral_load_quality(usable_foot_load, *, target):
    """Encourage a little load on the weaker foot, without demanding symmetry."""
    return (usable_foot_load.amin(-1) / target).clamp(0, 1)


def prepared_support_load(quality, load, cfg):
    """Credit load on each prepared foot, without requiring simultaneous placement.

    Gates depend only on geometry, never contact: unloading cannot switch off a
    deficit. The current usable force and existing filtered total both bound
    load credit, so a well-positioned airborne foot cannot borrow the other
    foot's force. Raw support/success measurements remain unchanged.
    """
    if "support_geometry_start" not in cfg:
        return load, None
    gates = smooth_height_gate(quality["hip_geometry_per_foot"],
        start=cfg["support_geometry_start"], full=cfg["support_geometry_full"])
    prepared = (quality["usable_foot_load_per_foot"] * gates).sum(-1).clamp(0, 1)
    return torch.minimum(load, prepared), gates


def ground_normal_force(net: torch.Tensor, robot_contacts: torch.Tensor) -> torch.Tensor:
    """Valid only in a scene containing the robot and a ground plane."""
    return net - robot_contacts.sum(dim=-2)


def support_region_distance(point, vertices, valid):
    """Distance to the convex hull of valid XY sole points (zero if inside).

    Enumerating the 28 segments and 56 triangles of eight points avoids
    CPU hull construction. Any interior hull point belongs to a triangle.
    One valid point is also supported; an empty hull returns infinity.
    """
    pairs = torch.tensor(list(combinations(range(vertices.shape[1]), 2)), device=point.device)
    a, b = vertices[:, pairs[:, 0]], vertices[:, pairs[:, 1]]
    ab = b - a
    t = ((point[:, None] - a) * ab).sum(-1) / ab.square().sum(-1).clamp_min(1.e-12)
    nearest = a + t.clamp(0, 1)[..., None] * ab
    distance = (point[:, None] - nearest).norm(dim=-1)
    edges_valid = valid[:, pairs[:, 0]] & valid[:, pairs[:, 1]]
    distance = distance.masked_fill(~edges_valid, torch.inf).amin(-1)
    point_distance = (point[:, None] - vertices).norm(dim=-1).masked_fill(~valid, torch.inf).amin(-1)
    distance = torch.minimum(distance, point_distance)
    triples = torch.tensor(list(combinations(range(vertices.shape[1]), 3)), device=point.device)
    a, b, c = (vertices[:, triples[:, i]] for i in range(3))
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    p = point[:, None]
    sides = torch.stack((cross(b-a, p-a), cross(c-b, p-b), cross(a-c, p-c)), -1)
    inside = (sides >= -1.e-8).all(-1) | (sides <= 1.e-8).all(-1)
    inside &= cross(b-a, c-a).abs() > 1.e-8
    inside &= valid[:, triples].all(-1)
    return torch.where(inside.any(-1), 0.0, distance)


def support_quality(foot_fz, weight, sole_points, com_xy, contact_steps, *,
                    contact_force_ratio=0.05, persistence_steps=4, balance_sigma=0.08,
                    sole_height_tolerance=0.015):
    """Bounded load and slow-crouch geometric balance, without a velocity target.

    Sole points are relative to flat ground. Near-ground corners approximate
    the support region; this is not a measurement of contact points or CoP.
    """
    contact = foot_fz > contact_force_ratio * weight[:, None]
    steps = torch.where(contact, contact_steps + 1, 0).clamp_max(persistence_steps)
    persistence = steps.to(foot_fz.dtype) / persistence_steps
    near_ground = sole_points[..., 2].abs() < sole_height_tolerance
    usable = contact & near_ground.any(-1)
    relative_load = (foot_fz.clamp_min(0) / weight[:, None]).clamp_max(1) * usable
    total_load = relative_load.sum(-1)
    load = total_load.clamp_max(1) * (relative_load * persistence).sum(-1) / total_load.clamp_min(1.e-8)
    valid = near_ground & (usable & (steps >= persistence_steps))[..., None]
    distance = support_region_distance(com_xy, sole_points[..., :2].flatten(1, 2), valid.flatten(1))
    balance = torch.exp(-(distance / balance_sigma).square())
    potential = 0.5 * load + 0.5 * load * balance
    return {"load": load, "balance": balance, "potential": potential,
            "distance": distance, "contact_steps": steps, "contact": usable,
            "foot_load": relative_load * persistence / total_load.clamp_min(1)[:, None]}


def potential_reward(previous, current, boundary, *, gamma, scale):
    """Finite-horizon transitions: terminal/timeout potential is zero.

    The corresponding experiment must disable timeout bootstrapping.
    Reset transitions are never evaluated here. No extra dt factor: this is
    a discounted potential difference, not a per-second reward rate.
    """
    return scale * (gamma * torch.where(boundary, 0.0, current) - previous)


def crouch_fall_state(root_height, upright, previous_steps, *, minimum_root_height,
                      minimum_upright_projection, immediate_root_height, required_steps):
    """End a local crouch-to-stand attempt when it leaves its recovery envelope.

    The short debounce tolerates a transient dip/lean. A very low pelvis ends
    immediately. This predicate is deliberately not used for floor get-up.
    """
    outside = (root_height < minimum_root_height) | (upright < minimum_upright_projection)
    steps = torch.where(outside, previous_steps + 1, 0).clamp_max(required_steps)
    fallen = (steps >= required_steps) | (root_height < immediate_root_height)
    return fallen, steps


class SupportContactReader:
    """Four small PhysX views; explicit robot filtering excludes self-contact."""

    def __init__(self, env):
        from pxr import UsdPhysics
        from ref2act.isaac_compat import to_torch as _to_torch
        paths = {}
        for prim in env.sim.stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith('/World/envs/env_0/Robot/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths[prim.GetName()] = path
        self.views = []
        self._to_torch = _to_torch
        self.dt = env.physics_dt
        for name in ('left_ankle_roll_link', 'right_ankle_roll_link',
                     'left_hip_roll_link', 'right_hip_roll_link'):
            # Explicit paths avoid matching visual mesh children (which can
            # have the same name as their parent link) through broad globs.
            patterns = [paths[name].replace('/env_0/', f'/env_{i}/') for i in range(env.num_envs)]
            target_paths = sorted(paths.values()) if 'ankle' in name else [paths['pelvis']]
            filters = [[p.replace('/env_0/', f'/env_{i}/') for p in target_paths]
                       for i in range(env.num_envs)]
            bodies = env.sim.physics_sim_view.create_rigid_body_view(patterns)
            env_ids = [int(re.search(r'/env_(\d+)/', p).group(1)) for p in bodies.prim_paths]
            if sorted(env_ids) != list(range(env.num_envs)):
                raise RuntimeError(f'Contact view does not cover each environment exactly once: {name}')
            order = torch.tensor(env_ids, device=env.device).argsort()
            view = env.sim.physics_sim_view.create_rigid_contact_view(
                patterns, filter_patterns=filters, max_contact_data_count=env.num_envs * 128)
            if view.filter_count != len(target_paths):
                raise RuntimeError(f'Unexpected contact filter count for {name}: {view.filter_count}')
            self.views.append((view, order))

    def read(self):
        feet, hips = [], []
        for i, (view, order) in enumerate(self.views):
            matrix = self._to_torch(view.get_contact_force_matrix(dt=self.dt))[order]
            if i < 2:
                net = self._to_torch(view.get_net_contact_forces(dt=self.dt))[order]
                feet.append(ground_normal_force(net, matrix)[..., 2])
            else:
                hips.append(matrix[:, 0].norm(dim=-1))
        return torch.stack(feet, -1), torch.stack(hips, -1)


# Compatibility exports for existing evaluations and old test imports.
from .standup_transfer_rewards import (  # noqa: E402
    righting_quality, independent_height_deficits, shank_preparation_gate, support_transfer_rewards,
)

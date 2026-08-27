from __future__ import annotations

import re

import torch
import warp as wp

from isaaclab.cloner.cloner_utils import iter_clone_plan_matches
from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.sim.utils.queries import get_all_matching_child_prims, resolve_matching_prims_from_source
from isaaclab_physx.physics import PhysxManager as SimulationManager
from isaaclab_physx.sensors.contact_sensor import ContactSensor


class NestedContactSensor(ContactSensor):
    """PhysX contact sensor that supports rigid bodies nested at different USD depths."""

    def _resolve_indices_and_mask(self, env_ids=None, env_mask=None):
        """Build reset masks without relying on Isaac's broken Warp fancy-index view."""
        if env_ids is None and env_mask is None:
            return self._ALL_ENV_MASK
        if env_mask is not None:
            return env_mask
        mask_torch = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        mask_torch[torch.as_tensor(env_ids, dtype=torch.long, device=self._device)] = True
        self._reset_mask = wp.from_torch(mask_torch)
        self._reset_mask_torch = mask_torch
        return self._reset_mask

    def _initialize_impl(self) -> None:
        BaseContactSensor._initialize_impl(self)
        # Isaac Lab 3's sensor reset path uses indexed Torch writes into this
        # zero-copy mask.  Construct the view explicitly through DLPack because
        # the current Warp/ProxyArray combination can leave ``wp.to_torch`` as
        # a Warp array at runtime.
        self._reset_mask_torch = torch.from_dlpack(wp.to_dlpack(self._reset_mask))
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        parent_expr, leaf_pattern = self.cfg.prim_path.rsplit("/", 1)
        name_pattern = re.compile(leaf_pattern)

        def has_contact_report(prim) -> bool:
            return bool(name_pattern.fullmatch(prim.GetName())) and "PhysxContactReportAPI" in prim.GetAppliedSchemas()

        matches = resolve_matching_prims_from_source(parent_expr)
        if not matches:
            raise RuntimeError(f"No prim found at '{parent_expr}'.")
        asset_prim, _ = matches[0]
        prims = get_all_matching_child_prims(
            asset_prim.GetPath().pathString,
            predicate=has_contact_report,
            traverse_instance_prims=False,
        )
        if not prims:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find nested bodies with contact reporter API."
            )

        # Keep the view environment-major, as required by ContactSensor's buffer
        # reshapes, while avoiding PhysX globs.  PhysX's ``*`` also spans path
        # separators, so a glob ending in ``*_ankle_roll_link`` additionally
        # probes the identically named visual child below each ankle rigid body.
        # That child has neither a rigid-body nor a contact-report API.
        template_root_path = asset_prim.GetPath().pathString
        relative_body_paths = [prim.GetPath().pathString[len(template_root_path) :] for prim in prims]
        if self._parent_prims:
            instance_root_paths = [prim.GetPath().pathString for prim in self._parent_prims]
        elif self._clone_plan is not None:
            instance_root_paths_by_env = {}
            for source_root, destination_template, source_path, env_ids in iter_clone_plan_matches(
                self._clone_plan, parent_expr
            ):
                source_suffix = source_path[len(source_root) :]
                for env_id in env_ids:
                    instance_root_paths_by_env[env_id] = destination_template.format(env_id) + source_suffix
            if set(instance_root_paths_by_env) != set(range(self._num_envs)):
                raise RuntimeError(f"Failed to resolve nested contact roots for all {self._num_envs} environments.")
            instance_root_paths = [instance_root_paths_by_env[env_id] for env_id in range(self._num_envs)]
        else:
            raise RuntimeError(f"Failed to resolve nested contact roots at '{parent_expr}'.")
        body_paths = [
            instance_root_path + relative_path
            for instance_root_path in instance_root_paths
            for relative_path in relative_body_paths
        ]
        filter_patterns = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]
        contact_filter_patterns = [filter_patterns] * len(body_paths) if filter_patterns else []
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_paths)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_paths,
            filter_patterns=contact_filter_patterns,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(prims) * self._num_envs,
        )
        self._num_sensors = self.body_physx_view.count // self._num_envs
        if self._num_sensors != len(prims):
            raise RuntimeError(
                f"Failed to initialize nested contact reporter: expected {len(prims)}, got {self._num_sensors}."
            )
        if (self.cfg.track_contact_points or self.cfg.track_friction_forces) and not filter_patterns:
            raise ValueError("Contact point or friction tracking requires filter_prim_paths_expr.")
        self._create_buffers()


__all__ = ["NestedContactSensor"]

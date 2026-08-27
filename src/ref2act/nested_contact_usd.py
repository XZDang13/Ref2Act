from __future__ import annotations

from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone


def _activate_nested_contact_reports(root_prim, *, threshold: float = 0.0) -> int:
    """Apply contact reporting to every nested rigid body below ``root_prim``."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    count = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rigid_api.CreateSleepThresholdAttr().Set(0.0)
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        report_api.CreateThresholdAttr().Set(float(threshold))
        count += 1
    if count == 0:
        raise ValueError(f"No nested rigid bodies found below {root_prim.GetPath()}.")
    return count


@clone
def spawn_usd_with_nested_contact_reports(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD and activate reports before the prototype is cloned.

    IsaacLab's generic helper intentionally stops descending once it finds a
    rigid body.  The G1 USD nests link rigid bodies below its rigid pelvis, so
    the generic helper only activates the pelvis.  Applying the APIs to the
    prototype here makes the complete set propagate to every environment clone.
    """

    prim = _spawn_from_usd_file(
        prim_path,
        cfg.usd_path,
        cfg,
        translation,
        orientation,
    )
    _activate_nested_contact_reports(prim)
    return prim


__all__ = ["spawn_usd_with_nested_contact_reports"]

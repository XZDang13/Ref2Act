from __future__ import annotations

from isaaclab_physx.sensors.ray_caster.ray_caster import RayCaster


class DirectLeafRayCaster(RayCaster):
    """Bind a ray caster only to the explicitly matched nested rigid body."""

    def _initialize_impl(self) -> None:
        import carb

        setting_path = "/physics/tensors/recursiveLeafPatternMatch"
        settings = carb.settings.get_settings()
        recursive_leaf_match = settings.get_as_bool(setting_path)
        settings.set_bool(setting_path, False)
        try:
            super()._initialize_impl()
        finally:
            settings.set_bool(setting_path, recursive_leaf_match)


__all__ = ["DirectLeafRayCaster"]

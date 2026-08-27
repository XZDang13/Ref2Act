from __future__ import annotations

from isaaclab_physx.assets.articulation import Articulation


class DirectLeafArticulation(Articulation):
    """Create the PhysX articulation view without recursive leaf matching.

    Isaac Sim's tensor API recursively matches a named final path component by
    default.  For the nested G1 USD, the intended articulation root
    ``.../Geometry/pelvis`` contains a visual child that is also named
    ``pelvis``.  The recursive match therefore probes ``pelvis/pelvis`` as a
    second articulation and emits one warning per environment.

    The setting is scoped to view creation so other tensor views retain the
    process-wide default behavior.
    """

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


__all__ = ["DirectLeafArticulation"]

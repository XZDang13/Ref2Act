import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG

from .sampler import ReferenceMotions
from .utils import IndexLike

class ReferenceMotionViewer:
    def __init__(self, body_indices:IndexLike):
        marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        marker_cfg.markers["target"].radius = 0.05
        marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0)
        )
        marker_cfg.prim_path = "/Visuals/reference_motion"

        self.markers = VisualizationMarkers(marker_cfg)
        self.markers.set_visibility(True)

        self.body_indices = body_indices

    def visualize(self, reference_motion: ReferenceMotions):
        body_position = reference_motion.body_pos_relative[:, self.body_indices].view(-1, 3)
        self.markers.visualize(body_position)
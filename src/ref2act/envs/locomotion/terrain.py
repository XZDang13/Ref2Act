from __future__ import annotations

from typing import Literal

import numpy as np
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


def _height_mesh(size: tuple[float, float], resolution: float, height: np.ndarray) -> trimesh.Trimesh:
    """Create an upward-facing regular triangle mesh over ``[0, size]``."""

    num_x, num_y = height.shape
    x = np.linspace(0.0, size[0], num_x, dtype=np.float64)
    y = np.linspace(0.0, size[1], num_y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), height.ravel()))

    cell_x, cell_y = np.meshgrid(
        np.arange(num_x - 1, dtype=np.int64),
        np.arange(num_y - 1, dtype=np.int64),
        indexing="ij",
    )
    v00 = (cell_x * num_y + cell_y).ravel()
    v10 = ((cell_x + 1) * num_y + cell_y).ravel()
    v01 = (cell_x * num_y + cell_y + 1).ravel()
    v11 = ((cell_x + 1) * num_y + cell_y + 1).ravel()
    faces = np.concatenate(
        (
            np.column_stack((v00, v10, v01)),
            np.column_stack((v10, v11, v01)),
        ),
        axis=0,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _grid_shape(size: tuple[float, float], resolution: float) -> tuple[int, int]:
    if resolution <= 0.0:
        raise ValueError("Terrain mesh resolution must be positive.")
    return (
        max(2, int(np.ceil(size[0] / resolution)) + 1),
        max(2, int(np.ceil(size[1] / resolution)) + 1),
    )


@configclass
class Ref2ActFlatTerrainCfg(SubTerrainBaseCfg):
    function = "ref2act.envs.locomotion.terrain:flat_terrain"


@configclass
class Ref2ActSlopeTerrainCfg(SubTerrainBaseCfg):
    """Continuous four-sided slope with a flat spawn platform and zero-height edges."""

    function = "ref2act.envs.locomotion.terrain:slope_terrain"
    slope_range_deg: tuple[float, float] = (2.0, 12.0)
    platform_width: float = 1.5
    resolution: float = 0.20
    inverted: bool = False

    def __post_init__(self) -> None:
        low, high = self.slope_range_deg
        if low < 0.0 or high < low or high >= 45.0:
            raise ValueError(f"Invalid slope_range_deg: {self.slope_range_deg}.")
        if self.platform_width <= 0.0 or self.platform_width >= min(self.size):
            raise ValueError("platform_width must be positive and smaller than the terrain patch.")
        _grid_shape(self.size, self.resolution)


@configclass
class Ref2ActUnevenTerrainCfg(SubTerrainBaseCfg):
    """Smooth band-limited terrain that does not introduce discrete obstacles."""

    function = "ref2act.envs.locomotion.terrain:uneven_terrain"
    amplitude_range: tuple[float, float] = (0.01, 0.06)
    wavelength_range: tuple[float, float] = (0.8, 2.5)
    resolution: float = 0.15
    num_waves: int = 8

    def __post_init__(self) -> None:
        low, high = self.amplitude_range
        if low < 0.0 or high < low:
            raise ValueError(f"Invalid amplitude_range: {self.amplitude_range}.")
        wave_low, wave_high = self.wavelength_range
        if wave_low <= 0.0 or wave_high < wave_low:
            raise ValueError(f"Invalid wavelength_range: {self.wavelength_range}.")
        if self.num_waves < 1:
            raise ValueError("num_waves must be positive.")
        _grid_shape(self.size, self.resolution)


def flat_terrain(
    difficulty: float,
    cfg: Ref2ActFlatTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    del difficulty
    height = np.zeros((2, 2), dtype=np.float64)
    mesh = _height_mesh(cfg.size, max(cfg.size), height)
    return [mesh], np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0))


def slope_terrain(
    difficulty: float,
    cfg: Ref2ActSlopeTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a continuous slope; ``inverted`` switches uphill/downhill starts."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    slope_deg = cfg.slope_range_deg[0] + difficulty * (cfg.slope_range_deg[1] - cfg.slope_range_deg[0])
    slope = np.tan(np.deg2rad(slope_deg))

    num_x, num_y = _grid_shape(cfg.size, cfg.resolution)
    x = np.linspace(0.0, cfg.size[0], num_x, dtype=np.float64)
    y = np.linspace(0.0, cfg.size[1], num_y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    distance_to_edge = np.minimum.reduce((xx, cfg.size[0] - xx, yy, cfg.size[1] - yy))
    ramp_width = 0.5 * (min(cfg.size) - cfg.platform_width)
    height = slope * np.minimum(distance_to_edge, ramp_width)
    if cfg.inverted:
        height = -height

    center_x = int(np.argmin(np.abs(x - 0.5 * cfg.size[0])))
    center_y = int(np.argmin(np.abs(y - 0.5 * cfg.size[1])))
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], height[center_x, center_y]))
    return [_height_mesh(cfg.size, cfg.resolution, height)], origin


def uneven_terrain(
    difficulty: float,
    cfg: Ref2ActUnevenTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate reproducible smooth uneven ground with edges tapered to zero."""

    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    difficulty_key = int(round(difficulty * 1_000_000))
    seed = getattr(cfg, "seed", None)
    rng = np.random.default_rng(np.random.SeedSequence((int(seed or 0), difficulty_key, 0x52454632)))

    num_x, num_y = _grid_shape(cfg.size, cfg.resolution)
    x = np.linspace(0.0, cfg.size[0], num_x, dtype=np.float64)
    y = np.linspace(0.0, cfg.size[1], num_y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    height = np.zeros_like(xx)
    for _ in range(cfg.num_waves):
        direction = rng.uniform(0.0, 2.0 * np.pi)
        wavelength = rng.uniform(*cfg.wavelength_range)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        coordinate = np.cos(direction) * xx + np.sin(direction) * yy
        height += np.sin((2.0 * np.pi / wavelength) * coordinate + phase)

    peak = float(np.max(np.abs(height)))
    if peak > 0.0:
        height *= amplitude / peak
    edge_taper = np.sin(np.pi * xx / cfg.size[0]) * np.sin(np.pi * yy / cfg.size[1])
    height *= edge_taper

    center_region = (
        (np.abs(xx - 0.5 * cfg.size[0]) <= 0.6)
        & (np.abs(yy - 0.5 * cfg.size[1]) <= 0.6)
    )
    origin_z = float(np.max(height[center_region]))
    origin = np.asarray((0.5 * cfg.size[0], 0.5 * cfg.size[1], origin_z))
    return [_height_mesh(cfg.size, cfg.resolution, height)], origin


TerrainMode = Literal["slope", "uneven", "mixed"]


def make_locomotion_terrain_cfg(
    mode: TerrainMode = "mixed",
    *,
    patch_size: tuple[float, float] = (8.0, 8.0),
    num_rows: int = 8,
    num_cols: int = 20,
    max_init_terrain_level: int = 1,
    slope_range_deg: tuple[float, float] = (2.0, 12.0),
    uneven_amplitude_range: tuple[float, float] = (0.01, 0.06),
) -> TerrainImporterCfg:
    """Build a Ref2Act-owned terrain set using Isaac Lab only as the mesh importer."""

    if mode not in {"slope", "uneven", "mixed"}:
        raise ValueError(f"Unsupported locomotion terrain mode: {mode!r}.")
    sub_terrains: dict[str, SubTerrainBaseCfg] = {}
    if mode == "mixed":
        sub_terrains["flat"] = Ref2ActFlatTerrainCfg(proportion=0.4)
    if mode in {"slope", "mixed"}:
        slope_proportion = 0.5 if mode == "slope" else 0.15
        sub_terrains["uphill"] = Ref2ActSlopeTerrainCfg(
            proportion=slope_proportion,
            slope_range_deg=slope_range_deg,
            inverted=True,
        )
        sub_terrains["downhill"] = Ref2ActSlopeTerrainCfg(
            proportion=slope_proportion,
            slope_range_deg=slope_range_deg,
            inverted=False,
        )
    if mode in {"uneven", "mixed"}:
        sub_terrains["uneven"] = Ref2ActUnevenTerrainCfg(
            proportion=1.0 if mode == "uneven" else 0.3,
            amplitude_range=uneven_amplitude_range,
        )

    material = PhysxRigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    generator = TerrainGeneratorCfg(
        seed=0,
        curriculum=True,
        size=patch_size,
        border_width=10.0,
        border_height=1.0,
        num_rows=num_rows,
        num_cols=num_cols,
        color_scheme="none",
        difficulty_range=(0.0, 1.0),
        sub_terrains=sub_terrains,
        use_cache=False,
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=max_init_terrain_level,
        collision_group=-1,
        physics_material=material,
        debug_vis=False,
    )


__all__ = [
    "Ref2ActFlatTerrainCfg",
    "Ref2ActSlopeTerrainCfg",
    "Ref2ActUnevenTerrainCfg",
    "TerrainMode",
    "flat_terrain",
    "make_locomotion_terrain_cfg",
    "slope_terrain",
    "uneven_terrain",
]

from __future__ import annotations

from importlib import resources
from pathlib import Path


def asset_path(*parts: str) -> Path:
    return Path(str(resources.files("ref2act.assets").joinpath(*parts)))


def robot_asset_path(*parts: str) -> Path:
    return asset_path("robots", *parts)


def scene_asset_path(*parts: str) -> Path:
    return asset_path("scenes", *parts)


"""CLI path validation and user-facing path errors."""
from pathlib import Path

import typer

from gsstudio.infrastructure.paths import find_data_root, scene_dir


def data_root() -> Path:
    return find_data_root()


def scene_path(location_id: str, scene_id: str) -> Path:
    path = scene_dir(data_root(), location_id, scene_id)
    if not (path / "scene.yaml").is_file():
        raise typer.BadParameter(f"Scene does not exist: {location_id}/{scene_id}")
    return path

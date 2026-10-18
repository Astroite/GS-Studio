"""Sparse COLMAP depth anchors used to regularize floater geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def export_sparse_depth_anchors(train_images: dict, rows: list[dict], points: dict,
                                image_id_to_row_index: dict[int, int]) -> dict[str, np.ndarray]:
    """Project retained SfM points into training views as depth anchors.

    ``train_images`` holds COLMAP ``Image`` records with filtered ``point3D_ids``.
    ``points`` maps retained point id to ``Point3D`` with world ``xyz``.
    """
    row_index: list[int] = []
    u: list[float] = []
    v: list[float] = []
    depth: list[float] = []
    point_id: list[int] = []
    for image_id, image in train_images.items():
        if image_id not in image_id_to_row_index:
            raise RuntimeError('Training image missing package row')
        world_to_camera = np.asarray(rows[image_id_to_row_index[image_id]]['world_to_camera'], dtype=np.float64)
        positions = np.flatnonzero(np.asarray(image.point3D_ids) >= 0)
        for position in positions:
            pid = int(image.point3D_ids[position])
            point = points.get(pid)
            if point is None:
                continue
            xyz = np.asarray(point.xyz, dtype=np.float64)
            camera_xyz = world_to_camera[:3, :3] @ xyz + world_to_camera[:3, 3]
            z = float(camera_xyz[2])
            if not np.isfinite(z) or z <= 1e-6:
                continue
            xy = np.rint(np.asarray(image.xys[position], dtype=np.float64)).astype(np.int64)
            row = rows[image_id_to_row_index[image_id]]
            if not (0 <= int(xy[0]) < row['width'] and 0 <= int(xy[1]) < row['height']):
                continue
            row_index.append(int(image_id_to_row_index[image_id]))
            u.append(float(xy[0]))
            v.append(float(xy[1]))
            depth.append(z)
            point_id.append(pid)
    return dict(
        row_index=np.asarray(row_index, dtype=np.int32),
        u=np.asarray(u, dtype=np.float32),
        v=np.asarray(v, dtype=np.float32),
        depth=np.asarray(depth, dtype=np.float32),
        point_id=np.asarray(point_id, dtype=np.int64),
    )


def write_sparse_depth(path: Path, anchors: dict[str, np.ndarray]) -> None:
    np.savez(path, **anchors)


def load_sparse_depth(package: Path) -> dict[str, np.ndarray] | None:
    path = package / 'sparse_depth.npz'
    if not path.is_file():
        return None
    with np.load(path) as data:
        required = {'row_index', 'u', 'v', 'depth'}
        if not required <= set(data.files):
            raise RuntimeError('sparse_depth.npz is missing required fields')
        return {name: data[name] for name in data.files}

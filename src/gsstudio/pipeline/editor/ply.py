"""Standard SH3 Gaussian Splatting PLY layout and header reader."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


FLOAT_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def read_ply_header(path: Path) -> tuple[bytes, int, list[str]]:
    with path.open("rb") as stream:
        header = bytearray()
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            header.extend(line)
            if line.strip() == b"end_header":
                break
    text = header.decode("ascii")
    if "format binary_little_endian 1.0" not in text:
        raise ValueError("Only binary_little_endian PLY 1.0 is supported")
    match = re.search(r"^element vertex (\d+)$", text, re.MULTILINE)
    if match is None:
        raise ValueError("PLY header has no vertex count")
    properties = re.findall(r"^property float (\S+)$", text, re.MULTILINE)
    return bytes(header), int(match.group(1)), properties


def _rotate_quaternions_wxyz(quaternions: np.ndarray, rotation: np.ndarray) -> None:
    """Apply a world-space rotation to the PLY's scalar-first quaternions."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        rotating = np.array([
            scale / 4.0,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            rotating = np.array([(matrix[2, 1] - matrix[1, 2]) / scale,
                                 scale / 4.0,
                                 (matrix[0, 1] + matrix[1, 0]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale])
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            rotating = np.array([(matrix[0, 2] - matrix[2, 0]) / scale,
                                 (matrix[0, 1] + matrix[1, 0]) / scale,
                                 scale / 4.0,
                                 (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            rotating = np.array([(matrix[1, 0] - matrix[0, 1]) / scale,
                                 (matrix[0, 2] + matrix[2, 0]) / scale,
                                 (matrix[1, 2] + matrix[2, 1]) / scale,
                                 scale / 4.0])
    rw, rx, ry, rz = rotating / np.linalg.norm(rotating)
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    transformed = np.column_stack((
        rw*w - rx*x - ry*y - rz*z,
        rw*x + rx*w + ry*z - rz*y,
        rw*y - rx*z + ry*w + rz*x,
        rw*z + rx*y - ry*x + rz*w,
    ))
    norm = np.linalg.norm(transformed, axis=1)
    if not np.all(np.isfinite(norm)) or np.any(norm == 0):
        raise ValueError("PLY contains an invalid Gaussian rotation")
    quaternions[:] = transformed / norm[:, None]


def _sh_basis(directions: np.ndarray) -> np.ndarray:
    """Degree 1–3 real SH basis in the standard 3D Gaussian PLY order."""
    x, y, z = np.asarray(directions, dtype=np.float64).T
    xx, yy, zz = x*x, y*y, z*z
    return np.column_stack((
        -0.4886025119029199*y,
        0.4886025119029199*z,
        -0.4886025119029199*x,
        1.0925484305920792*x*y,
        -1.0925484305920792*y*z,
        0.31539156525252005*(2*zz-xx-yy),
        -1.0925484305920792*x*z,
        0.5462742152960396*(xx-yy),
        -0.5900435899266435*y*(3*xx-yy),
        2.890611442640554*x*y*z,
        -0.4570457994644658*y*(4*zz-xx-yy),
        0.3731763325901154*z*(2*zz-3*xx-3*yy),
        -0.4570457994644658*x*(4*zz-xx-yy),
        1.445305721320277*z*(xx-yy),
        -0.5900435899266435*x*(xx-3*yy),
    ))


def _sh_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Map SH coefficients so rotating the splat preserves its world-space color."""
    count = 128
    indices = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / count
    angle = indices * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(1.0 - z*z)
    directions = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), z))
    basis = _sh_basis(directions)
    # For a rotated splat, the original view direction is R^-1 d.
    rotated_basis = _sh_basis(directions @ np.asarray(rotation, dtype=np.float64))
    return np.linalg.lstsq(basis, rotated_basis, rcond=None)[0]

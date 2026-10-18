"""Coordinates: internal world is Three.js RUB; raster cameras are COLMAP RDF.

Quaternions are scalar first (w,x,y,z). SH uses the real 3DGS/gsplat ordering.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y

CV_TO_WORLD = np.diag([1., -1., -1., 1.])
SH0 = 0.28209479177387814


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-15)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def matrix_quat(m):
    return Rotation.from_matrix(m).as_quat()[..., [3, 0, 1, 2]]


def euler_matrix(euler):
    return Rotation.from_euler("XYZ", euler).as_matrix()


def sh_basis(directions, degree):
    d = np.asarray(directions, dtype=np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-15)
    theta = np.arccos(d[..., 2].clip(-1, 1))
    phi = np.arctan2(d[..., 1], d[..., 0])
    columns = []
    for band in range(degree + 1):
        for order in range(-band, band + 1):
            value = sph_harm_y(band, abs(order), theta, phi)
            columns.append(value.real if order == 0 else math.sqrt(2) *
                           (value.imag if order < 0 else value.real))
    return np.stack(columns, axis=-1)


def rotate_sh(sh, rotation):
    """Rotate directional radiance using an exact band-limited basis fit.

    For active world rotation R: new radiance(d)=old radiance(R^-1 d).
    Oversampled deterministic collocation avoids convention-sensitive Wigner tables.
    Degree <= 4 residuals are verified by tests against independently sampled rays.
    """
    degree = round(math.sqrt(sh.shape[1]) - 1)
    if (degree + 1) ** 2 != sh.shape[1] or degree > 4:
        raise ValueError("Only complete SH bands through degree 4 are supported")
    if np.allclose(rotation, np.eye(3), atol=1e-12):
        return sh.copy()
    n = 128
    z = 1 - 2 * (np.arange(n) + .5) / n
    phi = np.arange(n) * (math.pi * (3 - math.sqrt(5)))
    d = np.stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z], -1)
    before, after = sh_basis(d, degree), sh_basis(d @ rotation, degree)
    result = np.empty_like(sh)
    for band in range(degree + 1):
        s = slice(band * band, (band + 1) ** 2)
        transform = np.linalg.lstsq(before[:, s], after[:, s], rcond=None)[0]
        result[:, s] = np.einsum("ij,njc->nic", transform, sh[:, s])
    return result


def apply_edits(cloud, transform=None, crop=None):
    """Crop in model local coordinates, then uniform scale/XYZ rotate/translate.

    The viewport must attach its crop modifier to the untransformed SplatMesh.
    """
    result = {k: np.asarray(v).copy() for k, v in cloud.items()}
    if crop and crop.get("enabled"):
        lower, upper = np.asarray(crop["min"]), np.asarray(crop["max"])
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or (lower > upper).any():
            raise ValueError("Crop min must not exceed max")
        mask = ((result["means"] >= lower) & (result["means"] <= upper)).all(-1)
        if crop.get("invert", False):
            mask = ~mask
        result = {key: value[mask] for key, value in result.items()}
    if transform:
        scale = float(transform.get("scale", 1))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Uniform scale must be finite and positive")
        position,rotation = np.asarray(transform.get("position",[0,0,0])),np.asarray(transform.get("rotation",[0,0,0]))
        if position.shape!=(3,) or rotation.shape!=(3,) or not np.isfinite([position,rotation]).all():
            raise ValueError("Transform position/rotation must be finite XYZ vectors")
        r = euler_matrix(transform.get("rotation", [0, 0, 0]))
        result["means"] = result["means"] @ r.T * scale + np.asarray(transform.get("position", [0,0,0]))
        result["scales"] += math.log(scale)
        result["quats"] = matrix_quat(r @ quat_matrix(result["quats"]))
        result["sh"] = rotate_sh(result["sh"], r)
    return result


def colmap_to_world(points, w2c):
    return points @ CV_TO_WORLD[:3,:3].T, w2c @ CV_TO_WORLD

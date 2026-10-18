# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper utils for processing equirectangular data."""

# GS Studio changes (2026-10-17): retain only the used function closure;
# replace module imports with local/minimal dependencies. Function bodies unchanged.
# Provenance, source and excerpt SHA256, and licenses: ../provenance.json.

import numpy as np
import torch

# https://gist.github.com/fgolemo/94b5caf0e209a6e71ab0ce2d75ad3ed8
def euler_rodriguez_rotation_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Generates a 3x3 rotation matrix from an axis and angle using Euler-Rodriguez formula.

    Args:
        axis (torch.Tensor): Axis about which to rotate.
        theta (torch.Tloat): Angle to rotate by.

    Returns:
        torch.Tensor: 3x3 Rotation matrix.
    """
    axis = axis / torch.sqrt(torch.dot(axis, axis))
    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def remap_cubic(
    img: torch.Tensor, map_x: torch.Tensor, map_y: torch.Tensor, border_mode: str = "border"
) -> torch.Tensor:
    """Remap image using bicubic interpolation.

    Args:
        img (torch.Tensor): Image tensor
        map_x (torch.Tensor): x mapping
        map_y (torch.Tensor): y mapping
        border_mode (str, optional): What to do with borders. Defaults to "border".

    Returns:
        torch.Tensor: _description_
    """
    batch_size, channels, height, width = img.shape

    grid_x = (map_x / width + 1) * 2 - 1
    grid_y = (map_y / height + 1) * 2 - 1

    if border_mode == "border":
        grid_x = torch.clamp(grid_x, -1, 1)
        grid_y = torch.clamp(grid_y, -1, 1)
    elif border_mode == "wrap":
        grid_x = torch.remainder(grid_x + 1, 2) - 1
        grid_y = torch.remainder(grid_y + 1, 2) - 1

    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.nn.functional.grid_sample(img, grid, mode="bicubic", padding_mode="zeros").clamp(0.0, 1.0)


def equirect2persp(img: torch.Tensor, fov: int, theta: int, phi: int, hd: int, wd: int) -> torch.Tensor:
    """Pytorch reimlement of https://github.com/kaustubh-sadekar/OmniCV-Lib for equirectangular to perspective projection.

    Args:
        img (torch.Tensor): Image tensor
        fov (int): Horizontal field of view in degrees
        theta (int): Horizontal angle in degrees
        phi (int): Vertical angle in degrees
        hd (int): Number of pixels in height
        wd (int): Number of pixels in width

    Returns:
        torch.Tensor: Planar image tensor
    """
    device = img.device
    theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device)
    phi_tensor = torch.tensor(phi, dtype=torch.float32, device=device)
    # theta is left/right angle, phi is up/down angle, both in degree
    equ_h, equ_w = img.shape[2:]

    equ_cx = (equ_w) / 2.0
    equ_cy = (equ_h) / 2.0

    wfov = fov
    hfov = float(hd) / wd * wfov

    c_x = (wd) / 2.0
    c_y = (hd) / 2.0

    w_len = 2 * torch.tan(torch.deg2rad(torch.tensor(wfov / 2.0, device=device)))
    w_interval = w_len / wd

    h_len = 2 * torch.tan(torch.deg2rad(torch.tensor(hfov / 2.0, device=device)))
    h_interval = h_len / hd

    x_map = torch.zeros([hd, wd], dtype=torch.float32, device=device) + 1
    y_map = torch.tile((torch.arange(0, wd, device=device) - c_x) * w_interval, [hd, 1])
    z_map = -torch.tile((torch.arange(0, hd, device=device) - c_y) * h_interval, [wd, 1]).T
    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)

    xyz = torch.zeros([hd, wd, 3], dtype=torch.float32, device=device)
    xyz[:, :, 0] = (x_map / D)[:, :]
    xyz[:, :, 1] = (y_map / D)[:, :]
    xyz[:, :, 2] = (z_map / D)[:, :]

    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    R1 = euler_rodriguez_rotation_matrix(z_axis, torch.deg2rad(theta_tensor)).to(device)
    R2 = euler_rodriguez_rotation_matrix(torch.mm(R1, y_axis.view(3, 1)).squeeze(), torch.deg2rad(-phi_tensor)).to(
        device
    )

    xyz = xyz.view(hd * wd, 3).T
    xyz = torch.mm(R1, xyz)
    xyz = torch.mm(R2, xyz).T
    lat = torch.arcsin(xyz[:, 2] / 1)
    lon = torch.zeros([hd * wd], dtype=torch.float32, device=device)
    theta_tensor = torch.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0
    idx3 = ~idx1 & idx2
    idx4 = ~idx1 & ~idx2

    lon[idx1] = theta_tensor[idx1]
    lon[idx3] = theta_tensor[idx3] + np.pi
    lon[idx4] = theta_tensor[idx4] - np.pi

    lon = lon.view(hd, wd) / torch.pi * 180
    lat = -lat.view(hd, wd) / torch.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    return remap_cubic(img, lon, lat, border_mode="wrap")

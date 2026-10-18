"""Bilateral-grid photometric compensation (BilARF-style locally affine ISP)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class BilateralGrid(torch.nn.Module):
    """Per panorama-group bilateral grid approximating a locally affine ISP.

    Each group owns an ``(2, 3, bins, spatial, spatial)`` volume of affine
    coefficients ``(a, b)`` so that ``out = a(x, y, intensity) * rgb + b``.
    Initial state is the identity map.
    """

    def __init__(self, num_groups: int, spatial: int = 8, bins: int = 8):
        super().__init__()
        if num_groups < 1 or spatial < 1 or bins < 1:
            raise ValueError('BilateralGrid dimensions must be positive')
        self.spatial = spatial
        self.bins = bins
        grid = torch.zeros(num_groups, 2, 3, bins, spatial, spatial)
        grid[:, 0] = 1.0
        self.grid = torch.nn.Parameter(grid)

    def forward(self, rgb: torch.Tensor, group: int) -> torch.Tensor:
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected HxWx3 RGB')
        if not 0 <= int(group) < self.grid.shape[0]:
            raise IndexError('Unknown photometric group')
        height, width, _ = rgb.shape
        coefficients = self.grid[int(group)]
        volume = coefficients.reshape(1, 6, self.bins, self.spatial, self.spatial)
        device = rgb.device
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        intensity = rgb.detach().mean(dim=-1) * 2.0 - 1.0
        coords = torch.stack([gx, gy, intensity], dim=-1)[None, None]
        sampled = F.grid_sample(volume, coords, mode='bilinear', padding_mode='border', align_corners=True)
        affine = sampled[0, :, 0].permute(1, 2, 0)
        return affine[..., :3] * rgb + affine[..., 3:]

    def regularization(self) -> torch.Tensor:
        grid = self.grid
        near_identity = (grid[:, 0] - 1.0).square().mean() + grid[:, 1].square().mean()
        if self.spatial < 2 and self.bins < 2:
            return near_identity
        spatial_tv = (
            (grid[..., :, 1:, :] - grid[..., :, :-1, :]).square().mean()
            + (grid[..., :, :, 1:] - grid[..., :, :, :-1]).square().mean()
        )
        intensity_tv = (grid[..., 1:, :, :] - grid[..., :-1, :, :]).square().mean() if self.bins > 1 else grid.sum() * 0.0
        return near_identity + 0.1 * spatial_tv + 0.1 * intensity_tv

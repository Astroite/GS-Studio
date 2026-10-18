"""Opacity-aware, center-in-rectangle selection for gsplat 1.5.3."""
from __future__ import annotations

import numpy as np

from gsstudio.pipeline.editor.edit import project_points


def compositing_weights(alpha, pixels):
    """Front-to-back alpha*T weights for pairs sorted by pixel then depth."""
    import torch

    if not alpha.numel():
        return alpha
    log_alpha = torch.log1p(-alpha.clamp(max=.999))
    running = torch.cumsum(log_alpha, 0)
    positions = torch.arange(alpha.numel(), device=alpha.device)
    starts = torch.cummax(torch.where(
        torch.cat((torch.ones(1, dtype=torch.bool, device=alpha.device), pixels[1:] != pixels[:-1])),
        positions, 0), 0).values
    before_group = running[(starts - 1).clamp_min(0)]
    before_group = torch.where(starts > 0, before_group, 0)
    return alpha * torch.exp(running - log_alpha - before_group)


def visible_centers(editor, camera: dict, rectangle, *, threshold: float = 1 / 255,
                    tile_size: int = 96, max_intersections: int = 2_000_000) -> np.ndarray:
    """Select center candidates with a visible alpha*T contribution in the ROI.

    Tiled re-projection keeps intersection storage bounded. Errors propagate to
    the workbench; the caller must never substitute depth-only picking.
    """
    import torch
    from gsplat import rasterization, rasterize_to_indices_in_range

    xyz = editor.xyz()
    xy, depth = project_points(xyz, camera)
    x0, y0, x1, y1 = rectangle
    left, right = max(0, int(np.floor(min(x0, x1)))), min(camera["width"], int(np.ceil(max(x0, x1))))
    top, bottom = max(0, int(np.floor(min(y0, y1)))), min(camera["height"], int(np.ceil(max(y0, y1))))
    candidates = (editor.keep & (depth > 0) & (xy[:, 0] >= min(x0, x1)) &
                  (xy[:, 0] <= max(x0, x1)) & (xy[:, 1] >= min(y0, y1)) &
                  (xy[:, 1] <= max(y0, y1)))
    result = np.zeros(len(xyz), dtype=bool)
    if not candidates.any() or left >= right or top >= bottom:
        return result
    kept_ids = np.flatnonzero(editor.keep)
    data = editor.materialize()
    device = "cuda"
    means = torch.as_tensor(data[:, :3], device=device)
    quats = torch.as_tensor(data[:, 58:62].copy(), device=device)
    scales = torch.as_tensor(data[:, 55:58].copy(), device=device).exp()
    opacities = torch.as_tensor(data[:, 54].copy(), device=device).sigmoid()
    colors = torch.zeros((len(data), 3), dtype=torch.float32, device=device)
    view = torch.as_tensor(camera["world_to_camera"], dtype=torch.float32, device=device)[None]
    base_K = np.asarray(camera["K"], dtype=np.float32)
    candidate_tensor = torch.as_tensor(candidates[kept_ids], device=device)
    with torch.no_grad():
        for tile_y in range(top, bottom, tile_size):
            for tile_x in range(left, right, tile_size):
                width = min(tile_size, right - tile_x)
                height = min(tile_size, bottom - tile_y)
                K = base_K.copy()
                K[0, 2] -= tile_x
                K[1, 2] -= tile_y
                _, _, meta = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities,
                    colors=colors, viewmats=view,
                    Ks=torch.as_tensor(K, device=device)[None], width=width, height=height,
                    packed=False,
                )
                gaussian_ids, pixel_ids, _ = rasterize_to_indices_in_range(
                    0, 2**31 - 1, torch.ones((1, height, width), device=device),
                    meta["means2d"], meta["conics"], meta["opacities"],
                    width, height, meta["tile_size"], meta["isect_offsets"], meta["flatten_ids"],
                )
                count = gaussian_ids.numel()
                if count > max_intersections:
                    raise RuntimeError("可见拾取交点超出内存预算；请缩小选框")
                if not count:
                    continue
                order = torch.argsort(meta["depths"][0, gaussian_ids], stable=True)
                order = order[torch.argsort(pixel_ids[order], stable=True)]
                ids, pixels = gaussian_ids[order].long(), pixel_ids[order].long()
                means2d = meta["means2d"][0, ids]
                conics = meta["conics"][0, ids]
                dx = (pixels % width).float() + .5 - means2d[:, 0]
                dy = (pixels // width).float() + .5 - means2d[:, 1]
                sigma = .5 * (conics[:, 0] * dx.square() + 2 * conics[:, 1] * dx * dy +
                              conics[:, 2] * dy.square())
                alpha = (meta["opacities"][0, ids] * torch.exp(-sigma)).clamp(0, .999)
                contribution = compositing_weights(alpha, pixels)
                pixel_x = tile_x + (pixels % width).float() + .5
                pixel_y = tile_y + (pixels // width).float() + .5
                inside_rectangle = ((pixel_x >= min(x0, x1)) & (pixel_x <= max(x0, x1)) &
                                    (pixel_y >= min(y0, y1)) & (pixel_y <= max(y0, y1)))
                accepted = ids[candidate_tensor[ids] & inside_rectangle & (contribution >= threshold)]
                if accepted.numel():
                    result[kept_ids[accepted.unique().cpu().numpy()]] = True
    return result

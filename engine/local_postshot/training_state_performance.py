"""Optional single-camera strategy state accumulation over the same GS metadata.

This module changes no loss, optimizer, topology decision, RNG, or checkpoint
format. The caller must freeze its selection in the recipe. Unsupported packed,
multi-camera, or non-float32 inputs return False before changing state, allowing
the original gsplat 1.5.3 _update_state to run instead.
"""
from __future__ import annotations


def update_single_camera_state(state, info, *, key_for_gradient="means2d",
                               absgrad=False, track_radii=True, packed=False):
    """Return True after a fixed-length equivalent of gsplat's C=1 update.

    Upstream selects visible IDs then index_adds once per ID. With one camera,
    each Gaussian has at most one contribution, so a masked dense add has the
    same per-element arithmetic and avoids dynamic nonzero/gather operations.
    Invisible non-finite gradients are masked with where, not multiplied by0.
    Existing grad2d/count/radii tensors are updated in place as upstream does.
    """
    import torch
    if packed or info.get("n_cameras") != 1:
        return False
    for key in ("width", "height", "n_cameras", "radii", "gaussian_ids", key_for_gradient):
        if key not in info:
            raise AssertionError(f"{key} is required but missing.")
    variable = info[key_for_gradient]
    source = getattr(variable, "absgrad", None) if absgrad else variable.grad
    radii = info["radii"]
    if source is None:
        raise ValueError("Strategy metadata has no retained screen gradient")
    if (source.dtype != torch.float32 or source.ndim != 3 or source.shape[0] != 1
            or source.shape[-1] != 2 or radii.shape != source.shape):
        return False
    n = source.shape[1]
    for key in ("grad2d", "count") + (("radii",) if track_radii else ()):
        value = state.get(key)
        if value is not None and (value.shape != (n,) or value.dtype != torch.float32
                                  or value.device != source.device):
            raise ValueError(f"Strategy state {key} differs from Gaussian metadata")
    with torch.no_grad():
        # Match upstream clone, then independent x/y scaling in the same order.
        gradient = source.clone()
        gradient[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        gradient[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        visible = (radii > 0.0).all(dim=-1)[0]
        for key in ("grad2d", "count") + (("radii",) if track_radii else ()):
            if state.get(key) is None:
                state[key] = torch.zeros(n, device=source.device, dtype=torch.float32)
        state["grad2d"].add_(torch.where(visible, gradient[0].norm(dim=-1), 0.0))
        state["count"].add_(visible.to(dtype=torch.float32))
        if track_radii:
            normalized = radii[0].max(dim=-1).values / float(max(info["width"], info["height"]))
            maximum = torch.maximum(state["radii"], normalized)
            state["radii"].copy_(torch.where(visible, maximum, state["radii"]))
    return True

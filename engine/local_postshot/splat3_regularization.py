"""Independent mathematical reference for a statically observed scale gradient.

Current-version evidence: evidence/splat3/round2-regularization/contract.json.
The observed_v4 recipe explicitly integrates this gradient before Adam. Anchors
use the recovered depth/focal formula with public gsplat visibility; this support
test is a substitute for the native projection kernel.
The scalar below is an independently integrated potential producing the observed
gradient, not a claim that the original reports or reduces this scalar as loss.
"""
from __future__ import annotations

import math


ANCHOR_VISIBILITY_IMPLEMENTATION = "gsplat1.5.3 fully_fused_projection public substitute for native i_aN8 support culling"


def _inputs(log_scales, min_scales, visible_mask, upper_weight, lower_weight=.1):
    import torch
    if (not isinstance(log_scales, torch.Tensor) or log_scales.ndim != 2
            or log_scales.shape[1] != 3 or not log_scales.is_floating_point()):
        raise ValueError("log_scales must be an explicitly floating-point Nx3 tensor")
    if not math.isfinite(float(upper_weight)) or upper_weight < 0:
        raise ValueError("upper_weight must be finite and nonnegative")
    if not math.isfinite(float(lower_weight)) or lower_weight < 0:
        raise ValueError("lower_weight must be finite and nonnegative")
    count = len(log_scales)
    anchors = torch.as_tensor(min_scales, device=log_scales.device, dtype=log_scales.dtype).detach()
    visible = torch.as_tensor(visible_mask, device=log_scales.device)
    if anchors.shape != (count,) or visible.shape != (count,) or visible.dtype != torch.bool:
        raise ValueError("One scalar anchor and an explicit boolean visibility flag per Gaussian are required")
    if not torch.isfinite(log_scales).all() or not torch.isfinite(anchors).all():
        raise ValueError("Finite log-scales and anchors are required")
    if ((anchors < 0) & (anchors != -1)).any():
        raise ValueError("Anchors must be nonnegative; -1 is the explicit inactive sentinel")
    scale = log_scales.exp()
    if not torch.isfinite(scale).all():
        raise ValueError("Physical scales overflowed the input dtype")
    return scale, anchors[:, None], (visible & (anchors >= 0))[:, None]


def scale_anchor_potential(log_scales, min_scales, visible_mask, upper_weight=.05, lower_weight=.1):
    """Summed potential with the observed log-scale gradient and detached anchor.

    For physical scale s, stored scalar anchor m and delta=s-m:
      .5 * sum(.1*min(delta,0)^2 + upper_weight*max(delta,0)^2).
    No Gaussian-count, axis-count or image-size mean is introduced. Only the
    explicitly visible active rows contribute. The original passes lambda
    directly into i_Od9; normalization cannot be guessed from the field's name.
    """
    import torch
    scale, anchor, active = _inputs(log_scales, min_scales, visible_mask, upper_weight, lower_weight)
    delta = scale - anchor
    potential = .5 * (lower_weight * delta.clamp_max(0).square() + upper_weight * delta.clamp_min(0).square())
    return torch.where(active, potential, 0).sum()


def scale_anchor_gradient(log_scales, min_scales, visible_mask, upper_weight=.05, lower_weight=.1):
    """Return the independent direct translation of i_Od9's added gradient.

    This does not apply an optimizer update, mutate model state or consume RNG.
    Anchors/visibility are frozen inputs; derivatives through their geometry
    producer would add terms absent from the recovered backward kernel.
    """
    import torch
    with torch.no_grad():
        scale, anchor, active = _inputs(log_scales, min_scales, visible_mask, upper_weight, lower_weight)
        delta = scale - anchor
        gradient = scale * (lower_weight * delta.clamp_max(0) + upper_weight * delta.clamp_min(0))
        return torch.where(active, gradient, 0)


def regularization_loss(log_scales, anchors, visible_mask, upper_coefficient=.05, lower_coefficient=.1):
    """Canonical integration API; sum reduction and detached anchors are fixed."""
    return scale_anchor_potential(log_scales, anchors, visible_mask, upper_coefficient, lower_coefficient)


def regularization_gradient(log_scales, anchors, visible_mask, upper_coefficient=.05, lower_coefficient=.1):
    """Canonical explicit log-scale gradient, with no optimizer/model mutation."""
    return scale_anchor_gradient(log_scales, anchors, visible_mask, upper_coefficient, lower_coefficient)


def add_regularization_gradient_(log_scales, anchors, visible_mask, upper_coefficient=.05, lower_coefficient=.1):
    """Add to scale.grad AFTER photometric backward and BEFORE optimizer.step.

    Avoids constructing an additional autograd loss graph. Only this explicit
    helper mutates scale.grad; it never changes the parameter or optimizer state.
    The original lower coefficient is fixed0.1; other values are experiments.
    """
    import torch
    with torch.no_grad():
        gradient = regularization_gradient(log_scales, anchors, visible_mask, upper_coefficient, lower_coefficient)
        if log_scales.grad is None:
            log_scales.grad = gradient
        else:
            log_scales.grad.add_(gradient)


def fast_add_regularization_gradient_(log_scales, anchors, visible_mask, upper=.05, lower=.1):
    """Production opt-in: metadata checks only, no GPU scalar reads/synchrony.

    Preconditions: finite validated log-scales; finite anchors>=0 or exactly-1;
    explicit per-view boolean visibility; same device/dtype; finite nonnegative
    Python scalar coefficients. Caller retains its normal finite-loss/error
    checks. Use regularization_gradient/add_regularization_gradient_ for checked
    external inputs. No detach through anchors is differentiated in this no-grad
    update. This helper changes only grad, not parameters/optimizer/RNG.
    """
    import torch
    if (not isinstance(log_scales, torch.Tensor) or log_scales.ndim != 2
            or log_scales.shape[1] != 3 or not log_scales.is_floating_point()):
        raise ValueError("log_scales must be a floating Nx3 tensor")
    if (not isinstance(anchors, torch.Tensor) or anchors.shape != (len(log_scales),)
            or anchors.device != log_scales.device or anchors.dtype != log_scales.dtype):
        raise ValueError("Anchors must have matching shape/device/dtype")
    if (not isinstance(visible_mask, torch.Tensor) or visible_mask.shape != (len(log_scales),)
            or visible_mask.device != log_scales.device or visible_mask.dtype != torch.bool):
        raise ValueError("Explicit boolean visibility must share the model device")
    if any(isinstance(value, torch.Tensor) or not math.isfinite(float(value)) or value < 0 for value in (upper, lower)):
        raise ValueError("Coefficients must be finite nonnegative Python scalars")
    with torch.no_grad():
        scale = log_scales.exp()
        delta = scale - anchors[:, None]
        gradient = scale * (lower * delta.clamp_max(0) + upper * delta.clamp_min(0))
        gradient = torch.where((visible_mask & (anchors >= 0))[:, None], gradient, 0)
        if log_scales.grad is None:
            log_scales.grad = gradient
        else:
            log_scales.grad.add_(gradient)


def compute_min_scale_anchors(params, cameras, train_ids, min_scale_px=.25, *, projector=None):
    """Return detached[N] anchors using only the explicitly supplied train views.

    The observed formula is min(z_camera*min_scale_px/max(fx_pixels,fy_pixels))
    over views whose projected Gaussian support intersects the image. Native
    i_aN8 computes support in its own rasterizer. This reference uses public
    gsplat fully_fused_projection for SUPPORT VISIBILITY ONLY; its exact clipping
    and radius rounding are a documented substitute, not recovered native code.

    cameras.w2c is RDF (positive camera depth), with the independent engine's
    canonical world coordinates; do not pass Three.js camera matrixWorldInverse.
    Original epoch refresh is separate: call at initialization and eachN train
    samples, then inherit anchors during clone/split until the next refresh.
    No source RGB or holdout view is read. Invalid anchors are the sentinel-1.
    CPU tests inject a projector stub; default gsplat projection requires CUDA.
    """
    import numpy as np
    import torch
    if not math.isfinite(float(min_scale_px)) or min_scale_px <= 0:
        raise ValueError("min_scale_px must be finite and positive")
    ids = list(train_ids)
    if (not ids or len(ids) != len(set(ids))
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(cameras) for i in ids)):
        raise ValueError("Explicit nonempty unique training-camera indices are required")
    means = params["means"]
    if means.ndim != 2 or means.shape[1] != 3 or not means.is_floating_point():
        raise ValueError("means must be an Nx3 floating tensor")
    # Camera records are CPU JSON/NumPy data. Validate before entering the GPU
    # loop so no per-camera isfinite()/bool() forces a device synchronization.
    prepared = []
    for index in ids:
        camera = cameras[index]
        w2c, K = np.asarray(camera["w2c"], dtype=np.float64), np.asarray(camera["K"], dtype=np.float64)
        if (w2c.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(w2c).all()
                or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise ValueError("Camera must have finite RDF w2c and positive pixel focal lengths")
        width, height = int(camera["width"]), int(camera["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Positive camera image dimensions are required")
        prepared.append((w2c, K, width, height, float(max(K[0, 0], K[1, 1]))))
    if projector is None:
        if means.device.type != "cuda":
            raise RuntimeError("The public substitute visibility projector requires CUDA; provide an explicit CPU test projector")
        from gsplat.cuda._wrapper import fully_fused_projection
        projector = fully_fused_projection
    with torch.no_grad():
        scales = params["scales"].exp()
        opacity = params["opacities"].sigmoid()
        anchors = torch.full((len(means),), float("inf"), dtype=means.dtype, device=means.device)
        for cpu_w2c, cpu_K, width, height, max_focal in prepared:
            w2c = torch.as_tensor(cpu_w2c, dtype=means.dtype, device=means.device)
            K = torch.as_tensor(cpu_K, dtype=means.dtype, device=means.device)
            radii = projector(means=means, covars=None, quats=params["quats"], scales=scales,
                viewmats=w2c[None], Ks=K[None], width=width, height=height,
                near_plane=.1, far_plane=1e8, eps2d=.3, packed=False, opacities=opacity)[0][0]
            supported = (radii > 0).all(-1) if radii.ndim == 2 else radii > 0
            depth = means @ w2c[2, :3] + w2c[2, 3]
            valid = supported & torch.isfinite(depth) & (depth >= .1)
            candidate = depth * (min_scale_px / max_focal)
            anchors = torch.minimum(anchors, torch.where(valid, candidate, float("inf")))
        return torch.where(torch.isfinite(anchors), anchors, -1).detach()

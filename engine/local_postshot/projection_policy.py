"""Explicit frozen projection settings over the independent public rasterizer.

The recovered native training near plane is camera-focal-dependent; the normal
GUI evaluation camera uses its own clip plane. No original kernel is executed.
"""
from __future__ import annotations
import math
import numpy as np


DEFAULT_PROJECTION = {"schema": 1, "train_near_mode": "fixed", "train_near_plane": .01,
                      "eval_near_plane": .01, "far_plane": 1e8, "eps2d": .3,
                      "far_mode": "fixed", "rasterize_mode": "classic",
                      "train_rasterize_mode":None,"eval_rasterize_mode":None,
                      "train_background":"black","compositor":"gsplat"}


def validated_projection(value=None):
    if value is not None and not isinstance(value, dict):
        raise ValueError("Projection policy must be an explicit object")
    # Optional v6 controls are deliberately absent from older normalized recipes.
    # A saved v5 projection is never enriched on restore.
    if set(value or {}) - (set(DEFAULT_PROJECTION) | {"tile_culling", "aa_compensation"}):
        raise ValueError("Unknown projection policy fields")
    cfg = {**DEFAULT_PROJECTION, **(value or {})}
    if cfg["schema"] != 1 or cfg["train_near_mode"] not in ("fixed", "native_focal_v1"):
        raise ValueError("Unsupported frozen projection policy")
    if cfg["rasterize_mode"] not in ("classic", "antialiased"):
        raise ValueError("Unsupported rasterization mode")
    if cfg["far_mode"] not in ("fixed", "unbounded_perspective"):
        raise ValueError("Unsupported far clipping mode")
    for name in ("train_rasterize_mode","eval_rasterize_mode"):
        if cfg[name] not in (None,"classic","antialiased"):raise ValueError("Unsupported stage rasterization mode")
    if cfg["train_background"] not in ("black","random_rgb_v1"):
        raise ValueError("Unsupported training background policy")
    if cfg["compositor"] not in ("gsplat","native_v1"):
        raise ValueError("Unsupported compositor")
    if cfg.get("tile_culling", "public") not in ("public", "conservative_v1", "native_exact_v1"):
        raise ValueError("Unsupported frozen tile culling")
    if cfg.get("aa_compensation", "public") not in ("public", "native_floor_v1"):
        raise ValueError("Unsupported frozen AA compensation")
    if (cfg.get("tile_culling", "public") != "public" or cfg.get("aa_compensation", "public") != "public") and cfg["compositor"] != "native_v1":
        raise ValueError("Private tile/AA policies require the paired native compositor")
    for name in ("train_near_plane", "eval_near_plane", "far_plane", "eps2d"):
        if isinstance(cfg[name], bool) or not math.isfinite(float(cfg[name])) or cfg[name] <= 0:
            raise ValueError(f"Projection {name} must be finite and positive")
        cfg[name] = float(cfg[name])
    if cfg["far_plane"] <= max(cfg["train_near_plane"], cfg["eval_near_plane"], .3):
        raise ValueError("Far plane must exceed near clipping")
    return cfg


def projection_artifact_identity(policy):
    """Freeze optional native wheels separately from the unchanged v5 compositor.

    Old/public recipes return an empty record without loading new extensions.
    This is also used before resume, so unavailable/mismatched code fails before
    mutating optimizer/model state. Identity helpers must not initialize CUDA.
    """
    result = {}
    if policy and policy.get("tile_culling", "public") != "public":
        from .splat3_tiles import tiles_identity
        result["tiles"] = tiles_identity()
    if policy and policy.get("aa_compensation", "public") != "public":
        from .splat3_projection import projection_identity
        result["aa_projection"] = projection_identity()
    return result


def native_training_near(camera):
    """clamp(4*.036f*max(abs(fx),abs(fy))/W,.1f,.3f).

    Both axes use WIDTH in the final denominator. Input K is in pixels. Original
    camera construction and FMA rounding are not claimed bit-identical to this
    independent float32 evaluation. This is separate from the anchor's fixed.1.
    """
    K = np.asarray(camera["K"], dtype=np.float64)
    width, height = int(camera["width"]), int(camera["height"])
    if K.shape != (3, 3) or not np.isfinite(K).all() or width <= 0 or height <= 0:
        raise ValueError("Finite pixel K and positive image dimensions required")
    f = np.float32
    x = f(f(.036) * f(f(K[0, 0]) / f(width)))
    y = f(f(f(f(K[1, 1]) / f(height)) * f(height)) * f(.036))
    y = f(y / f(width))
    return float(f(min(f(.3), max(f(.1), f(f(4) * max(abs(x), abs(y)))))))


def projection_kwargs(camera, policy=None, *, training=False):
    # Old checkpoints with no policy take exactly the historical public route.
    if policy is None:
        return {"near_plane": .01, "far_plane": 1e8}
    if training and policy["train_near_mode"] == "native_focal_v1":
        near = native_training_near(camera)
    else:
        near = policy["train_near_plane"] if training else policy["eval_near_plane"]
    mode=policy.get("train_rasterize_mode" if training else "eval_rasterize_mode") or policy["rasterize_mode"]
    return {"near_plane": near, "far_plane": float("inf") if policy["far_mode"]=="unbounded_perspective" else policy["far_plane"],
            "eps2d": policy["eps2d"], "rasterize_mode": mode}


def position_spatial_factor(cameras, train_ids, mode="public_all_median"):
    if mode not in ("public_all_median", "native_train_mean_v1"):
        raise ValueError("Unknown position learning-rate spatial context")
    selected = cameras if mode == "public_all_median" else [cameras[i] for i in train_ids]
    if not selected:
        raise ValueError("Position context requires camera centers")
    centers = np.stack([np.linalg.inv(np.asarray(c["w2c"], dtype=np.float64))[:3, 3] for c in selected])
    if not np.isfinite(centers).all():
        raise ValueError("Finite camera centers required")
    if mode == "public_all_median":
        return max(float(np.linalg.norm(centers - np.median(centers, axis=0), axis=-1).max()), .001)
    focal = np.mean([float(c["K"][0][0]) / int(c["width"]) for c in selected])
    radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=-1).max())
    # Equivalent native formula; original reduction order/float rounding differ.
    return radius * float(np.float32(1.1)) / max(float(focal), 1.)


def position_scheduler(optimizer, horizon, phase="post_step_legacy"):
    """Preserve old schedules or use native completed-step clamped decay.

    LambdaLR's initial epoch0 supplies native t=1 to the first Adam call. After
    t updates it prepares t+1; checkpoint restoration preserves that next LR.
    The native default horizon is30000, but an original user can override it.
    """
    import torch
    if phase=="post_step_legacy":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=math.exp(math.log(.01)/horizon))
    if phase!="completed_step_native":raise ValueError("Unknown position LR phase")
    return torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda epoch:math.exp(math.log(.01)*min((epoch+1)/horizon,1.)))

"""Versioned independent training recipes over public gsplat APIs.

This is an independent public-algorithm trainer, not a claim that the complete
proprietary Splat3 algorithm has been recovered. Explicit opt-in fields can use
mathematical behavior verified from retained current-version static evidence.
The quality_v2 recipe must be measured on a new project. Existing checkpoints
without a recipe continue through training.py's legacy path and saved schedules.

Primary references (read 2026-10-15):
  gsplat v1.5.3 examples/simple_trainer.py, create_splats_with_optimizers and loss
  https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py
  https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/default.py
  Gaussian SSIM definition, sigma1.5/window11:
  https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py

No source is imported from the original product. Original public 3DGS source is
used as algorithm evidence only; this implementation uses independently written
Torch operations and gsplat's public parameter/optimizer mutation functions.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math

import numpy as np


def recipe_config(name: str = "legacy_adc", steps: int = 30000, overrides: dict | None = None,
                  *, train_image_count: int | None = None) -> dict:
    """Return JSON-serializable config, with a frozen original schedule horizon.

    Flat overrides may change strategy fields (e.g. prune_scale2d, absgrad),
    loss fields (ssim_backend, ssim_lambda), initialization, or explicit optional
    regularizers. Unknown fields fail. Save the entire returned config in the
    checkpoint; never regenerate it when resuming or extending target steps.

    observed_v3 binds reset cadence and pause to the actual RGB training split.
    Without train_image_count it returns a validation-only sentinel that cannot
    initialize a model or strategy. Explicit reset_every is a requested interval;
    its effective value remains at least2*N. Unless explicitly overridden, the
    tail margin is derived AFTER that effective interval is resolved. An explicit
    pause or margin override is preserved and recorded in the frozen recipe.
    """
    aliases = {"legacy": "legacy_adc", "legacy_adc": "legacy_adc",
               "v2": "quality_v2", "public-v2": "quality_v2", "quality_v2": "quality_v2",
               "observed_v3": "observed_v3", "observed_v4": "observed_v4", "observed_v5":"observed_v5", "observed_v6":"observed_v6"}
    if name not in aliases or isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("Unknown recipe or invalid positive integer step horizon")
    name, steps = aliases[name], int(steps)
    if train_image_count is not None:
        if (isinstance(train_image_count, bool) or int(train_image_count) != train_image_count
                or train_image_count <= 0):
            raise ValueError("train_image_count must be the positive integer training-split size")
        train_image_count = int(train_image_count)
    quality = name in ("quality_v2", "observed_v3", "observed_v4", "observed_v5","observed_v6")
    factor = steps / 30000
    strategy = {"prune_opa": .005, "grow_grad2d": .0002, "grow_scale3d": .01,
                "grow_scale2d": .05, "prune_scale3d": .1, "prune_scale2d": .15,
                "refine_start_iter": round(500 * factor), "refine_stop_iter": round(15000 * factor),
                "refine_every": max(1, round(100 * factor)), "reset_every": max(1, round(3000 * factor)),
                "pause_refine_after_reset": 0, "absgrad": False, "revised_opacity": False,
                "refine_scale2d_stop_iter": round(15000 * factor) if quality else 0,
                "verbose": False}
    if quality:
        # Fixed candidate schedule, informed by the retained original GUI/CUDA
        # timeline (runtime-schedule.json), NOT recovery of a kernel formula.
        # Keep the public 3DGS 500/100/3000 schedule independent of max_steps.
        # Reset-step refinement is allowed below, then reset; subsequent 400
        # iterations are paused, with the next refinement at reset+500.
        strategy.update(refine_start_iter=500, refine_stop_iter=min(15000, steps),
                        refine_every=100, reset_every=3000, pause_refine_after_reset=500,
                        refine_scale2d_stop_iter=min(15000, steps))
    config = {"schema": 1, "name": name, "schedule_steps": steps,
              "init_mode": "public_rms", "init_opacity": .1, "init_scale": 1., "init_scale_floor": 1e-6,
              "init_rotation": "identity", "ssim_kind": "gaussian11" if quality else "uniform11",
              "ssim_padding": "valid" if quality else "same",
              "ssim_backend": "auto" if quality else "torch",
              "ssim_lambda": .2, "opacity_reg": 0., "scale_reg": 0.,
              "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-15,
              "opacity_reset_multiplier": 2., "reset_stop_margin": 0,
              "cache_trim_enabled": True, "cache_trim_min_reserved_mb": 4096,
              "cache_trim_min_slack_mb": 1024,
              "global_screen_guard_every": 0, "global_screen_guard_threshold": .15,
              "state_accumulation_backend": "upstream",
              "camera_tensor_cache": False, "deduplicate_final_artifacts": False,
              "projection": None, "position_lr_horizon": steps,
              "position_lr_spatial": "public_all_median", "position_lr_phase":"post_step_legacy",
              "strategy": strategy, "cap_policy": "bounded_screen_v2" if quality else "gradient_budget"}
    if name in ("observed_v3", "observed_v4", "observed_v5","observed_v6"):
        strategy.update(prune_opa=5 / 256,
                        pause_refine_after_reset=train_image_count or 0)
        config.update(init_mode="splat3_observed", ssim_padding="same", adam_beta2=.997,
                      opacity_reset_multiplier=1.5, train_image_count=train_image_count,
                      train_image_count_source="caller_training_split" if train_image_count is not None else "unresolved_validation_only",
                      training_binding_status="resolved" if train_image_count is not None else "validation_only",
                      reset_interval_requested=3000,
                      unrecovered_original_behavior=[
                          "scale_rel_l2reg=.05 formula, normalization and application remain unimplemented",
                          "clone/split weight statistics and complete selection remain public substitutes",
                          "point-cloud color encoding is unresolved; known COLMAP sRGB is not gamma-converted",
                          "original RNG seed and draw order are not reproduced"])
    if name in ("observed_v4","observed_v5","observed_v6"):
        config.update(cap_policy="observed_error_v4", contribution_backend="native_cuda_v1",
                      native_min_error=.1, native_prune_min_weight=5/256,
                      scale_anchor_regularization=True, scale_rel_l2reg=.05,
                      scale_lower_l2reg=.1, min_scale_px=.25,
                      anchor_visibility="gsplat_projection_v1", camera_tensor_cache=True,
                      deduplicate_final_artifacts=True,
                      unrecovered_original_behavior=[
                          "Projection, alpha cap and support culling use public gsplat, not original PTX",
                          "Torch RNG and cumulative-sum rounding differ from original PCG/scan kernels",
                          "Inactive slots are physically compacted, not retained in original DynTable",
                          "Point-color producer encoding and optional appearance/opacity regularizers remain unresolved"])
    if name in ("observed_v5","observed_v6"):
        # Published independent adaptation: the controlled R13 arm retains the
        # R8 training context and corrects the eval clip, with the new matched
        # compositor/statistics. Recovered native context remains opt-in. Saved
        # experimental v5 checkpoints always restore their full frozen recipe.
        config.update(contribution_backend="native_renderer_v1",scale_rel_l2reg=.00001,scale_lower_l2reg=.00002,
                      projection={"train_near_mode":"fixed","train_near_plane":.01,"eval_near_plane":.1,
                          "far_mode":"unbounded_perspective","train_rasterize_mode":"classic",
                          "eval_rasterize_mode":"classic","train_background":"black","compositor":"native_v1"},
                      unrecovered_original_behavior=[
                          "Projection/Jacobian/culling and AA floor use explicitly identified public gsplat substitutes",
                          "Torch RNG/scan rounding, physical inactive compaction and GPU arithmetic differ",
                          "Original effective per-project LR/AA overrides and source color producer remain unresolved",
                          "Recovered focal near/random background/AA/train-mean/30k contexts are explicit opt-ins; the measured default keeps R8 training context",
                          "Weak scale coefficients are independent adaptation, not original recovered coefficients"])
    if name=="observed_v6":
        config["camera_center_cache"]=True
        config["projection"]["tile_culling"]="conservative_v1"
        config["unrecovered_original_behavior"].append("Conservative tile rejection preserves the public support; exact native support/AA are explicit opt-ins")
    immutable = ("schema", "name", "schedule_steps", "strategy", "cap_policy", "train_image_count",
                 "train_image_count_source", "training_binding_status", "reset_interval_requested",
                 "unrecovered_original_behavior")
    for key, value in (overrides or {}).items():
        if key in strategy:
            strategy[key] = value
        elif key in config and key not in immutable:
            config[key] = value
        else:
            raise ValueError(f"Unknown or immutable recipe override: {key}")
    if name in ("observed_v3", "observed_v4", "observed_v5","observed_v6"):
        requested = strategy["reset_every"]
        if isinstance(requested, bool) or int(requested) != requested or requested < 1:
            raise ValueError("Recipe reset_every must be a positive integer requested reset interval")
        config["reset_interval_requested"] = int(requested)
        strategy["reset_every"] = max(int(requested), 2 * train_image_count) if train_image_count is not None else int(requested)
        if "reset_stop_margin" not in (overrides or {}):
            config["reset_stop_margin"] = strategy["reset_every"] // 2
    if config["init_mode"] not in ("public_rms", "splat3_observed"):
        raise ValueError("Initialization mode must be public_rms or splat3_observed")
    if config["init_mode"] == "splat3_observed":
        if "init_scale_floor" not in (overrides or {}):
            config["init_scale_floor"] = 1e-7
        if "init_rotation" not in (overrides or {}):
            config["init_rotation"] = "shoemake"
    for key in ("prune_opa", "grow_grad2d", "grow_scale3d", "grow_scale2d", "prune_scale3d", "prune_scale2d"):
        if not math.isfinite(float(strategy[key])) or float(strategy[key]) <= 0:
            raise ValueError(f"Recipe {key} must be finite and positive")
    if not 0 < strategy["prune_opa"] < .5:
        raise ValueError("prune_opa must be between0 and .5 so opacity reset stays finite")
    for key in ("refine_start_iter", "refine_stop_iter", "refine_every", "reset_every", "pause_refine_after_reset", "refine_scale2d_stop_iter"):
        if isinstance(strategy[key], bool) or int(strategy[key]) != strategy[key] or strategy[key] < 0:
            raise ValueError(f"Recipe {key} must be a nonnegative integer")
    if strategy["refine_every"] < 1 or strategy["reset_every"] < 1:
        raise ValueError("Refine and opacity-reset periods must be positive")
    if strategy["pause_refine_after_reset"] >= strategy["reset_every"]:
        raise ValueError("Opacity recovery pause must be shorter than reset interval")
    if config["ssim_kind"] not in ("gaussian11", "uniform11") or config["ssim_padding"] not in ("valid", "same"):
        raise ValueError("Unsupported explicit SSIM definition")
    if config["ssim_backend"] not in ("auto", "torch", "fused") or not 0 <= config["ssim_lambda"] <= 1:
        raise ValueError("Invalid SSIM backend or mixture weight")
    if config["state_accumulation_backend"] not in ("upstream","dense_single_v1"):
        raise ValueError("Unsupported explicit strategy state backend")
    if config["projection"] is not None:
        from .projection_policy import validated_projection
        config["projection"] = validated_projection(config["projection"])
    if (isinstance(config["position_lr_horizon"],bool) or int(config["position_lr_horizon"])!=config["position_lr_horizon"]
            or config["position_lr_horizon"]<1):
        raise ValueError("Position LR horizon must be a positive integer")
    if config["position_lr_spatial"] not in ("public_all_median","native_train_mean_v1"):
        raise ValueError("Unsupported position LR spatial context")
    if config["position_lr_phase"] not in ("post_step_legacy","completed_step_native"):
        raise ValueError("Unsupported position LR step phase")
    for key in ("camera_tensor_cache","deduplicate_final_artifacts"):
        if not isinstance(config[key],bool):raise ValueError(f"{key} must be boolean")
    if "camera_center_cache" in config and not isinstance(config["camera_center_cache"],bool):
        raise ValueError("camera_center_cache must be boolean")
    if name in ("observed_v4","observed_v5","observed_v6"):
        if config["contribution_backend"] not in ("native_cuda_v1","native_renderer_v1") or config["anchor_visibility"]!="gsplat_projection_v1":
            raise ValueError("Unsupported frozen native-statistics or anchor backend")
        native_renderer=bool(config["projection"] and config["projection"].get("compositor")=="native_v1")
        if native_renderer!=(config["contribution_backend"]=="native_renderer_v1"):
            raise ValueError("Compositor and contribution statistics must use identical frozen semantics")
        if native_renderer and strategy["absgrad"]:raise ValueError("Native compositor does not implement absgrad")
        if config["ssim_padding"]!="same" or config["ssim_kind"]!="gaussian11" or config["ssim_lambda"]!=.2:
            raise ValueError("Observed error statistics require the explicit Gaussian11 same/.2 map")
        for key in ("native_min_error","native_prune_min_weight","scale_rel_l2reg","scale_lower_l2reg","min_scale_px"):
            if not math.isfinite(float(config[key])) or config[key]<0:raise ValueError(f"Invalid v4 {key}")
        if not 0<config["native_prune_min_weight"]<1 or config["min_scale_px"]<=0:
            raise ValueError("V4 requires a valid contribution threshold and positive pixel anchor")
        if not isinstance(config["scale_anchor_regularization"],bool):raise ValueError("Explicit regularization flag required")
    if not 0 < config["init_opacity"] < 1 or config["init_rotation"] not in ("identity", "shoemake"):
        raise ValueError("Initialization requires finite opacity and identity/Shoemake quaternions")
    for key in ("init_scale", "init_scale_floor", "global_screen_guard_threshold"):
        if not math.isfinite(float(config[key])) or config[key] <= 0:
            raise ValueError(f"Recipe {key} must be positive")
    for key in ("opacity_reg", "scale_reg"):
        if not math.isfinite(float(config[key])) or config[key] < 0:
            raise ValueError(f"Recipe {key} must be finite and nonnegative")
    for key in ("adam_beta1", "adam_beta2"):
        if not math.isfinite(float(config[key])) or not 0 <= config[key] < 1:
            raise ValueError(f"Recipe {key} must be finite and between0 and1 (exclusive1)")
    if not math.isfinite(float(config["adam_epsilon"])) or config["adam_epsilon"] <= 0:
        raise ValueError("Recipe adam_epsilon must be finite and positive")
    multiplier = config["opacity_reset_multiplier"]
    if not math.isfinite(float(multiplier)) or multiplier <= 0 or multiplier * strategy["prune_opa"] >= 1:
        raise ValueError("Opacity reset multiplier must produce a finite alpha cap strictly between0 and1")
    margin = config["reset_stop_margin"]
    if isinstance(margin, bool) or int(margin) != margin or margin < 0:
        raise ValueError("Opacity reset stop margin must be a nonnegative integer")
    if not isinstance(config["cache_trim_enabled"], bool):
        raise ValueError("cache_trim_enabled must be a boolean")
    for key in ("cache_trim_min_reserved_mb", "cache_trim_min_slack_mb"):
        if not math.isfinite(float(config[key])) or config[key] < 0:
            raise ValueError(f"Recipe {key} must be finite and nonnegative")
    if int(config["global_screen_guard_every"]) != config["global_screen_guard_every"] or config["global_screen_guard_every"] < 0:
        raise ValueError("Optional global guard period must be a nonnegative integer")
    if strategy["absgrad"] and "grow_grad2d" not in (overrides or {}):
        # Public gsplat documents this separate threshold for AbsGS. It remains
        # opt-in and rasterization(absgrad=True) must ALSO be enabled by caller.
        strategy["grow_grad2d"] = .0008
    return config


def recipe_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def recipe_from_checkpoint(state: dict) -> dict | None:
    """None means exact old code path, not a guessed recipe built from defaults."""
    value = state.get("training_recipe")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schema") != 1 or value.get("name") not in ("legacy_adc", "quality_v2", "observed_v3", "observed_v4", "observed_v5","observed_v6"):
        raise ValueError("Unsupported saved training recipe; refuse silent resume changes")
    require_training_recipe_ready(value)
    return deepcopy(value)


def require_training_recipe_ready(recipe: dict) -> None:
    """Reject unresolved v3 validation sentinels; never infer checkpoint fields.

    Existing v2/legacy records are intentionally not enriched or regenerated.
    The caller must supply len(train_indices), not all reconstruction cameras,
    to recipe_config when starting a fresh observed_v3 training run.
    """
    if recipe.get("name") not in ("observed_v3","observed_v4","observed_v5","observed_v6"):
        return
    count = recipe.get("train_image_count")
    if (count is None or isinstance(count, bool) or not isinstance(count, int) or count < 1
            or recipe.get("training_binding_status") != "resolved"
            or not isinstance(recipe.get("train_image_count_source"), str)
            or not recipe["train_image_count_source"]):
        raise ValueError("observed_v3 training requires explicit train_image_count from the actual training split; validation-only or incomplete recipes cannot train")
    interval = recipe.get("strategy", {}).get("reset_every")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 2 * count:
        raise ValueError("Saved observed_v3 effective reset interval must be at least2*train_image_count; refuse silent schedule reconstruction")


def initial_arrays(points, colors, degree: int, max_splats: int, recipe: dict, rng=None) -> dict:
    """Public baseline or explicitly observed current-cache initialization math.

    public_rms preserves the previous public baseline. splat3_observed uses the
    arithmetic3NN distance mean, then max(distance,1e-7)/1.5, then natural log,
    directly traced in current-cache i_iG8/i_YH8 (init-loss-initialization-math
    evidence). NumPy Shoemake rotations match the distribution, NOT the original
    PCG state/seed/order. Opacity .1 also matches the statically established
    original shared default; active user overrides were not instrumented.
    No unverified gamma conversion is applied.
    Inputs are RGB in0..1, never inferred byte units.
    """
    require_training_recipe_ready(recipe)
    from scipy.spatial import cKDTree
    points, colors = np.asarray(points), np.asarray(colors)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or len(points) < 4:
        raise ValueError("At least four points with matching Nx3 RGB colors are required")
    if not np.isfinite(points).all() or not np.isfinite(colors).all() or np.any((colors < 0) | (colors > 1)):
        raise ValueError("Finite sparse positions and explicitly normalized RGB colors in0..1 are required")
    if degree not in range(5) or max_splats < 4:
        raise ValueError("SH degree0..4 and at least four allowed Gaussians are required")
    if len(points) > max_splats:
        indices = (rng or np.random).choice(len(points), max_splats, replace=False)
        points, colors = points[indices], colors[indices]
    distance = cKDTree(points).query(points, k=4, workers=-1)[0][:, 1:]
    if recipe.get("init_mode", "public_rms") == "splat3_observed":
        scale = np.maximum(distance.mean(axis=-1), recipe["init_scale_floor"]) / 1.5
        scale = np.log(scale * recipe["init_scale"])
    else:
        scale = np.sqrt(np.mean(distance ** 2, axis=-1)) * recipe["init_scale"]
        scale = np.log(np.maximum(scale, recipe["init_scale_floor"]))
    sh = np.zeros((len(points), (degree + 1) ** 2, 3), np.float32)
    sh[:, 0] = (colors - .5) / .28209479177387814
    quaternion = np.zeros((len(points), 4), np.float32)
    quaternion[:, 0] = 1
    if recipe["init_rotation"] == "shoemake":
        # Uniform S3, stored in the independent engine's scalar-first wxyz.
        # Isotropic initial covariance is invariant to these rotations. The
        # original also uses three random variates but its PCG seed is unknown.
        uniform = (rng or np.random).random((len(points), 3))
        a, b = np.sqrt(1 - uniform[:, 0]), np.sqrt(uniform[:, 0])
        t, u = 2 * np.pi * uniform[:, 1], 2 * np.pi * uniform[:, 2]
        quaternion = np.stack([b * np.cos(u), a * np.sin(t), a * np.cos(t), b * np.sin(u)], axis=-1).astype(np.float32)
    opacity = math.log(recipe["init_opacity"] / (1 - recipe["init_opacity"]))
    return {"means": np.asarray(points, dtype=np.float32), "scales": np.repeat(scale[:, None], 3, 1).astype(np.float32),
            "quats": quaternion, "opacities": np.full(len(points), opacity, np.float32), "sh0": sh[:, :1], "shN": sh[:, 1:]}


@lru_cache(maxsize=1)
def _fused_ssim():
    try:
        from fused_ssim import fused_ssim
        return fused_ssim
    except (ImportError, OSError):
        return None


def resolve_ssim_backend(device="cuda", backend="auto", dtype=None) -> str:
    """Resolve once and SAVE the result; resume must not silently switch backend."""
    import torch
    if backend not in ("auto", "torch", "fused"):
        raise ValueError("Unknown SSIM backend")
    if backend == "torch":
        return "torch"
    compatible = torch.device(device).type == "cuda" and dtype in (None, torch.float32)
    available = compatible and _fused_ssim() is not None
    if backend == "fused" and not available:
        raise RuntimeError("Requested fused Gaussian SSIM requires its official CUDA extension and float32 CUDA inputs")
    return "fused" if backend != "torch" and available else "torch"


def training_ssim(prediction, target, backend="auto", kind="gaussian11", padding="valid"):
    """NHWC SSIM for training only; existing uniform11 evaluation is unchanged.

    Gaussian11 sigma1.5 with VALID support matches the public gsplat trainer.
    The pure Torch separable reference also runs on CPU and is differentiable.
    No clipping is applied to rendered colors, matching the public objective.
    """
    import torch
    import torch.nn.functional as F
    if prediction.ndim != 4 or prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("Training SSIM requires matching NHWC RGB batches")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("Training prediction and target must share device/dtype")
    if padding not in ("valid", "same"):
        raise ValueError("SSIM padding must be explicitly valid or same")
    if kind == "uniform11":
        size = min(11, prediction.shape[1], prediction.shape[2])
        size -= (size % 2 == 0)
        x, y = prediction.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
        mean = lambda t: F.avg_pool2d(t, size, 1, size // 2 if padding == "same" else 0)
        a, b = mean(x), mean(y)
        vx, vy, cov = mean(x * x) - a * a, mean(y * y) - b * b, mean(x * y) - a * b
    elif kind == "gaussian11":
        if min(prediction.shape[1:3]) < 11:
            raise ValueError("Gaussian11 SSIM requires images of at least11 pixels on each axis")
        chosen = resolve_ssim_backend(prediction.device, backend, prediction.dtype)
        x, y = prediction.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
        if chosen == "fused":
            return _fused_ssim()(x.contiguous(), y.contiguous(), padding=padding)
        coordinate = torch.arange(-5, 6, dtype=x.dtype, device=x.device)
        kernel = torch.exp(-.5 * (coordinate / 1.5) ** 2)
        kernel = kernel / kernel.sum()
        values = torch.cat([x, y, x * x, y * y, x * y], dim=1)
        channels = values.shape[1]
        pad = 5 if padding == "same" else 0
        values = F.conv2d(values, kernel[None, None, None, :].expand(channels, 1, 1, 11),
                          groups=channels, padding=(0, pad))
        values = F.conv2d(values, kernel[None, None, :, None].expand(channels, 1, 11, 1),
                          groups=channels, padding=(pad, 0))
        a, b, xx, yy, xy = values.chunk(5, dim=1)
        vx, vy, cov = xx - a * a, yy - b * b, xy - a * b
    else:
        raise ValueError("Unknown training SSIM definition")
    return (((2 * a * b + .01 ** 2) * (2 * cov + .03 ** 2)) /
            ((a * a + b * b + .01 ** 2) * (vx + vy + .03 ** 2))).mean()


def training_loss(prediction, target, params, recipe: dict) -> dict:
    import torch
    l1 = (prediction - target).abs().mean()
    score = training_ssim(prediction, target, backend=recipe["ssim_backend"], kind=recipe["ssim_kind"], padding=recipe["ssim_padding"])
    total = (1 - recipe["ssim_lambda"]) * l1 + recipe["ssim_lambda"] * (1 - score)
    result = {"l1": l1, "ssim": score}
    if recipe["opacity_reg"]:
        result["opacity_regularization"] = torch.sigmoid(params["opacities"]).mean() * recipe["opacity_reg"]
        total = total + result["opacity_regularization"]
    if recipe["scale_reg"]:
        result["scale_regularization"] = torch.exp(params["scales"]).mean() * recipe["scale_reg"]
        total = total + result["scale_regularization"]
    return {"loss": total, **result}


def trim_cuda_cache_if_needed(recipe: dict, device) -> dict | None:
    """Release unused allocator blocks only under explicit memory pressure.

    Runs after a refinement, not every optimizer step. It changes neither model
    tensors, optimizer state nor RNG. Thresholds are MiB; defaults are4GiB total
    reserved with more than1GiB reserved-but-unallocated memory. CPU is a no-op.
    """
    import torch
    device = torch.device(device)
    if not recipe.get("cache_trim_enabled", True) or device.type != "cuda":
        return None
    reserved, allocated = torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)
    minimum = recipe.get("cache_trim_min_reserved_mb", 4096) * 1048576
    slack = recipe.get("cache_trim_min_slack_mb", 1024) * 1048576
    if reserved > minimum and reserved > allocated + slack:
        torch.cuda.empty_cache()
        return {"reserved_before": reserved, "allocated": allocated, "unused_bytes_released_requested": reserved - allocated}
    return None


def make_v2_strategy(strategy_config: dict, max_splats: int):
    """Public ADC with screen rules, disjoint split/clone masks and a hard cap.

    Only training-frame projection metadata enters the strategy. Screen pruning
    is an explicit optional geometric heuristic: it can remove large background
    Gaussians and is not guaranteed to improve every scene. Setting
    refine_scale2d_stop_iter=0 disables BOTH screen growth and screen pruning.
    """
    require_training_recipe_ready(strategy_config)
    import torch
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import duplicate, remove, reset_opa, split
    config = strategy_config.get("strategy", strategy_config)
    reset_multiplier = strategy_config.get("opacity_reset_multiplier", 2.)
    reset_stop_margin = strategy_config.get("reset_stop_margin", 0)
    schedule_steps = strategy_config.get("schedule_steps")
    if reset_stop_margin and (schedule_steps is None or schedule_steps <= 0):
        raise ValueError("A nonzero reset-stop margin needs the full frozen recipe with schedule_steps")
    if isinstance(max_splats, bool) or int(max_splats) != max_splats or max_splats < 4:
        raise ValueError("At least four Gaussian slots are required")

    class BoundedScreenStrategy(DefaultStrategy):
        def _update_state(self,params,state,info,packed=False):
            if strategy_config.get("state_accumulation_backend","upstream")=="dense_single_v1":
                from .training_state_performance import update_single_camera_state
                if update_single_camera_state(state,info,key_for_gradient=self.key_for_gradient,
                        absgrad=self.absgrad,track_radii=self.refine_scale2d_stop_iter>0,packed=packed):
                    return
            return super()._update_state(params,state,info,packed=packed)

        @torch.no_grad()
        def _grow_gs(self, params, optimizers, state, step):
            n = len(params["means"])
            if n > max_splats:
                raise ValueError("Model already exceeds its frozen v2 Gaussian budget")
            gradients = state["grad2d"] / state["count"].clamp_min(1)
            high = gradients > self.grow_grad2d
            small = params["scales"].exp().amax(-1) <= self.grow_scale3d * state["scene_scale"]
            split_mask = high & ~small
            if step < self.refine_scale2d_stop_iter:
                split_mask |= state["radii"] > self.grow_scale2d
            # A small 3D Gaussian can still have a huge near-camera footprint.
            # Screen splitting takes precedence; cloning AND splitting the same
            # parent would consume two net slots and violate a one-slot budget.
            clone_mask = high & small & ~split_mask
            eligible = clone_mask | split_mask
            budget = max_splats - n
            if int(eligible.sum()) > budget:
                keep = torch.zeros_like(eligible)
                if budget:
                    scores = gradients.masked_fill(~eligible, -float("inf"))
                    keep[scores.topk(budget).indices] = True
                clone_mask &= keep
                split_mask &= keep
            clones, splits = int(clone_mask.sum()), int(split_mask.sum())
            if clones:
                duplicate(params, optimizers, state, clone_mask)
            if splits:
                split_mask = torch.cat([split_mask, torch.zeros(clones, dtype=torch.bool, device=split_mask.device)])
                split(params, optimizers, state, split_mask, revised_opacity=self.revised_opacity)
            if len(params["means"]) > max_splats:
                raise RuntimeError("V2 screen/gradient growth exceeded its hard Gaussian budget")
            return clones, splits

        @torch.no_grad()
        def step_post_backward(self, params, optimizers, state, step, info, packed=False):
            if step >= self.refine_stop_iter:
                return
            self._update_state(params, state, info, packed=packed)
            phase = step % self.reset_every
            allowed = phase == 0 or phase >= self.pause_refine_after_reset
            if step >= self.refine_start_iter and step > 0 and step % self.refine_every == 0 and allowed:
                # Prune first so dead/oversized GSs free budget for useful splits.
                # Allocator trimming below runs only if memory pressure exceeds
                # configured thresholds, rather than unconditionally per refine.
                removed = self._prune_gs(params, optimizers, state, step)
                if not len(params["means"]):
                    raise RuntimeError("V2 pruning removed all Gaussians; increase size thresholds or disable screen rules")
                clones, splits = self._grow_gs(params, optimizers, state, step)
                state["last_refinement"] = {"step": int(step), "pruned": int(removed), "cloned": clones, "split": splits,
                                            "count": len(params["means"])}
                for key in ("grad2d", "count", "radii"):
                    if state.get(key) is not None:
                        state[key].zero_()
                trim = trim_cuda_cache_if_needed(strategy_config, params["means"].device)
                if trim is not None:
                    state["cache_trim_count"] = state.get("cache_trim_count", 0) + 1
                    state["last_cache_trim"] = {"step": int(step), **trim}
            # Explicit Python boolean fixes the pinned upstream chained compare.
            # A zero margin preserves the original v2 behavior even for custom
            # refinement horizons beyond schedule_steps. An explicit margin is
            # an opt-in candidate based on the original tail-reset guard.
            tail_allows_reset = not reset_stop_margin or step + reset_stop_margin <= schedule_steps
            if step > 0 and step % self.reset_every == 0 and tail_allows_reset:
                reset_opa(params=params, optimizers=optimizers, state=state, value=self.prune_opa * reset_multiplier)
    return BoundedScreenStrategy(**config)


def training_camera_guard(params, cameras, train_indices, threshold=.15, projector=None):
    """Return (remove mask, diagnostics), using ONLY specified training cameras.

    This optional all-training-camera sweep is separate from minibatch ADC. It
    reads no RGB, holdout image, project or checkpoint and never mutates params.
    Caller may apply gsplat.strategy.ops.remove before refine-stop and save the
    decision/config in its checkpoint. Keep disabled for an isolated v2 ablation
    unless its additional cost and sky/background risk are explicitly desired.
    """
    import torch
    if not math.isfinite(float(threshold)) or not 0 < threshold <= 1:
        raise ValueError("Global screen radius threshold must be in0..1")
    indices = list(train_indices)
    if not indices or len(set(indices)) != len(indices) or any(not isinstance(i, int) or i < 0 or i >= len(cameras) for i in indices):
        raise ValueError("An explicit nonempty unique training-camera index list is required")
    if projector is None:
        from gsplat.cuda._wrapper import fully_fused_projection
        projector = fully_fused_projection
    means = params["means"]
    maximum = torch.zeros(len(means), dtype=means.dtype, device=means.device)
    seen = torch.zeros(len(means), dtype=torch.int32, device=means.device)
    with torch.no_grad():
        scales, opacity = params["scales"].exp(), params["opacities"].sigmoid()
        for index in indices:
            camera = cameras[index]
            radii = projector(means=means, covars=None, quats=params["quats"], scales=scales,
                viewmats=torch.as_tensor(camera["w2c"], dtype=means.dtype, device=means.device)[None],
                Ks=torch.as_tensor(camera["K"], dtype=means.dtype, device=means.device)[None],
                width=int(camera["width"]), height=int(camera["height"]), near_plane=.01,
                far_plane=1e8, packed=False, opacities=opacity)[0][0]
            visible = (radii > 0).all(-1)
            normalized = radii.amax(-1).to(means.dtype) / max(camera["width"], camera["height"])
            maximum = torch.maximum(maximum, torch.where(visible, normalized, 0))
            seen += visible.to(torch.int32)
    return maximum > threshold, {"max_normalized_radius": maximum, "visible_training_camera_count": seen,
                                  "training_camera_indices": indices, "threshold": threshold,
                                  "heldout_RGB_used": False, "parameters_mutated": False}

"""Render independent PLY/SPZ models at measured original-GUI output cameras.

Default mode validates immutable camera evidence and prints a CPU-only plan.
--execute-gpu explicitly enables forward-only gsplat rendering. It never starts
training, reads a checkpoint, changes a project, or loads original product code.
The output directory must be new; all source models and original PNGs remain
untouched. Each output manifest embeds verified Kout and w2c, so the PNG evaluator
recognizes the local PNG as already on Kout and performs no second resampling.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, PngImagePlugin

from evaluate_reference_pngs import (CalibrationError, digest, evidence_entries,
    intrinsic, matrix, pose_errors, read_json, read_png, resolve,
    verify_physical_intrinsics, write_json)


def validate(args):
    plan_path = args.plan.resolve(strict=True)
    contract_path = args.calibration.resolve(strict=True)
    inputs_path = args.input_manifest.resolve(strict=True)
    plan, contract, inputs = [read_json(p) for p in (plan_path, contract_path, inputs_path)]
    if plan.get("status") != "camera_verified" or contract.get("status") != "validated":
        raise CalibrationError("A verified immutable camera contract and evaluator-generated render plan are required")
    if plan.get("calibration_sha256") != digest(contract_path) or plan.get("input_manifest_sha256") != digest(inputs_path):
        raise CalibrationError("Camera plan hash no longer matches the camera contract/input manifest")
    if contract.get("color_space") != "sRGB" or contract.get("color_space_verified") is not True:
        raise CalibrationError("Original output sRGB verification is missing")
    evidence = evidence_entries(contract.get("camera_evidence", contract.get("evidence")))
    if not evidence:
        raise CalibrationError("Measured camera evidence is missing")
    for entry in evidence:
        path = resolve(entry["path"], contract_path.parent)
        if not entry.get("sha256") or digest(path) != entry["sha256"]:
            raise CalibrationError(f"Camera evidence changed: {path}")
    references = {row["index"]: row for group in inputs["sets"].values() for row in group["images"]}
    frames = {row["reference_index"]: row for row in contract["frames"]}
    plan_indices = [row["reference_index"] for row in plan["frames"]]
    if len(plan_indices) != len(frames) or set(plan_indices) != set(frames):
        raise CalibrationError("Render plan must cover each explicitly calibrated camera exactly once")
    for row in plan["frames"]:
        index = row["reference_index"]
        frame, camera, reference = frames[index], row["camera"], references[index]
        if row.get("camera_verified") is not True or any(frame.get(flag) is not True for flag in
                ("pose_verified", "K_verified", "resolution_verified", "frame_mapping_verified")):
            raise CalibrationError(f"Unverified camera/frame mapping: {index}")
        if row["name"] != frame["name"] or row["original_frame"] != frame["frame"]:
            raise CalibrationError(f"Camera plan changed explicit frame mapping: {index}")
        if int(camera["width"]) != int(frame["observed_width"]) or int(camera["height"]) != int(frame["observed_height"]):
            raise CalibrationError(f"Camera plan changed measured PNG resolution: {index}")
        k = intrinsic(camera["K"], "Plan K")
        if not np.allclose(k, intrinsic(frame["observed_K"], "Observed K"), atol=1e-10, rtol=0):
            raise CalibrationError(f"Camera plan changed measured K: {index}")
        if not np.allclose(matrix(camera["w2c"], (4, 4), "Plan w2c"), frame["observed_w2c_rub_world"], atol=1e-10, rtol=0):
            raise CalibrationError(f"Camera plan changed measured pose: {index}")
        error = pose_errors(camera["w2c"], reference["w2c_rub_world"])
        if error["camera_center_distance"] > 1e-5 or error["rotation_error_degrees"] > .001:
            raise CalibrationError(f"Measured pose differs from the reference input: {index}")
        verify_physical_intrinsics(frame, k, camera["width"], camera["height"])
        _, metadata = read_png(resolve(frame["png_path"], contract_path.parent), frame["sha256"])
        if (metadata["width"], metadata["height"]) != (camera["width"], camera["height"]):
            raise CalibrationError(f"Original PNG header changed: {index}")
    models = {}
    for spec in args.model:
        if "=" not in spec:
            raise CalibrationError("Use --model label=independent-model.ply")
        label, path_string = spec.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label) or label == "reference" or label in models:
            raise CalibrationError("Model labels must be unique simple names and cannot be 'reference'")
        path = Path(path_string).resolve(strict=True)
        if path.suffix.lower() not in (".ply", ".spz"):
            raise CalibrationError("Only independent standard 3DGS PLY/SPZ model files are accepted")
        models[label] = {"path": str(path), "sha256": digest(path)}
    if not models:
        raise CalibrationError("At least one independent model is required")
    if args.output_dir.exists():
        raise CalibrationError("The output directory must be new, preserving previous PNGs and source artifacts")
    provenance = {"plan": {"path": str(plan_path), "sha256": digest(plan_path)},
                  "calibration": {"path": str(contract_path), "sha256": digest(contract_path)},
                  "input_manifest": {"path": str(inputs_path), "sha256": digest(inputs_path)}}
    return plan, models, provenance


def execute(args, plan, models, provenance):
    # GPU libraries are deliberately not imported by validation/dry-run mode.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
    import torch
    from local_postshot.formats import load_cloud
    from local_postshot.training import render_parameters, tensors_from_cloud
    from local_postshot.projection_policy import validated_projection
    policy=getattr(args,"projection_policy",None)
    policy=validated_projection(policy) if policy is not None else None
    if not torch.cuda.is_available():
        raise RuntimeError("The isolated CUDA runtime/GPU is unavailable")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", **provenance, "models": {},
              "runtime": {"python": sys.executable, "torch": torch.__version__,
                          "gsplat": importlib.metadata.version("gsplat"), "device": torch.cuda.get_device_name()},
              "render_policy": {"training": False, "model_editing": False, "camera_source": "measured ABC and GUI output Kout",
                                "coordinates": "canonical RUB world to RDF camera; explicit measured w2c",
                                "background": "black", "SH": "all complete bands present in independent model",
                                "antialiasing": (policy.get("eval_rasterize_mode") or policy["rasterize_mode"]) if policy else "classic",
                                "PNG": "RGB8, clamp[0,1], round(value*255), encoded sRGB numeric values; no tone/exposure fitting",
                                "resample_existing_PNG": False}}
    result["render_policy"]["projection_policy"]=policy
    if policy and policy.get("compositor")=="native_v1":
        from local_postshot.splat3_renderer import renderer_identity
        result["runtime"]["compositor_artifact"]=renderer_identity()
    from local_postshot.projection_policy import projection_artifact_identity
    artifacts=projection_artifact_identity(policy)
    if artifacts:result["runtime"]["projection_artifacts"]=artifacts
    for label, source in models.items():
        torch.cuda.reset_peak_memory_stats()
        cloud = load_cloud(source["path"])
        if digest(Path(source["path"])) != source["sha256"]:
            raise CalibrationError("Independent model changed during loading")
        params = tensors_from_cloud(cloud)
        degree = round(math.sqrt(cloud["sh"].shape[1]) - 1)
        destination = output / label
        destination.mkdir()
        rows = []
        for frame in plan["frames"]:
            camera = frame["camera"]
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                image, alpha, info = render_parameters(params, camera, degree,projection_policy=policy)
                pixels = image[0].detach().cpu().numpy()
                alpha_min, alpha_max = float(alpha.min()), float(alpha.max())
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not np.isfinite(pixels).all():
                raise ValueError("Non-finite forward-render output")
            target = destination / f"eval_{frame['reference_index']:04d}.png"
            png_metadata = PngImagePlugin.PngInfo()
            png_metadata.add(b"sRGB", b"\x00")
            png_metadata.add_text("LocalPostshot/source_model_sha256", source["sha256"])
            png_metadata.add_text("LocalPostshot/camera_plan_sha256", provenance["plan"]["sha256"])
            temp = target.with_suffix(".png.tmp")
            Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(temp, format="PNG", pnginfo=png_metadata)
            temp.replace(target)
            row = {**frame, "png_path": str(target), "sha256": digest(target),
                   "camera_verified": True, "already_at_observed_Kout": True, "forward_seconds": elapsed,
                   "unclipped_encoded_RGB": {"minimum": float(pixels.min()), "maximum": float(pixels.max()),
                                             "out_of_range_channel_fraction": float(((pixels < 0) | (pixels > 1)).mean())},
                   "alpha": {"minimum": alpha_min, "maximum": alpha_max}}
            rows.append(row)
            print(json.dumps({"model": label, "reference_index": frame["reference_index"], "png": str(target)}, ensure_ascii=False), flush=True)
            del image, alpha, info
        manifest = {"status": "rendered", "source_model": source, **provenance,
                    "points": len(cloud["means"]), "SH_degree": degree,
                    "metric_input": "RGB8 PNG; differs from previous unclipped float-render evaluations",
                    "already_at_observed_Kout": True, "frames": rows,
                    "projection_policy":policy,
                    "forward_seconds_total": sum(row["forward_seconds"] for row in rows),
                    "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "memory_scope": "Torch process allocator, not whole process/WDDM VRAM"}
        if digest(Path(source["path"])) != source["sha256"]:
            raise CalibrationError("Source model changed during rendering")
        write_json(destination / "renders.json", manifest)
        result["models"][label] = {key: value for key, value in manifest.items() if key != "frames"}
        result["models"][label]["manifest"] = str(destination / "renders.json")
        del params, cloud
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    write_json(output / "rendering.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=INDEPENDENT_MODEL_PLY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-gpu", action="store_true", help="Run forward-only GPU work; default is CPU validation only")
    parser.add_argument("--projection-policy", default="null", help="Explicit JSON policy; default preserves historical public rendering")
    args = parser.parse_args(argv)
    try:
        args.projection_policy=json.loads(args.projection_policy)
        plan, models, provenance = validate(args)
        if not args.execute_gpu:
            print(json.dumps({"status": "validated_dry_run", "GPU_executed": False, "views": len(plan["frames"]),
                              "models": models, **provenance}, ensure_ascii=False))
            return 0
        result = execute(args, plan, models, provenance)
        print(json.dumps({"status": result["status"], "report": str(args.output_dir.resolve() / "rendering.json")}, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Rendering refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

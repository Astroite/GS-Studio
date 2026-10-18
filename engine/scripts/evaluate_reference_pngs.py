"""CPU-only comparison of normally exported Postshot PNGs and existing PNG renders.

No original application API, model, process memory, GPU, training or rendering is
used. A measured camera/frame contract is required; missing calibration produces
an uncalibrated report without quality numbers and exit status 2.

Example:
  python scripts/evaluate_reference_pngs.py --input-manifest input-manifest.json \
    --calibration camera-alignment-contract.json --gt-mapping intrinsics \
    --independent-project E:/AI/LocalPostshot/projects/SouthBuilding-Quality/project.json \
    --independent raw=raw-evaluation.json --independent clean=evaluation.json \
    --output reference-png-evaluation.json

Contract: status="validated", color_space="sRGB", color_space_verified=true,
camera_evidence=[{path,sha256}], frames=[{frame,reference_index,name,png_path,
sha256,observed_K,observed_w2c_rub_world,observed_width,observed_height,
pose_verified,K_verified,resolution_verified,frame_mapping_verified}].
Observed w2c maps canonical RUB world into RDF camera coordinates. It must be
recovered from the actual camera export and fixed coordinate conversion, not
copied from reference intrinsics/poses. Evidence hashes are checked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageCms
from scipy.ndimage import map_coordinates, minimum_filter, uniform_filter


class CalibrationError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(value: str, parent: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CalibrationError(f"Expected JSON object: {path}")
    return value


def matrix(value: Any, shape: tuple, label: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise CalibrationError(f"{label}: expected finite {shape} matrix")
    return a


def intrinsic(value: Any, label: str) -> np.ndarray:
    k = matrix(value, (3, 3), label)
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-10):
        raise CalibrationError(f"{label}: invalid pinhole intrinsics")
    return k


def pose_errors(observed: Any, expected: Any) -> dict:
    a = matrix(observed, (4, 4), "Observed w2c")
    b = matrix(expected, (4, 4), "Reference w2c")
    rotations, corrections, residuals = [], [], []
    for p in (a, b):
        if not np.allclose(p[3], [0, 0, 0, 1], atol=1e-9):
            raise CalibrationError("Camera w2c homogeneous row is invalid")
        if not np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=2e-5, rtol=0) or np.linalg.det(p[:3, :3]) <= 0:
            raise CalibrationError("Camera rotation must be a right-handed orthonormal matrix")
        # Float32 Alembic/Blender matrices retain tiny orthogonality errors.
        # acos(trace(Ra Rb^T)) amplifies those into a false angular mismatch.
        # Project only the angle calculation to SO(3); retain raw matrices for
        # geometry, preserve and report the correction, never fit either pose.
        u, _, vt = np.linalg.svd(p[:3, :3])
        rotation = u @ vt
        rotations.append(rotation)
        corrections.append(float(np.max(np.abs(rotation - p[:3, :3]))))
        residuals.append(float(np.max(np.abs(p[:3, :3] @ p[:3, :3].T - np.eye(3)))))
    ca, cb = np.linalg.inv(a), np.linalg.inv(b)
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotations[0] @ rotations[1].T) - 1) / 2, -1, 1))))
    return {"camera_center_distance": float(np.linalg.norm(ca[:3, 3] - cb[:3, 3])),
            "rotation_error_degrees": angle, "SO3_angle_only_max_corrections": corrections,
            "rotation_orthogonality_residuals": residuals}


def verify_physical_intrinsics(frame: dict, k: np.ndarray, width: int, height: int) -> dict:
    props = frame.get("camera_properties")
    if not isinstance(props, dict):
        raise CalibrationError("Camera contract needs measured ABC lens/sensor camera_properties")
    lens, sw, sh = [float(props[name]) for name in ("lens_mm", "sensor_width_mm", "sensor_height_mm")]
    if not all(math.isfinite(n) and n > 0 for n in (lens, sw, sh)):
        raise CalibrationError("Camera lens and sensor dimensions must be finite and positive")
    if abs(float(props.get("shift_x", float("nan")))) > 1e-8 or abs(float(props.get("shift_y", float("nan")))) > 1e-8:
        raise CalibrationError("This independently verified camera adapter supports measured zero lens shift only")
    if not math.isfinite(float(props.get("shift_x", float("nan")))) or not math.isfinite(float(props.get("shift_y", float("nan")))):
        raise CalibrationError("Measured camera lens shifts are required")
    derived = np.array([[width * lens / sw, 0, width / 2], [0, height * lens / sh, height / 2], [0, 0, 1]])
    error = float(np.max(np.abs(k - derived)))
    if error > .001:
        raise CalibrationError(f"Observed K differs from measured ABC lens/sensor/output resolution by {error} pixels")
    fov = np.degrees(2 * np.arctan(np.array([sw, sh]) / (2 * lens)))
    proof = {"camera_properties": props, "derived_K": derived.tolist(), "K_max_absolute_error_pixels": error,
             "ABC_fov_degrees": fov.tolist()}
    if "gui_fov_degrees" in frame:
        gui = matrix(frame["gui_fov_degrees"], (2,), "GUI horizontal/vertical FOV")
        tolerance = float(frame.get("gui_fov_rounding_half_interval_degrees", 5e-5))
        if not 0 < tolerance <= .001:
            raise CalibrationError("GUI FOV rounding precision must be explicit and at most .001 degree")
        if np.max(np.abs(gui - fov)) > tolerance + 1e-9:
            raise CalibrationError(f"GUI FOV is inconsistent with ABC lens/sensor: {gui.tolist()} vs {fov.tolist()}")
        proof.update({"GUI_fov_degrees": gui.tolist(), "GUI_rounding_half_interval_degrees": tolerance})
    return proof


def read_png(path: Path, expected_hash: str | None = None) -> tuple[np.ndarray, dict]:
    before = digest(path)
    if expected_hash and before != expected_hash:
        raise CalibrationError(f"PNG hash mismatch: {path}")
    with path.open("rb") as f:
        header = f.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise CalibrationError(f"Expected original PNG input: {path}")
    w, h, depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    if depth != 8:
        raise CalibrationError(f"PNG bit depth {depth} needs an explicit precision policy; RGB8 decoding is not silently substituted: {path}")
    with Image.open(path) as image:
        # Preserve text including oiio:ColorSpace/Postshot/version as well as
        # gAMA/sRGB/cHRM. Binary chunks are described by hash and byte length;
        # every original chunk remains intact in the unchanged source PNG.
        metadata = {}
        for key, value in image.info.items():
            if isinstance(value, bytes):
                metadata[key] = {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
            elif isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
                metadata[key] = value
        profile = image.info.get("icc_profile")
        if profile:
            metadata["icc_profile_sha256"] = hashlib.sha256(profile).hexdigest()
            metadata["icc_profile_bytes"] = len(profile)
            try:
                metadata["icc_profile_name"] = ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip()
            except Exception:
                metadata["icc_profile_name"] = "unrecognized; original profile retained in original PNG"
        original_mode = image.mode
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float64) / 255.0
    alpha = rgba[:, :, 3]
    # PNG stores unassociated/straight alpha. This intentionally composites
    # encoded sRGB numeric values onto black, matching the current gsplat engine's
    # encoded-RGB black-background training/render convention (no gamma fitting).
    rgb = rgba[:, :, :3] * alpha[:, :, None]
    return rgb, {"path": str(path), "sha256": before, "width": w, "height": h,
                 "mode": original_mode, "bit_depth": depth, "png_color_type": color_type,
                 "metadata": metadata,
                 "alpha": {"minimum": float(alpha.min()), "maximum": float(alpha.max()),
                           "minimum_8bit": int(round(float(alpha.min()) * 255)), "maximum_8bit": int(round(float(alpha.max()) * 255)),
                           "transparent_pixels": int((alpha == 0).sum()),
                           "nonopaque_pixels": int((alpha < 1).sum()),
                           "composite": "straight alpha over black in encoded sRGB numeric space"}}


def warp_intrinsics(image: np.ndarray, source_k: np.ndarray, output_k: np.ndarray,
                    width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form pinhole ray mapping, with no fitted image registration.

    Raster pixel centers are (column+.5,row+.5). scipy array indices are at
    integer centers, hence the explicit -.5 after homogeneous projection.
    """
    h, w = image.shape[:2]
    homography = source_k @ np.linalg.inv(output_k)
    if (w, h) == (width, height) and np.allclose(source_k, output_k, atol=1e-10, rtol=0):
        return image.copy(), np.ones((height, width), dtype=bool), homography
    y, x = np.mgrid[:height, :width].astype(np.float64)
    rays = np.stack([x + .5, y + .5, np.ones_like(x)], axis=-1) @ homography.T
    sx, sy = rays[:, :, 0] / rays[:, :, 2] - .5, rays[:, :, 1] / rays[:, :, 2] - .5
    for coords, limit in ((sx, w - 1), (sy, h - 1)):
        coords[np.abs(coords) < 1e-7] = 0
        coords[np.abs(coords - limit) < 1e-7] = limit
    valid = np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sy >= 0) & (sx <= w - 1) & (sy <= h - 1)
    warped = np.stack([map_coordinates(image[:, :, c], [sy, sx], order=1,
                                       mode="constant", cval=0, prefilter=False) for c in range(3)], axis=-1)
    return warped, valid, homography


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = lambda values: uniform_filter(values, size=(11, 11, 1), mode="constant", cval=0)
    ma, mb = mean(a), mean(b)
    va, vb, covariance = mean(a * a) - ma * ma, mean(b * b) - mb * mb, mean(a * b) - ma * mb
    return (((2 * ma * mb + .01 ** 2) * (2 * covariance + .03 ** 2)) /
            ((ma * ma + mb * mb + .01 ** 2) * (va + vb + .03 ** 2))).mean(axis=2)


def metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        raise CalibrationError("The shared evaluation mask has no valid pixels")
    difference = prediction[mask] - truth[mask]
    mse = float(np.mean(difference ** 2))
    # Keep original engine zero-padding behavior at physical image borders;
    # exclude centers whose in-frame 11x11 SSIM support touches invalid/watermark pixels.
    centers = minimum_filter(mask.astype(np.uint8), size=11, mode="constant", cval=1).astype(bool)
    if not centers.any():
        raise CalibrationError("The shared mask has no valid 11x11 SSIM windows")
    return {"psnr": float(-10 * math.log10(max(mse, 1e-12))), "ssim": float(ssim_map(prediction, truth)[centers].mean()),
            "mse": mse, "rgb_mean_abs": float(np.abs(difference).mean()), "pixels": int(mask.sum()),
            "pixel_fraction": float(mask.mean()), "ssim_window_centers": int(centers.sum())}


def rectangle_pixels(rectangle: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Round-trip integer mask bounds exactly despite normalized float error."""
    values = np.array(rectangle) * np.array([width, height, width, height])
    nearest = np.rint(values)
    values = np.where(np.abs(values - nearest) < 1e-7, nearest, values)
    return math.floor(values[0]), math.floor(values[1]), math.ceil(values[2]), math.ceil(values[3])


def evidence_entries(value: Any) -> list[dict]:
    if isinstance(value, list):
        return [item for entry in value for item in evidence_entries(entry)]
    if isinstance(value, dict):
        if "path" in value:
            return [value]
        return [item for entry in value.values() for item in evidence_entries(entry)]
    return []


def load_independent(specs: list[str], project_path: Path | None, manifest: dict) -> dict:
    result = {}
    project = read_json(project_path) if project_path else None
    if project and project.get("training", {}).get("dataset_key") != manifest["dataset_key"]:
        raise CalibrationError("Independent project dataset key differs from original input manifest")
    for spec in specs:
        if "=" not in spec:
            raise CalibrationError("Use --independent label=evaluation.json")
        label, filename = spec.split("=", 1)
        if not label or label == "reference" or label in result:
            raise CalibrationError("Independent method labels must be unique and not 'reference'")
        file = Path(filename).resolve(strict=True)
        data = read_json(file)
        rows = data.get("views", data.get("frames", data.get("evaluation", {}).get("views")))
        if not isinstance(rows, list):
            raise CalibrationError(f"No views/frames in independent evaluation manifest: {file}")
        indexed = {}
        for row in rows:
            index = row.get("reference_index", row.get("index"))
            if not isinstance(index, int) or index in indexed:
                raise CalibrationError(f"Invalid/duplicate independent reference index: {file}")
            image = resolve(row.get("png_path", row.get("render", row.get("file", ""))), file.parent)
            if "camera" in row:
                if not row.get("camera_verified"):
                    raise CalibrationError(f"Direct-render camera metadata lacks verification: {image}")
                camera = row["camera"]
            elif project:
                camera = project["reconstruction"]["cameras"][index]
                if row.get("name") not in (None, camera["name"], camera.get("source_name")):
                    raise CalibrationError(f"Independent frame name/index mismatch: {image}")
            else:
                raise CalibrationError("Existing independent PNGs require --independent-project or verified per-frame camera metadata")
            indexed[index] = {"path": image, "camera": camera, "sha256": row.get("sha256")}
        result[label] = {"manifest": str(file), "manifest_sha256": digest(file), "frames": indexed}
    return result


def evaluate(args: argparse.Namespace) -> dict:
    manifest_path, calibration_path = args.input_manifest.resolve(strict=True), args.calibration.resolve(strict=True)
    manifest, calibration = read_json(manifest_path), read_json(calibration_path)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "uncalibrated",
              "input_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
              "calibration": {"path": str(calibration_path), "sha256": digest(calibration_path)},
              "quality_metrics": None, "reasons": [], "split": args.split,
              "metric_definition": {"input": "decoded original RGB8 PNG values; encoded sRGB, no tone/exposure/image fitting",
                                    "psnr": "data_range=1; MSE floor1e-12/PSNR ceiling120dB; per-view mean and pooled MSE reported",
                                    "ssim": "uniform11x11 stride1 zero-padded local statistics; C1=.01^2 C2=.03^2; CPU float64 equivalent of existing engine formula",
                                    "masked_ssim": "Window centers whose complete in-frame11x11 support avoids excluded/invalid pixels",
                                    "sampling": "pixel centers +0.5; calibrated Ksource@inverse(Koutput); bilinear; no fitted registration"},
              "watermark_policy": {"original_files_preserved": True, "repair_or_removal": False,
                                   "excluded_normalized_rectangles": args.exclude_rect,
                                   "reason": args.mask_rationale},
              "initialization_note": "Both methods share full128-view sparse geometry and fused point colors. Holdout isolates RGB loss sampling, not all holdout-derived information."}
    try:
        if calibration.get("status") != "validated":
            raise CalibrationError("Camera contract status is not validated")
        reference_manifest = calibration.get("reference_manifest", {})
        if reference_manifest.get("sha256") and reference_manifest["sha256"] != report["input_manifest"]["sha256"]:
            raise CalibrationError("Camera contract was validated against a different input manifest")
        mask_contract_path = getattr(args, "mask_contract", None)
        if mask_contract_path:
            mask_contract_path = mask_contract_path.resolve(strict=True)
            mask_contract = read_json(mask_contract_path)
            if mask_contract.get("verified") is not True:
                raise CalibrationError("Public watermark mask contract is not verified")
            fixed_rect = rectangle(",".join(map(str, mask_contract["proposed_fixed_common_exclude_rect_normalized"])))
            mask_rows = {resolve(row["path"], mask_contract_path.parent): row for row in mask_contract["frames"]}
            for frame in calibration.get("frames", []):
                path = resolve(frame["png_path"], calibration_path.parent)
                if path not in mask_rows or mask_rows[path].get("sha256") != frame["sha256"]:
                    raise CalibrationError(f"Public watermark contract does not cover the exact original frame: {path}")
            if fixed_rect not in args.exclude_rect:
                args.exclude_rect.append(fixed_rect)
            report["watermark_policy"]["mask_contract"] = {"path": str(mask_contract_path), "sha256": digest(mask_contract_path), "method": mask_contract.get("method")}
        if calibration.get("color_space") != "sRGB" or not calibration.get("color_space_verified"):
            raise CalibrationError("Original PNG sRGB output has not been explicitly verified")
        evidence = evidence_entries(calibration.get("camera_evidence", calibration.get("evidence")))
        if not evidence:
            raise CalibrationError("Camera contract lacks immutable camera/GUI evidence")
        checked_evidence = []
        for entry in evidence:
            path = resolve(entry["path"], calibration_path.parent)
            if not entry.get("sha256") or digest(path) != entry["sha256"]:
                raise CalibrationError(f"Camera evidence hash missing or changed: {path}")
            checked_evidence.append({"path": str(path), "sha256": entry["sha256"]})
        report["camera_evidence"] = checked_evidence
        report["render_settings"] = calibration.get("render_settings", {key: calibration[key] for key in ("tone_mapping", "exposure_EV", "lens_shift", "alpha_policy") if key in calibration})
        report["calibration_intended_use"] = calibration.get("intended_use", calibration.get("split"))
        target = manifest["sets"][args.split]
        references = {row["index"]: row for row in target["images"]}
        frame_rows = calibration.get("frames")
        if not isinstance(frame_rows, list) or not frame_rows:
            raise CalibrationError("Camera contract has no explicit frame mapping")
        indices = [row.get("reference_index") for row in frame_rows]
        if len(set(indices)) != len(indices) or any(i not in references for i in indices):
            raise CalibrationError("Frame map contains repeated or non-selected reference indices")
        if not args.allow_subset and set(indices) != set(references):
            raise CalibrationError("Camera map is a subset; explicitly use --allow-subset and report coverage")
        independent = load_independent(args.independent, args.independent_project, manifest)
        report["independent_manifests"] = [{"name": label, "path": item["manifest"], "sha256": item["manifest_sha256"]} for label, item in independent.items()]
        rows, snapshots = [], {}
        render_plan = []
        for frame in frame_rows:
            index = frame["reference_index"]
            reference = references[index]
            for field in ("pose_verified", "K_verified", "resolution_verified", "frame_mapping_verified"):
                if frame.get(field) is not True:
                    raise CalibrationError(f"Frame {frame.get('frame')}, index {index}: {field} is not verified")
            if frame.get("name") not in (reference["name"], reference.get("source_name")):
                raise CalibrationError(f"Original frame {index}: explicit name/index mapping differs")
            errors = pose_errors(frame["observed_w2c_rub_world"], reference["w2c_rub_world"])
            if errors["camera_center_distance"] > args.position_tolerance or errors["rotation_error_degrees"] > args.rotation_tolerance_degrees:
                raise CalibrationError(f"Original frame {index}: pose mismatch {errors}")
            k_out = intrinsic(frame["observed_K"], "Observed K")
            k_gt = intrinsic(reference["K"], "Ground truth K")
            output_w, output_h = int(frame["observed_width"]), int(frame["observed_height"])
            if output_w <= 0 or output_h <= 0:
                raise CalibrationError(f"Invalid observed output dimensions for {index}")
            intrinsic_proof = verify_physical_intrinsics(frame, k_out, output_w, output_h)
            original_path = resolve(frame["png_path"], calibration_path.parent)
            if not frame.get("sha256"):
                raise CalibrationError(f"Original frame {index} has no immutable PNG hash")
            predicted, original_meta = read_png(original_path, frame["sha256"])
            if (original_meta["width"], original_meta["height"]) != (output_w, output_h):
                raise CalibrationError(f"Original PNG dimensions disagree with verified camera {index}")
            gt_path = resolve(reference["name"], resolve(target["path"], manifest_path.parent))
            truth_native, gt_meta = read_png(gt_path, reference["sha256"])
            if (gt_meta["width"], gt_meta["height"]) != (reference["width"], reference["height"]):
                raise CalibrationError(f"Ground truth header differs from immutable manifest for {index}")
            if args.gt_mapping == "identity" and ((output_w, output_h) != (reference["width"], reference["height"]) or not np.allclose(k_out, k_gt, atol=.001, rtol=0)):
                raise CalibrationError(f"Frame {index}: K/resolution differs; use explicitly verified --gt-mapping intrinsics, not arbitrary resizing")
            truth, valid, homography = warp_intrinsics(truth_native, k_gt, k_out, output_w, output_h)
            images, image_meta = {"reference": predicted}, {"reference": original_meta}
            for label, item in independent.items():
                if index not in item["frames"]:
                    raise CalibrationError(f"Independent method {label} is missing reference index {index}")
                entry = item["frames"][index]
                camera = entry["camera"]
                source_k = intrinsic(camera["K"], "Independent K")
                source_pose = camera.get("w2c_rub_world", camera.get("w2c"))
                delta = pose_errors(source_pose, reference["w2c_rub_world"])
                if delta["camera_center_distance"] > args.position_tolerance or delta["rotation_error_degrees"] > args.rotation_tolerance_degrees:
                    raise CalibrationError(f"Independent method {label}, frame {index}: camera pose differs")
                values, meta = read_png(entry["path"], entry["sha256"])
                if (meta["width"], meta["height"]) != (int(camera["width"]), int(camera["height"])):
                    raise CalibrationError(f"Independent method {label}, frame {index}: PNG/header camera mismatch")
                values, support, transform = warp_intrinsics(values, source_k, k_out, output_w, output_h)
                images[label], image_meta[label] = values, {**meta, "output_to_source_homography": transform.tolist(),
                                                          "resampled_existing_render": not ((meta["width"], meta["height"]) == (output_w, output_h) and np.allclose(source_k, k_out, atol=1e-10, rtol=0))}
                valid &= support
            regions = {"all_common_valid_pixels": valid.copy()}
            if args.exclude_rect:
                public = valid.copy()
                for fixed_rectangle in args.exclude_rect:
                    x0, y0, x1, y1 = rectangle_pixels(fixed_rectangle, output_w, output_h)
                    public[y0:y1, x0:x1] = False
                regions["fixed_public_region_excluded"] = public
            scores = {region: {label: metrics(values, truth, mask) for label, values in images.items()} for region, mask in regions.items()}
            for meta in [gt_meta, *image_meta.values()]:
                snapshots[meta["path"]] = meta["sha256"]
            rows.append({"frame": frame["frame"], "reference_index": index, "name": reference["name"],
                         "output_width": output_w, "output_height": output_h, "observed_K": k_out.tolist(),
                         "pose_errors": errors, "physical_intrinsics_proof": intrinsic_proof,
                         "output_to_gt_homography": homography.tolist(),
                         "ground_truth": gt_meta, "images": image_meta, "scores": scores})
            render_plan.append({"reference_index": index, "name": reference["name"], "original_frame": frame["frame"],
                                "camera": {"width": output_w, "height": output_h, "K": k_out.tolist(), "w2c": frame["observed_w2c_rub_world"]},
                                "camera_verified": True})
        for path, sha in snapshots.items():
            if digest(Path(path)) != sha:
                raise CalibrationError(f"Input changed during evaluation: {path}")
        summary = {}
        for region in rows[0]["scores"]:
            summary[region] = {}
            for method in rows[0]["scores"][region]:
                scores = [row["scores"][region][method] for row in rows]
                pixels = sum(score["pixels"] for score in scores)
                pooled_mse = sum(score["mse"] * score["pixels"] for score in scores) / pixels
                summary[region][method] = {"views": len(scores), "psnr_mean": float(np.mean([s["psnr"] for s in scores])),
                                          "ssim_mean": float(np.mean([s["ssim"] for s in scores])),
                                          "pooled_psnr": float(-10 * math.log10(max(pooled_mse, 1e-12))), "pixels_total": pixels}
        report.update({"status": "calibrated", "quality_metrics": summary, "frames": rows,
                       "coverage": {"evaluated": len(rows), "expected": len(references), "subset": len(rows) != len(references)},
                       "camera_gate": {"position_tolerance_world_units": args.position_tolerance,
                                       "rotation_tolerance_degrees": args.rotation_tolerance_degrees,
                                       "intrinsics_policy": args.gt_mapping},
                       "comparison_limit": "Per-frame images[].resampled_existing_render records whether an independent PNG required intrinsic reprojection. Matching-K manifests are rendered at the observed K; ground truth uses the declared known intrinsic warp. No fitted registration or appearance adjustment."})
        if args.render_plan:
            write_json(args.render_plan, {"status": "camera_verified", "calibration_sha256": digest(calibration_path),
                                         "input_manifest_sha256": digest(manifest_path), "frames": render_plan,
                                         "gpu_execution_authorized_by_this_file": False})
    except (CalibrationError, OSError, KeyError, TypeError, ValueError) as exc:
        report["reasons"].append(str(exc))
        report["status"] = "uncalibrated"
        report["quality_metrics"] = None
    return report


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def rectangle(spec: str) -> list[float]:
    values = list(map(float, spec.split(",")))
    if len(values) != 4 or not all(map(math.isfinite, values)) or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise argparse.ArgumentTypeError("Use normalized x0,y0,x1,y1 within0..1")
    return values


def check_output_paths(args: argparse.Namespace) -> None:
    """Refuse overwriting any supplied input or evidence, including JSON inputs."""
    manifest_path, calibration_path = args.input_manifest.resolve(), args.calibration.resolve()
    manifest, calibration = read_json(manifest_path), read_json(calibration_path)
    protected = {manifest_path, calibration_path}
    if args.independent_project:
        protected.add(args.independent_project.resolve())
    if getattr(args, "mask_contract", None):
        protected.add(args.mask_contract.resolve())
    for group in manifest.get("sets", {}).values():
        parent = resolve(group["path"], manifest_path.parent)
        protected.update(resolve(row["name"], parent) for row in group.get("images", []))
    protected.update(resolve(entry["path"], calibration_path.parent) for entry in evidence_entries(calibration.get("camera_evidence", calibration.get("evidence"))))
    protected.update(resolve(frame["png_path"], calibration_path.parent) for frame in calibration.get("frames", []) if frame.get("png_path"))
    for spec in args.independent:
        if "=" in spec:
            path = Path(spec.split("=", 1)[1]).resolve()
            protected.add(path)
            data = read_json(path)
            for row in data.get("views", data.get("frames", data.get("evaluation", {}).get("views", []))):
                name = row.get("png_path", row.get("render", row.get("file")))
                if name:
                    protected.add(resolve(name, path.parent))
    outputs = [path.resolve() for path in (args.output, args.render_plan) if path]
    if len(set(outputs)) != len(outputs):
        raise CalibrationError("Report and render-plan output paths must differ")
    for path in outputs:
        if path.suffix.lower() != ".json" or path in protected or path.with_name(path.name + ".tmp") in protected:
            raise CalibrationError(f"Output must be a separate JSON artifact, preserving all inputs/evidence: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("holdout", "train"), default="holdout")
    parser.add_argument("--gt-mapping", choices=("identity", "intrinsics"), default="identity")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--independent", action="append", default=[], metavar="LABEL=EVALUATION_JSON")
    parser.add_argument("--independent-project", type=Path)
    parser.add_argument("--exclude-rect", type=rectangle, action="append", default=[])
    parser.add_argument("--mask-contract", type=Path, help="Verified original-PNG-only watermark location evidence and fixed shared rectangle")
    parser.add_argument("--mask-rationale", default="Explicit fixed common public watermark region; full common-valid metrics are also reported")
    parser.add_argument("--position-tolerance", type=float, default=1e-5)
    parser.add_argument("--rotation-tolerance-degrees", type=float, default=.001)
    parser.add_argument("--render-plan", type=Path, help="Prepare measured cameras only; never executes GPU rendering")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.position_tolerance, args.rotation_tolerance_degrees)):
        parser.error("Pose tolerances must be positive")
    try:
        check_output_paths(args)
    except (CalibrationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Refused output: {exc}", file=sys.stderr)
        return 2
    report = evaluate(args)
    write_json(args.output.resolve(), report)
    print(json.dumps({"status": report["status"], "report": str(args.output.resolve()),
                      "coverage": report.get("coverage"), "quality_metrics": report["quality_metrics"],
                      "reasons": report["reasons"]}, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "calibrated" else 2


if __name__ == "__main__":
    raise SystemExit(main())

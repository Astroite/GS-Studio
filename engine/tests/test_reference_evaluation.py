"""CPU-only calibration gates and agreement with the existing metric definition."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

_source = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_reference_pngs.py"
_spec = importlib.util.spec_from_file_location("reference_png_evaluation", _source)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def fixture(tmp_path):
    pixels = np.random.default_rng(14).integers(0, 256, (31, 37, 3), dtype=np.uint8)
    gt = tmp_path / "image.png"
    predicted = tmp_path / "original.png"
    Image.fromarray(pixels).save(gt)
    Image.fromarray(pixels).save(predicted)
    evidence = tmp_path / "camera-evidence.txt"
    evidence.write_text("independently measured camera fixture", encoding="utf-8")
    camera = {"K": [[37, 0, 18.5], [0, 31, 15.5], [0, 0, 1]],
              "w2c_rub_world": np.eye(4).tolist(), "width": 37, "height": 31}
    manifest = {"dataset_key": "fixture", "sets": {"holdout": {"path": str(tmp_path), "images": [
        {**camera, "index": 8, "name": gt.name, "sha256": evaluation.digest(gt)}]}}}
    contract = {"status": "validated", "color_space": "sRGB", "color_space_verified": True,
                "camera_evidence": [{"path": str(evidence), "sha256": evaluation.digest(evidence)}],
                "frames": [{"frame": 0, "reference_index": 8, "name": gt.name,
                            "png_path": str(predicted), "sha256": evaluation.digest(predicted),
                            "observed_K": camera["K"], "observed_w2c_rub_world": camera["w2c_rub_world"],
                            "observed_width": 37, "observed_height": 31,
                            "pose_verified": True, "K_verified": True, "resolution_verified": True,
                            "frame_mapping_verified": True, "camera_properties": {
                                "lens_mm": 36, "sensor_width_mm": 36, "sensor_height_mm": 36,
                                "shift_x": 0, "shift_y": 0}, "gui_fov_degrees": [53.1301, 53.1301]}]}
    manifest_path, contract_path = tmp_path / "inputs.json", tmp_path / "contract.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    args = argparse.Namespace(input_manifest=manifest_path, calibration=contract_path, split="holdout",
        output=tmp_path / "result.json", render_plan=tmp_path / "render-plan.json", independent=[],
        independent_project=None, exclude_rect=[], mask_rationale="fixture", gt_mapping="identity",
        allow_subset=False, position_tolerance=1e-5, rotation_tolerance_degrees=.001)
    return args, contract


def test_pixel_center_intrinsics_warp_has_known_ray_mapping():
    y, x = np.mgrid[:20, :24]
    image = np.stack((x / 24, y / 20, (x + y) / 44), axis=-1)
    k = np.array([[20, 0, 12], [0, 20, 10], [0, 0, 1]], dtype=float)
    same, valid, h = evaluation.warp_intrinsics(image, k, k, 24, 20)
    np.testing.assert_array_equal(same, image)
    assert valid.all()
    shifted = k.copy()
    shifted[0, 2] -= 1
    warped, valid, h = evaluation.warp_intrinsics(image, k, shifted, 24, 20)
    np.testing.assert_allclose(warped[:, :-1], image[:, 1:], atol=1e-15)
    assert valid[:, :-1].all() and not valid[:, -1].any()
    np.testing.assert_allclose(h, [[1, 0, 1], [0, 1, 0], [0, 0, 1]])


def test_png_preserves_metadata_and_composites_straight_alpha(tmp_path):
    path = tmp_path / "original.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("oiio:ColorSpace", "sRGB")
    info.add_text("Postshot/version", "fixture")
    Image.fromarray(np.array([[[200, 100, 50, 128], [77, 90, 10, 0]]], dtype=np.uint8)).save(path, pnginfo=info)
    before = path.read_bytes()
    rgb, metadata = evaluation.read_png(path, evaluation.digest(path))
    np.testing.assert_allclose(rgb[0, 0], np.array([200, 100, 50]) * 128 / 255 ** 2)
    np.testing.assert_array_equal(rgb[0, 1], 0)
    assert metadata["metadata"]["Postshot/version"] == "fixture"
    assert metadata["alpha"]["maximum_8bit"] == 128
    assert path.read_bytes() == before


def test_uniform11_ssim_agrees_with_existing_engine_on_cpu():
    import torch
    from local_postshot.training import ssim
    rng = np.random.default_rng(34)
    a, b = rng.random((29, 37, 3)), rng.random((29, 37, 3))
    legacy = ssim(torch.tensor(a, dtype=torch.float64, device="cpu")[None],
                  torch.tensor(b, dtype=torch.float64, device="cpu")[None])
    assert evaluation.ssim_map(a, b).mean() == pytest.approx(float(legacy), abs=1e-12)


def test_mask_does_not_leak_watermark_into_ssim_windows():
    truth = np.full((40, 40, 3), .4)
    prediction = truth.copy()
    prediction[12:18, 12:18] = 1
    mask = np.ones((40, 40), dtype=bool)
    mask[12:18, 12:18] = False
    assert evaluation.metrics(prediction, truth, mask)["ssim"] == pytest.approx(1, abs=1e-12)
    assert evaluation.metrics(prediction, truth, np.ones_like(mask))["ssim"] < .99


def test_normalized_watermark_rectangle_recovers_exact_integer_bounds():
    rect = [312 / 1612, 108 / 1204, 412 / 1612, 208 / 1204]
    assert evaluation.rectangle_pixels(rect, 1612, 1204) == (312, 108, 412, 208)


def test_float32_rotation_roundoff_is_not_false_angle():
    pose = np.eye(4)
    pose[0, 0] -= 1e-7
    errors = evaluation.pose_errors(pose, np.eye(4))
    assert errors["rotation_error_degrees"] == pytest.approx(0, abs=1e-8)
    assert errors["SO3_angle_only_max_corrections"][0] > 0
    pose[0, 0] = .999
    with pytest.raises(evaluation.CalibrationError, match="orthonormal"):
        evaluation.pose_errors(pose, np.eye(4))


def test_full_calibrated_evaluation_and_measured_render_plan(tmp_path):
    args, contract = fixture(tmp_path)
    report = evaluation.evaluate(args)
    assert report["status"] == "calibrated", report["reasons"]
    score = report["quality_metrics"]["all_common_valid_pixels"]["reference"]
    assert score["psnr_mean"] == 120 and score["ssim_mean"] == pytest.approx(1)
    plan = json.loads(args.render_plan.read_text(encoding="utf-8"))
    assert plan["frames"][0]["reference_index"] == 8
    assert plan["frames"][0]["camera"]["K"] == contract["frames"][0]["observed_K"]
    assert plan["gpu_execution_authorized_by_this_file"] is False


@pytest.mark.parametrize("invalid", ["pose", "K", "physical", "mapping", "hash", "color"])
def test_calibration_gate_rejects_unverified_or_wrong_inputs(tmp_path, invalid):
    args, contract = fixture(tmp_path)
    frame = contract["frames"][0]
    if invalid == "pose":
        frame["observed_w2c_rub_world"][0][3] += .001
    elif invalid == "K":
        frame["observed_K"][0][0] += 2
    elif invalid == "physical":
        frame["camera_properties"]["lens_mm"] += 2
    elif invalid == "mapping":
        frame["frame_mapping_verified"] = False
    elif invalid == "hash":
        frame["sha256"] = "0" * 64
    else:
        contract["color_space_verified"] = False
    args.calibration.write_text(json.dumps(contract), encoding="utf-8")
    report = evaluation.evaluate(args)
    assert report["status"] == "uncalibrated" and report["quality_metrics"] is None
    assert report["reasons"]
    assert not args.render_plan.exists()


def test_report_cannot_overwrite_camera_evidence(tmp_path):
    args, _ = fixture(tmp_path)
    args.output = args.input_manifest
    before = args.input_manifest.read_bytes()
    with pytest.raises(evaluation.CalibrationError, match="preserving"):
        evaluation.check_output_paths(args)
    assert args.input_manifest.read_bytes() == before


def test_direct_matching_camera_png_is_not_warped_twice(tmp_path):
    args, contract = fixture(tmp_path)
    frame = contract["frames"][0]
    manifest_path = tmp_path / "local-renders.json"
    manifest_path.write_text(json.dumps({"already_at_observed_Kout": True, "frames": [{
        "reference_index": frame["reference_index"], "png_path": frame["png_path"], "sha256": frame["sha256"],
        "camera_verified": True, "camera": {"width": frame["observed_width"], "height": frame["observed_height"],
            "K": frame["observed_K"], "w2c": frame["observed_w2c_rub_world"]}}]}), encoding="utf-8")
    args.independent = [f"local={manifest_path}"]
    result = evaluation.evaluate(args)
    assert result["status"] == "calibrated", result["reasons"]
    assert result["frames"][0]["images"]["local"]["resampled_existing_render"] is False
    assert result["quality_metrics"]["all_common_valid_pixels"]["local"]["psnr_mean"] == 120


def test_watermark_contract_must_cover_the_actual_original_png(tmp_path):
    args, _ = fixture(tmp_path)
    args.mask_contract = tmp_path / "watermark.json"
    args.mask_contract.write_text(json.dumps({"verified": True,
        "proposed_fixed_common_exclude_rect_normalized": [.1, .1, .2, .2], "frames": []}), encoding="utf-8")
    report = evaluation.evaluate(args)
    assert report["status"] == "uncalibrated" and report["quality_metrics"] is None
    assert "exact original frame" in report["reasons"][0]


def test_render_helper_dry_run_rejects_changed_camera_without_gpu(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path)
    assert evaluation.evaluate(args)["status"] == "calibrated"
    monkeypatch.syspath_prepend(str(_source.parent))
    path = _source.with_name("render_reference_cameras.py")
    spec = importlib.util.spec_from_file_location("render_reference_cameras", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    model = tmp_path / "independent.ply"
    model.write_bytes(b"CPU validation hashes the model; it does not allocate or render it")
    helper_args = argparse.Namespace(plan=args.render_plan, calibration=args.calibration,
        input_manifest=args.input_manifest, model=[f"raw={model}"], output_dir=tmp_path / "new-output")
    plan, models, provenance = helper.validate(helper_args)
    assert len(plan["frames"]) == 1 and models["raw"]["sha256"] == evaluation.digest(model)
    assert not helper_args.output_dir.exists()
    plan["frames"][0]["camera"]["K"][0][0] += 1
    args.render_plan.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(helper.CalibrationError, match="measured K"):
        helper.validate(helper_args)

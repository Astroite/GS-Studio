"""Shared write operations for the CLI and the GS-Studio workbench.

The methods here return domain objects.  They never infer success from an
artifact's existence, and callers must not turn an exception into success.
"""
from __future__ import annotations

import json
import shutil
import stat
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from gsstudio.infrastructure.persistence.manifests import load_capture_manifest, load_model, save_yaml
from gsstudio.pipeline.masks.finalize import MaskFinalizationMissingError, finalize_mask_dataset, validate_mask_finalization
from gsstudio.pipeline.masks.masking import validate_mask_filter
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.domain.models import (
    IMAGE_SUFFIXES, Camera, CaptureManifest, CaptureSource, LocationManifest,
    NormalizationSettings, PreparedInputConfig, Region, Rights, RunConfig,
    RunStatus, SceneManifest, SourceFile, SourceProbe, TimeSelection, utc_now,
)
from gsstudio.infrastructure.paths import host_path, location_dir, scene_dir
from gsstudio.pipeline.editor.ply import FLOAT_PROPERTIES, read_ply_header
from gsstudio.pipeline.stages import (
    ingest_capture, mask_run, preprocess_run, reconstruct_run, review_run,
    write_qa_report,
)
from gsstudio.pipeline.retention import auto_cleanup, cleanup_run
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.infrastructure.persistence.run_repository import create_run, load_run, save_run
from gsstudio.pipeline.input.sources import load_prepared_input, prepare_capture_input, probe_capture_source
from gsstudio.pipeline.training.runner import train_package
from gsstudio.pipeline.training.data import prepare_segment


from gsstudio.application.contracts import EventSink, _emit
from gsstudio.application._shared import _scene, _selected_dataset, _verify_model
from gsstudio.application.captures import verify_capture_sources


def create_and_preprocess(
    root: Path, location_id: str, scene_id: str, capture_id: str,
    *, candidate_fps: float = 1.0, selected_per_second: int = 1,
    primary_per_second: int = 1, projection_fov: float = 110.0,
    projection_size: int = 1746, mask_review_gate: bool = True,
    keep_intermediates: bool = False, sink: EventSink | None = None,
    mask_device: str = "cuda", mask_score_threshold: float = 0.25,
    mask_probability_threshold: float = 0.50, mask_gamma: float = 0.75,
    mask_dilation_pixels: int = 24, mask_discard_threshold: float = 0.005,
    mask_qa_sample_count: int = 16, mask_classes: str = "person",
    loop_closure: bool = False, vocabulary_tree: Path | None = None,
    loop_period: int = 20, loop_num_images: int = 5,
    vision_qa: bool = False, smoke: bool = False,
    reconstruction_backend: str = "realityscan",
) -> str:
    """Create one current-schema Run, then execute its preprocess stage."""
    if reconstruction_backend not in {'realityscan','colmap'}:
        raise ValueError('reconstruction_backend must be realityscan or colmap')
    scene = _scene(root, location_id, scene_id)
    capture = load_capture_manifest(scene / "captures" / f"{capture_id}.yaml")
    verify_capture_sources(capture)
    if smoke:
        if candidate_fps != 1.0 or selected_per_second != 1 or primary_per_second != 1:
            raise ValueError("Smoke profile fixes temporal density at 1/1/1")
        projection_size = min(projection_size, 1024)
        mask_qa_sample_count = min(mask_qa_sample_count, 8)
    effective_end = min(capture.selection.end_seconds, capture.selection.start_seconds + 5.0) if smoke else capture.selection.end_seconds
    loop_path = None
    loop_sha256 = None
    if loop_closure:
        if vocabulary_tree is None:
            raise ValueError("Loop closure requires a local vocabulary tree")
        tree = host_path(vocabulary_tree).absolute()
        if not tree.is_file():
            raise FileNotFoundError(tree)
        loop_path, loop_sha256 = str(tree), sha256_file(tree)
    run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    work = scene / run_id
    if work.exists():
        raise FileExistsError(work)
    work.mkdir(parents=True)
    _emit(sink, "started", "preprocess", "正在准备输入", run_id)
    try:
        candidate = prepare_capture_input(
            scene, capture, work / "inputs" / "primary", candidate_fps,
            selection_end_seconds=effective_end, resume=False,
        )
    except Exception:
        # This directory was created by this invocation and has no authoritative
        # manifest yet. Check the resolved boundary before removing partial input.
        boundary = work.resolve()
        if boundary.parent != scene.resolve():
            raise RuntimeError(f"Unsafe Run staging path; inspect manually: {work}")
        if not (work / "manifest.yaml").exists():
            for item in work.rglob("*"):
                attributes = getattr(item.lstat(), "st_file_attributes", 0)
                if (item.is_symlink() or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT or
                        not item.resolve().is_relative_to(boundary)):
                    raise RuntimeError(f"Unsafe Run staging content; inspect manually: {item}")
            shutil.rmtree(work)
        raise
    perspective = capture.source.is_perspective
    config = RunConfig(
        capture_id=capture_id, input_dataset_sha256=candidate.dataset_sha256,
        input_relative_path=f"inputs/primary/{candidate.path.name}",
        retention={"mode": "keep" if keep_intermediates else "minimal"},
        input=PreparedInputConfig(
            source_kind=capture.source.kind,
            source_sha256=[item.sha256 for item in capture.source.files],
            source_probe=candidate.source_probe,
            normalization=capture.normalization,
            selection=TimeSelection(start_seconds=capture.selection.start_seconds,
                                    end_seconds=effective_end),
            candidate_frame_indices=list(candidate.frame_indices),
            candidate_fps=candidate_fps,
            helper_version=candidate.helper_version,
            sdk_version=candidate.sdk_version,
        ),
        preprocess={"candidate_fps": candidate_fps, "selected_per_second": selected_per_second},
        masking={
            "device": mask_device, "score_threshold": mask_score_threshold,
            "probability_threshold": mask_probability_threshold,
            "inference_gamma": mask_gamma, "dilation_pixels": mask_dilation_pixels,
            "mask_discard_threshold": mask_discard_threshold,
            "qa_sample_count": mask_qa_sample_count,
            "classes": [item.strip() for item in mask_classes.split(",") if item.strip()],
            "mask_review_required": mask_review_gate,
        },
        vision_qa={"enabled": vision_qa},
        reconstruction={
            "backend": reconstruction_backend,
            # Only projected panorama views have an analytically known focal
            # length. A photo/video Capture's HFOV is an initialization prior;
            # both RealityScan and COLMAP must be allowed to estimate calibration.
            "fix_intrinsics": not perspective,
            "primary": {
                "temporal_rank_limit": primary_per_second,
                "projection_fov_degrees": capture.horizontal_fov_degrees if perspective else projection_fov,
                "panorama": None if perspective else {
                    "views": 14, "size": projection_size, "crop_bottom": 0.15,
                },
            },
            "loop_closure": {
                "enabled": loop_closure, "period": loop_period,
                "num_images": loop_num_images, "vocabulary_tree_path": loop_path,
                "vocabulary_tree_sha256": loop_sha256,
            },
        },
    )
    run = create_run(scene, location_id, scene_id, config, run_id=run_id)
    protocol = work / "inputs" / "primary" / ".protocol"
    if protocol.exists():
        (work / "logs").mkdir(exist_ok=True)
        protocol.rename(work / "logs" / "prepare-protocol")
    preprocess_run(scene, run)
    _emit(sink, "completed", "preprocess", "输入准备完成", run_id)
    return run_id


def resume_preprocess(root: Path, location_id: str, scene_id: str,
                      run_id: str, sink: EventSink | None = None,
                      resume: bool = True) -> None:
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    capture = load_capture_manifest(scene / "captures" / f"{run.config.capture_id}.yaml")
    verify_capture_sources(capture)
    _emit(sink, "started", "preprocess", "正在核验并恢复输入", run_id)
    preprocess_run(scene, run, resume=resume)
    _emit(sink, "completed", "preprocess", "输入已核验", run_id)


def audit_run(root: Path, location_id: str, scene_id: str,
              run_id: str) -> dict[str, str]:
    """Verify the evidence needed by the next stage before any automatic resume."""
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    work = scene / run_id
    stages = {name: item.status.value for name, item in run.stages.items()}
    if stages["preprocess"] == "succeeded" and stages["mask"] != "succeeded":
        prepared = load_prepared_input(work / run.config.input_relative_path)
        if prepared.dataset_sha256 != run.config.input_dataset_sha256:
            raise RuntimeError("Prepared input identity differs from the Run config")
    if stages["preprocess"] == "succeeded" and stages["mask"] != "succeeded":
        records = work / "selected-primary-metrics.jsonl"
        if not records.is_file() or not (work / "frames-primary").is_dir():
            raise RuntimeError("Successful preprocess lost the inputs required by mask; create a new Run")
    if stages["mask"] == "succeeded" and stages["reconstruct"] != "succeeded":
        dataset = work / "reconstruction-primary"
        validate_mask_filter(dataset, verify_hashes=True)
        if run.config.masking.mask_review_required:
            try:
                validate_mask_finalization(dataset)
            except MaskFinalizationMissingError:
                pass  # The next action is the primary manual review gate.
    if stages["reconstruct"] == "succeeded":
        if not run.selected_dataset:
            raise RuntimeError("Successful reconstruction has no selected dataset")
        dataset = _selected_dataset(scene, run)
        from gsstudio.pipeline.quality.segments import validate_segments

        filtered = validate_mask_filter(dataset, verify_hashes=True)
        included = {item["image"] for item in filtered["accepted"]}
        if included:
            final = validate_mask_finalization(dataset, included)
            included -= set(final["excluded_images"])
        label = run.metrics["selected_attempt"]
        records = [json.loads(line) for line in
                   (work / f"selected-{label}-metrics.jsonl").read_text(encoding="utf-8").splitlines()
                   if line]
        report = validate_segments(dataset, records, included,
                                   run.config.reconstruction.primary,
                                   run.config.segment_qa)
        if report.get("training_status") != "passed":
            raise RuntimeError("Selected reconstruction no longer has passed training segments")
    if stages["qa"] == "succeeded" and not (scene / "qa" / f"{run_id}.md").is_file():
        raise RuntimeError("Successful QA lost its report")
    return stages


def run_stage(root: Path, location_id: str, scene_id: str, run_id: str,
              stage: str, *, resume: bool = False,
              sink: EventSink | None = None) -> Any:
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    if stage in {"mask", "reconstruct"}:
        capture = load_capture_manifest(scene / "captures" / f"{run.config.capture_id}.yaml")
        verify_capture_sources(capture)
    _emit(sink, "started", stage, f"开始 {stage}", run_id)
    try:
        if stage == "mask":
            result = mask_run(scene, run, resume=resume)
        elif stage == "reconstruct":
            result = reconstruct_run(scene, run, resume=resume)
        elif stage == "qa":
            result = write_qa_report(scene, run, resume=resume)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    except Exception as error:
        _emit(sink, "waiting" if isinstance(error, MaskFinalizationMissingError) else "failed",
              stage, str(error), run_id)
        raise
    _emit(sink, "completed", stage, f"{stage} 完成", run_id)
    return result


def cleanup(root: Path, location_id: str, scene_id: str,
            run_id: str, *, apply: bool = False) -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    with run_session(scene / run_id):
        return cleanup_run(scene, load_run(scene, run_id), apply=apply)

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


def _source_files(source_type: str, sources: list[Path]) -> list[Path]:
    resolved = [host_path(item).absolute() for item in sources]
    if source_type in {"equirect_sequence", "perspective_images"} and len(resolved) == 1 and resolved[0].is_dir():
        resolved = sorted(
            item for item in resolved[0].iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    if not resolved or any(not item.is_file() for item in resolved):
        raise FileNotFoundError("Every source must resolve to an existing file")
    return resolved


def create_capture(
    root: Path, location_id: str, scene_id: str, capture_id: str,
    source_type: str, sources: list[Path], *, camera_model: str = "unknown",
    camera_make: str = "Insta360", fps: float | None = None,
    start_seconds: float = 0.0, end_seconds: float | None = None,
    output_width: int | None = None, output_height: int | None = None,
    horizontal_fov: float = 84.0,
    normalization: dict | None = None,
) -> CaptureManifest:
    scene = _scene(root, location_id, scene_id)
    path = scene / "captures" / f"{capture_id}.yaml"
    if path.exists():
        raise FileExistsError(path)
    if source_type not in {"equirect_video", "equirect_sequence", "insta360_insv", "perspective_video", "perspective_images"}:
        raise ValueError(f"Unsupported source type: {source_type}")
    files = _source_files(source_type, sources)
    records = [
        SourceFile(windows_path=str(item), sha256=sha256_file(item), byte_size=item.stat().st_size)
        for item in files
    ]
    perspective = source_type in {"perspective_video", "perspective_images"}
    if source_type == "perspective_images" and fps is None:
        fps = 1.0
    capture = CaptureManifest(
        id=capture_id, location_id=location_id, scene_id=scene_id,
        source=CaptureSource(
            kind=source_type, files=records,
            projection=("dual_fisheye" if source_type == "insta360_insv" else
                        "perspective" if perspective else "equirectangular"),
            probe=SourceProbe(fps=fps),
        ),
        camera=Camera(make=camera_make, model=camera_model),
        selection=TimeSelection(start_seconds=0.0, end_seconds=1.0),
        normalization=NormalizationSettings(**({"width": output_width, "height": output_height,
                                              **(normalization or {})})),
        horizontal_fov_degrees=horizontal_fov if perspective else None,
    )
    probe = probe_capture_source(capture, scene / "logs" / "probe-protocol")
    capture.source.probe = SourceProbe(**{
        key: value for key, value in probe.items() if key in SourceProbe.model_fields
    })
    stop = float(probe["duration_seconds"] if end_seconds is None else end_seconds)
    capture.selection = TimeSelection(start_seconds=start_seconds, end_seconds=stop)
    if stop > float(probe["duration_seconds"]) + 0.5:
        raise ValueError("Capture selection extends beyond the source duration")
    if not perspective and capture.normalization.width is None:
        capture.normalization.width = int(probe["width"])
        capture.normalization.height = int(probe["height"])
    capture = CaptureManifest.model_validate(capture)
    save_yaml(path, capture)
    return capture


def verify_capture_sources(capture: CaptureManifest) -> None:
    """Check source identity before a new Run or a resumed source-dependent step."""
    for source in capture.source.files:
        path = host_path(source.windows_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != source.byte_size or sha256_file(path) != source.sha256:
            raise RuntimeError(f"Capture source identity changed: {path}")


def relink_capture_sources(root: Path, location_id: str, scene_id: str,
                           capture_id: str, sources: list[Path]) -> CaptureManifest:
    """Replace only source paths, after checking the original ordered content identity."""
    scene = _scene(root, location_id, scene_id)
    manifest_path = scene / "captures" / f"{capture_id}.yaml"
    capture = load_capture_manifest(manifest_path)
    files = _source_files(capture.source.kind, sources)
    if len(files) != len(capture.source.files):
        raise ValueError("Relocated source count differs from the capture manifest")
    for old, new in zip(capture.source.files, files):
        if new.stat().st_size != old.byte_size or sha256_file(new) != old.sha256:
            raise RuntimeError(f"Relocated source does not match its recorded hash: {new}")
    updated = capture.model_copy(deep=True)
    updated.source.files = [
        old.model_copy(update={"windows_path": str(new)})
        for old, new in zip(capture.source.files, files)
    ]
    updated = CaptureManifest.model_validate(updated)
    save_yaml(manifest_path, updated)
    return updated


def probe_capture(root: Path, location_id: str, scene_id: str,
                  capture_id: str) -> dict[str, Any]:
    scene = _scene(root, location_id, scene_id)
    capture = load_capture_manifest(scene / "captures" / f"{capture_id}.yaml")
    verify_capture_sources(capture)
    return probe_capture_source(capture, scene / "logs" / "probe-protocol")


def ingest(root: Path, location_id: str, scene_id: str, capture_id: str) -> CaptureManifest:
    scene = _scene(root, location_id, scene_id)
    capture = load_capture_manifest(scene / "captures" / f"{capture_id}.yaml")
    verify_capture_sources(capture)
    return ingest_capture(scene, capture_id)

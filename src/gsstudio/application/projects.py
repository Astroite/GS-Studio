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


def create_location(root: Path, location_id: str, name: str,
                    capture_date: str | None = None, province: str | None = None,
                    city: str | None = None) -> LocationManifest:
    path = location_dir(root, location_id) / "location.yaml"
    if path.exists():
        raise FileExistsError(path)
    manifest = LocationManifest(
        id=location_id, display_name=name,
        capture_date=date.fromisoformat(capture_date) if capture_date else None,
        region=Region(province=province, city=city), rights=Rights(),
    )
    save_yaml(path, manifest)
    return manifest


def create_scene(root: Path, location_id: str, scene_id: str, name: str,
                 description: str | None = None) -> SceneManifest:
    load_model(location_dir(root, location_id) / "location.yaml", LocationManifest)
    path = scene_dir(root, location_id, scene_id)
    manifest_path = path / "scene.yaml"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest = SceneManifest(
        id=scene_id, location_id=location_id, display_name=name,
        description=description,
    )
    save_yaml(manifest_path, manifest)
    for relative in ("captures", "stitched", "exports", "qa"):
        (path / relative).mkdir(parents=True, exist_ok=True)
    return manifest

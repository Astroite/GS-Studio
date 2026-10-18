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


def _scene(root: Path, location_id: str, scene_id: str) -> Path:
    path = scene_dir(root, location_id, scene_id)
    load_model(path / "scene.yaml", SceneManifest)
    return path


def _selected_dataset(scene: Path, run) -> Path:
    label = run.metrics.get("selected_attempt")
    if label not in {"primary", "repair"}:
        raise RuntimeError("Selected reconstruction attempt is missing or invalid")
    expected = f"{run.id}/reconstruction-{label}"
    if run.selected_dataset != expected:
        raise RuntimeError("Selected dataset identity differs from the Run manifest")
    return scene / expected


def _verify_model(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"Published PLY hash changed: {path}")
    header, count, properties = read_ply_header(path)
    if (count < 1 or not set(FLOAT_PROPERTIES) <= set(properties) or
            path.stat().st_size != len(header) + count * len(properties) * 4):
        raise RuntimeError(f"Published PLY layout is invalid: {path}")

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


def finalize_masks(root: Path, location_id: str, scene_id: str, run_id: str,
                   attempt: str = "primary", *, expected_identity: str | None = None) -> dict[str, Any]:
    if attempt not in {"primary", "repair"}:
        raise ValueError("Attempt must be primary or repair")
    scene = _scene(root, location_id, scene_id)
    with run_session(scene / run_id):
        run = load_run(scene, run_id)
        if expected_identity is not None:
            from gsstudio.application.studio import _mask_dataset
            _,identity,_=_mask_dataset({'root':str(root),'location_id':location_id,'scene_id':scene_id,
                                       'run_id':run_id,'attempt':attempt})
            if identity!=expected_identity:
                raise RuntimeError('Mask review identity changed; reload before finalization')
        dataset = scene / run_id / f"reconstruction-{attempt}"
        if not (dataset / "images").is_dir() or not (dataset / "masks").is_dir():
            raise FileNotFoundError(dataset)
        filtered = validate_mask_filter(dataset, verify_hashes=False)
        if run.config.masking.mask_review_required:
            from gsstudio.pipeline.masks.review import MaskReviewDataset
            review = MaskReviewDataset(dataset)
            if any(review.review_for(item.image_id)['status'] not in {'ok','exclude'}
                   for item in review.items):
                raise RuntimeError('Every mask requires explicit ok/exclude review before finalization')
        validate_mask_filter(dataset, verify_hashes=True)
        result = finalize_mask_dataset(
            dataset, {str(item["image"]) for item in filtered["accepted"]},
            maximum_included_masked_fraction=run.config.masking.mask_discard_threshold,
        )
        auto_cleanup(scene, run)
        return result


def review_qa(root: Path, location_id: str, scene_id: str, run_id: str,
              accepted: bool, notes: str) -> Any:
    if type(accepted) is not bool:
        raise ValueError('QA accepted must be an explicit boolean')
    if not notes.strip():
        raise ValueError("Review notes are required")
    scene = _scene(root, location_id, scene_id)
    with run_session(scene / run_id):
        from gsstudio.application.runs import audit_run
        from gsstudio.application.captures import verify_capture_sources
        run = load_run(scene, run_id)
        capture = load_capture_manifest(scene/'captures'/f'{run.config.capture_id}.yaml')
        verify_capture_sources(capture)
        audit_run(root,location_id,scene_id,run_id)
        return review_run(scene, load_run(scene, run_id), accepted, notes)

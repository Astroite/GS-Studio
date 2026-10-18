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


def verified_models(root: Path, location_id: str, scene_id: str,
                    run_id: str) -> list[dict[str, str]]:
    """List only completed experiments whose published PLY still matches its hash."""
    scene = _scene(root, location_id, scene_id)
    run = load_run(scene, run_id)
    result = []
    for item in run.metrics.get("training_experiments", []):
        if item.get("status") != "succeeded" or not item.get("model_sha256") or item.get('backend')!='gsplat':
            continue
        output = Path(str(item["output"]))
        model = output / "model.ply"
        try:
            dispatch = json.loads((output / "dispatch.json").read_text(encoding="utf-8"))
            if (dispatch.get("status") != "succeeded" or
                    dispatch.get("model_sha256") != item["model_sha256"] or
                    dispatch.get("backend") != item["backend"]):
                continue
            _verify_model(model, item["model_sha256"])
            from gsstudio.pipeline.training.runner import verify_completed
            from gsstudio.application.studio import _package_for_dispatch
            _,meta=_package_for_dispatch(dispatch)
            verify_completed(output,meta,dispatch)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
        result.append({"segment": str(item["segment"]),
                       "backend": str(item["backend"]),
                       "path": str(model),
                       "sha256": str(item["model_sha256"]),
                       'output':str(output),'package_sha256':dispatch['package_sha256']})
    return result

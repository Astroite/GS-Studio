"""Run stages: preprocess -> mask -> reconstruct (primary + one bounded repair) -> qa.

There is one run shape and one implementation per stage. Prepared candidates live
inside the Run (`inputs/primary/<hash>`), identity is pinned by hashes, and resume
never invents a second repair round or resurrects rejected pixels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gsstudio.infrastructure.runtime.gpu_lock import gpu_locked
from gsstudio.infrastructure.persistence.manifests import canonical_hash, load_capture_manifest, save_yaml
from gsstudio.pipeline.masks.finalize import (
    MaskFinalizationMissingError,
    finalize_mask_dataset,
    validate_mask_finalization,
)
from gsstudio.pipeline.masks.review import MaskReviewDataset
from gsstudio.pipeline.masks.masking import (
    _recover_mask_filter,
    _recover_mask_records,
    create_mask_contact_sheets,
    filter_masked_images,
    generate_person_masks,
    image_files,
    summarize_detection_records,
    validate_mask_filter,
    validate_mask_set,
)
from gsstudio.infrastructure.adapters.media import (
    analyze_frames,
    create_temporal_subset,
    sha256_file,
    summarize_frame_metrics,
)
from gsstudio.domain.models import CaptureManifest, RunManifest, RunStatus
from gsstudio.infrastructure.paths import ensure_run_dir
from gsstudio.infrastructure.runtime.processes import CommandError
from gsstudio.pipeline.retention import auto_cleanup, safe_path
from gsstudio.pipeline.reconstruction.core import project_equirectangular_frames
from gsstudio.pipeline.reconstruction.realityscan import run_realityscan_alignment
from gsstudio.domain.run_state import begin_stage, complete_stage, fail_stage
from gsstudio.infrastructure.persistence.run_repository import save_run
from gsstudio.infrastructure.runtime.run_lock import run_locked
from gsstudio.pipeline.quality.segments import analyze_segments, frame_id, write_qa_report, write_segments
from gsstudio.pipeline.input.sources import (
    _validate_frame,
    load_prepared_input,
    probe_capture_source,
    source_adapter,
    validate_source_fingerprints,
)
from gsstudio.infrastructure.persistence.storage import link_or_copy
from gsstudio.pipeline.training.data import json_write
from gsstudio.pipeline.quality.vision import load_local_mask_qa_review, run_deepseek_mask_qa




def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _record_resources(
    run: RunManifest,
    work: Path,
    command_metrics: dict[str, float] | None = None,
) -> None:
    resources = run.metrics.setdefault("resources", {})
    observed = _tree_size(work)
    resources["disk_observed_peak_bytes"] = max(
        int(resources.get("disk_observed_peak_bytes", 0)), observed
    )
    if command_metrics and "gpu_memory_peak_mib" in command_metrics:
        resources["gpu_memory_observed_peak_mib"] = max(
            float(resources.get("gpu_memory_observed_peak_mib", 0)),
            command_metrics["gpu_memory_peak_mib"],
        )


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_records(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    temporary.replace(path)


def capture_path(scene_path: Path, capture_id: str) -> Path:
    return scene_path / "captures" / f"{capture_id}.yaml"


def load_capture(scene_path: Path, capture_id: str) -> CaptureManifest:
    return load_capture_manifest(capture_path(scene_path, capture_id))


def ingest_capture(scene_path: Path, capture_id: str) -> CaptureManifest:
    """Re-probe an immutable, explicitly listed source set and refresh its manifest."""
    from gsstudio.pipeline.input.sources import probe_capture_source

    capture = load_capture(scene_path, capture_id)
    probe = probe_capture_source(capture, scene_path / "logs" / "probe-protocol")
    capture.source.probe = type(capture.source.probe)(
        **{
            name: value
            for name, value in probe.items()
            if name in type(capture.source.probe).model_fields
        }
    )
    save_yaml(capture_path(scene_path, capture_id), capture)
    return capture


def _attempt_inventory(dataset: Path, run: RunManifest) -> tuple[dict[str, Any], set[str]]:
    filtered = validate_mask_filter(dataset, verify_hashes=False)
    accepted = {str(item["image"]) for item in filtered["accepted"]}
    finalized = validate_mask_finalization(dataset, accepted)
    included = accepted - set(finalized.get("excluded_images", []))
    return finalized, included

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




def review_run(scene_path: Path, run: RunManifest, accepted: bool, notes: str) -> RunManifest:
    if run.status != RunStatus.NEEDS_REVIEW:
        raise RuntimeError(
            f"Only needs_review runs can be reviewed; current status is {run.status.value}"
        )
    run.status = RunStatus.ACCEPTED if accepted else RunStatus.REJECTED
    run.review_notes = notes
    save_run(scene_path, run)
    return run

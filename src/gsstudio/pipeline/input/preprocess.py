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


from gsstudio.pipeline.stage_common import write_records


@run_locked
def preprocess_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "preprocess", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    save_run(scene_path, run)
    work = ensure_run_dir(scene_path, run.id)
    try:
        target = safe_path(work, run.config.input_relative_path)
        prepared = load_prepared_input(target)
        if prepared.dataset_sha256 != run.config.input_dataset_sha256:
            raise RuntimeError("Prepared input changed")
        duration = (
            run.config.input.selection.end_seconds - run.config.input.selection.start_seconds
        )
        records = analyze_frames(
            list(prepared.frame_paths),
            duration,
            work / "frame-metrics.jsonl",
            timestamps_seconds=prepared.timestamps_seconds,
        )
        records = [
            {**record, "source_frame_index": index, "source_sha256": digest}
            for record, index, digest in zip(
                records, prepared.frame_indices, prepared.frame_sha256s
            )
        ]
        write_records(work / "frame-metrics.jsonl", records)
        _, selected = create_temporal_subset(
            prepared.path / "frames",
            records,
            work / "frames-primary",
            run.config.input.selection.start_seconds,
            run.config.preprocess.selected_per_second,
            rank_limit=run.config.reconstruction.primary.temporal_rank_limit,
        )
        for record in selected:
            if sha256_file(work / "frames-primary" / record["file"]) != record["source_sha256"]:
                raise RuntimeError("Selected frame changed; refusing unverified resume input")
        write_records(work / "selected-primary-metrics.jsonl", selected)
        run.metrics["preprocess"] = dict(
            candidate_frame_count=len(records),
            selected_frame_count=len(selected),
            candidate_fps=run.config.preprocess.candidate_fps,
            selected_per_second=run.config.preprocess.selected_per_second,
            candidate=summarize_frame_metrics(records),
            selected=summarize_frame_metrics(selected),
        )
        run.metrics["input"] = prepared.source_probe.model_dump(mode="json")
        run.metrics["selection"] = run.config.input.selection.model_dump(mode="json")
        complete_stage(
            run,
            "preprocess",
            message=f"Prepared {len(records)} Run-owned candidates; selected {len(selected)}",
        )
    except Exception as error:
        fail_stage(run, "preprocess", str(error))
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    auto_cleanup(scene_path, run)
    return run

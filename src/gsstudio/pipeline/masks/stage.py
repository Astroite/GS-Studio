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


from gsstudio.pipeline.stage_common import _record_resources


def _cached_masking_result(
    scene_path: Path,
    dataset: Path,
    run: RunManifest,
    label: str,
) -> dict[str, Any] | None:
    cached = run.metrics.get("masking", {}).get(label)
    if not isinstance(cached, dict):
        return None
    try:
        filtered = validate_mask_filter(
            dataset, verify_hashes=not run.config.masking.mask_review_required
        )
    except (OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError):
        return None
    expected_count = len(filtered["accepted"])
    if int(cached.get("reconstruction_input_images", -1)) != expected_count:
        return None
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    images = image_files(images_dir)
    image_names = {path.relative_to(images_dir).as_posix() for path in images}
    if len(image_names) != expected_count:
        return None
    expected_masks = {
        (Path(name).parent / f"{Path(name).name}.png").as_posix() for name in image_names
    }
    actual_masks = {
        path.relative_to(masks_dir).as_posix() for path in image_files(masks_dir)
    }
    if actual_masks != expected_masks:
        return None
    sheets = [scene_path / str(path) for path in cached.get("contact_sheets", [])]
    sheets_required = run.config.vision_qa.enabled or run.config.masking.mask_review_required
    if sheets_required and (not sheets or not all(path.is_file() for path in sheets)):
        return None
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if not local_review_path.is_file():
            return None
        _, verdict = load_local_mask_qa_review(local_review_path, sheets, run.config.vision_qa)
        if verdict.decision != "pass":
            return None
    return dict(cached)


def _stage_views(source: Path, dataset: Path, attempt) -> list[Path]:
    """Training views under ``dataset/images``: projected views, or the frames themselves.

    A perspective source is already one frame per view, so nothing is projected and
    the prepared frames are linked in under their own names.
    """
    if attempt.panorama is not None:
        return project_equirectangular_frames(source, dataset, attempt)
    target = dataset / "images"
    target.mkdir(parents=True, exist_ok=True)
    frames = image_files(source)
    existing = image_files(target)
    if len(existing) != len(frames):
        for frame in frames:
            link_or_copy(frame, target / frame.name)
        existing = image_files(target)
        if len(existing) != len(frames):
            raise RuntimeError(
                f"Perspective frame staging is incomplete ({len(existing)}/{len(frames)})"
            )
    return existing


def _prepare_masked_dataset(
    scene_path: Path,
    work: Path,
    source: Path,
    dataset: Path,
    attempt,
    run: RunManifest,
    label: str,
) -> dict[str, Any]:
    mask_filter = _recover_mask_filter(
        dataset,
        run.config.masking.mask_discard_threshold,
        review_required=run.config.masking.mask_review_required,
    )
    recovered = mask_filter is not None
    cached = _cached_masking_result(scene_path, dataset, run, label) if not recovered else None
    if cached is not None:
        print(f"Masked dataset: {label} (validated cached result)", flush=True)
        return cached
    if recovered:
        records = _recover_mask_records(
            dataset, mask_filter, work / f"mask-metrics-{label}.jsonl"
        )
        projected_count = mask_filter["original_image_count"]
    else:
        images = _stage_views(source, dataset, attempt)
        projected_count = len(images)
        records = generate_person_masks(
            dataset / "images",
            dataset / "masks",
            run.config.masking,
            work / f"mask-metrics-{label}.jsonl",
        )
    if run.config.retention.mode == "keep" and not recovered:
        for item in records:
            if float(item["masked_fraction"]) > run.config.masking.mask_discard_threshold:
                for folder, key in (("images", "image"), ("masks", "mask")):
                    link_or_copy(
                        dataset / folder / str(item[key]),
                        work / "rejected-inputs" / label / folder / str(item[key]),
                    )
    mask_filter = (
        mask_filter
        if recovered
        else filter_masked_images(dataset, records, run.config.masking.mask_discard_threshold)
    )
    accepted_names = {str(item["image"]) for item in mask_filter["accepted"]}
    accepted_records = [item for item in records if str(item["image"]) in accepted_names]
    deterministic = validate_mask_set(
        dataset / "images",
        dataset / "masks",
        # Review edits can cross the automatic threshold. Finalization still
        # requires explicit exclusion; recovery must not re-filter them.
        1.0
        if recovered and run.config.masking.mask_review_required
        else run.config.masking.mask_discard_threshold,
    )
    if accepted_names and not run.config.masking.mask_review_required:
        finalize_mask_dataset(
            dataset,
            expected_images=accepted_names,
            maximum_included_masked_fraction=run.config.masking.mask_discard_threshold,
        )
    make_sheets = bool(accepted_records) and (
        run.config.masking.mask_review_required or run.config.vision_qa.enabled
    )
    existing_sheets = sorted((dataset / "mask-qa").glob("mask-contact-*.jpg"))
    review_bound = (dataset / "mask-qa" / "codex-local-review.json").is_file()
    sheets = (
        existing_sheets
        if recovered and (existing_sheets or review_bound)
        else (
            create_mask_contact_sheets(
                dataset / "images",
                dataset / "masks",
                accepted_records,
                dataset / "mask-qa",
                run.config.masking.qa_sample_count,
                run.config.vision_qa.max_contact_sheets,
            )
            if make_sheets
            else []
        )
    )
    vision: dict[str, Any]
    if run.config.vision_qa.enabled:
        local_review_path = dataset / "mask-qa" / "codex-local-review.json"
        if local_review_path.is_file():
            local_review, verdict = load_local_mask_qa_review(
                local_review_path, sheets, run.config.vision_qa
            )
            vision = {
                "provider": local_review.reviewer,
                "review_file": local_review_path.relative_to(scene_path).as_posix(),
                "contact_sheet_sha256": local_review.contact_sheet_sha256,
                **verdict.model_dump(mode="json"),
            }
        else:
            verdict = run_deepseek_mask_qa(sheets, run.config.vision_qa)
            vision = {"provider": "deepseek", **verdict.model_dump(mode="json")}
        if verdict.decision != "pass":
            raise RuntimeError(
                f"LLM mask QA rejected {label}: {verdict.rationale}; "
                f"false negatives={verdict.false_negative_views}"
            )
    else:
        vision = {
            "status": "disabled",
            "reason": (
                "Set vision_qa.enabled=true in a new run only after confirming an authorized "
                "DeepSeek multimodal endpoint"
            ),
        }
    result = {
        "planar_images": projected_count,
        "model": run.config.masking.model if run.config.masking.enabled else "disabled-all-white",
        "detection": summarize_detection_records(accepted_records),
        "validation": deterministic,
        "contact_sheets": [path.relative_to(scene_path).as_posix() for path in sheets],
        "vision_qa": vision,
    }
    surviving_frame_ids = sorted(
        {
            int(match.group(1))
            for item in mask_filter["accepted"]
            if (match := re.search(r"frame_(\d+)\.jpg$", str(item["image"])))
        }
    )
    result.update(
        {
            "projected_planar_images": projected_count,
            "automatic_rejected_images": len(mask_filter["rejected"]),
            "automatic_rejected_fraction": (
                len(mask_filter["rejected"]) / projected_count if projected_count else 0.0
            ),
            "reconstruction_input_images": len(mask_filter["accepted"]),
            "surviving_frame_ids": surviving_frame_ids,
            "mask_filter": (dataset / "mask-filter.json").relative_to(scene_path).as_posix(),
        }
    )
    return result


@gpu_locked
@run_locked
def mask_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "mask", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    save_run(scene_path, run)
    work = ensure_run_dir(scene_path, run.id)
    log_path = work / "logs" / "mask.log"
    try:
        primary = _prepare_masked_dataset(
            scene_path,
            work,
            work / "frames-primary",
            work / "reconstruction-primary",
            run.config.reconstruction.primary,
            run,
            "primary",
        )
        run.metrics.setdefault("masking", {})["primary"] = primary
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps(primary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _record_resources(run, work)
        complete_stage(
            run,
            "mask",
            message=(
                f"Projected and masked {primary['planar_images']} primary perspective views; "
                f"deterministic QA passed"
            ),
            log_path=log_path.relative_to(scene_path).as_posix(),
        )
        if run.config.masking.mask_review_required and primary.get("reconstruction_input_images", 0):
            run.status = RunStatus.WAITING_REVIEW
            run.stages["mask"].message = (
                f"{run.stages['mask'].message}; awaiting primary mask review/finalization"
            )
    except Exception as error:
        fail_stage(run, "mask", str(error), log_path.relative_to(scene_path).as_posix())
        save_run(scene_path, run)
        raise
    save_run(scene_path, run)
    auto_cleanup(scene_path, run)
    return run

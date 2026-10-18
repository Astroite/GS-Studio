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
    image_dimensions,
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


from gsstudio.pipeline.stage_common import _attempt_inventory, load_capture, read_records, write_records
from gsstudio.pipeline.masks.stage import _prepare_masked_dataset


def repair_plan(run: RunManifest, selected, candidates, report) -> dict:
    from gsstudio.pipeline.input.sources import anchored_frame_indices

    selection = run.config.input.selection
    padding = run.config.reconstruction.repair_padding_seconds
    times = {i: float(r["timestamp_seconds"]) for i, r in enumerate(selected, 1)}
    windows = []
    for f in set(report.get("weak_input_frames", [])) | set(report.get("missing_frames", [])):
        windows.append((times[f] - padding, times[f] + padding))
    for issue in report.get("issues", []):
        a = times[issue.get("from_frame", issue.get("frame"))]
        b = times[issue.get("to_frame", issue.get("frame"))]
        windows.append((a - padding, b + padding))
    # Coverage gaps can be absent from the pose-issue list by design.
    ordered = sorted(times.values())
    for a, b in zip(ordered, ordered[1:]):
        if b - a > run.config.segment_qa.max_gap_seconds:
            windows.append((a - padding, b + padding))
    merged = []
    for a, b in sorted(
        (max(selection.start_seconds, a), min(selection.end_seconds, b)) for a, b in windows
    ):
        if a >= b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(b, merged[-1][1])
        else:
            merged.append([a, b])
    probe = run.config.input.source_probe
    repair_rate=min(run.config.reconstruction.repair_candidate_fps,float(probe.fps))
    indices, timestamps = anchored_frame_indices(
        int(probe.frame_count),
        float(probe.fps),
        selection.start_seconds,
        selection.end_seconds,
        repair_rate,
    )
    seen = {int(r["source_frame_index"]) for r in candidates}
    extras = [
        dict(source_frame_index=i, timestamp_seconds=t)
        for i, t in zip(indices, timestamps)
        if i not in seen and any(a <= t <= b for a, b in merged)
    ]
    return dict(
        schema_version=1,
        attempt=1,
        config_hash=run.config_hash,
        primary_records_sha256=canonical_hash(selected),
        candidate_records_sha256=canonical_hash(candidates),
        source_sha256=run.config.input.source_sha256,
        windows=merged,
        candidate_fps=repair_rate,
        frames=extras,
        helper_version=run.config.input.helper_version,
        sdk_version=run.config.input.sdk_version,
    )


def _export_repair(scene_path: Path, run: RunManifest, plan: dict) -> list[Path]:
    work = scene_path / run.id
    capture = load_capture(scene_path, run.config.capture_id)
    if (
        [r.sha256 for r in capture.source.files] != run.config.input.source_sha256
        or capture.normalization != run.config.input.normalization
    ):
        raise RuntimeError("Capture identity changed before repair")
    folder = work / "inputs" / "repair"
    (folder / "frames").mkdir(parents=True, exist_ok=True)
    probe = probe_capture_source(capture, work / "logs" / "repair-protocol")
    for key in ("helper_version", "sdk_version"):
        if probe.get(key) != getattr(run.config.input, key):
            raise RuntimeError("Repair SDK/helper changed; create a new Run")
    for key in ("fps", "frame_count", "width", "height"):
        if probe[key] != getattr(run.config.input.source_probe, key):
            raise RuntimeError(f"Repair source probe changed: {key}")
    indices = [r["source_frame_index"] for r in plan["frames"]]
    outputs = [
        folder / "frames" / f"frame_{i:06d}.jpg" for i in range(1, len(indices) + 1)
    ]
    normalization = capture.normalization
    width = int(normalization.width or probe["width"])
    height = int(normalization.height or probe["height"])
    manifest = folder / "dataset.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved["plan_sha256"] != canonical_hash(plan):
            raise RuntimeError("Repair extraction identity changed")
        for path, record in zip(outputs, saved["frames"]):
            if _validate_frame(path, record.get('width',width), record.get('height',height)) != record["sha256"]:
                raise RuntimeError("Repair extraction changed")
        return outputs
    source_adapter(capture).export(
        capture,
        validate_source_fingerprints(capture),
        probe,
        indices,
        outputs,
        folder,
        work / "logs" / "repair-protocol",
        width,
        height,
        True,
    )
    validate_source_fingerprints(capture)
    records = []
    for row,path in zip(plan['frames'],outputs):
        frame_height,frame_width=image_dimensions(path)
        records.append({**row,'file':path.name,'width':frame_width,'height':frame_height,
                        'sha256':_validate_frame(path,frame_width,frame_height)})
    json_write(
        manifest,
        dict(
            schema_version=1,
            plan_sha256=canonical_hash(plan),
            frames=records,
            helper_version=probe.get("helper_version"),
            sdk_version=probe.get("sdk_version"),
        ),
    )
    return outputs


def merge_repair_inputs(work: Path, run: RunManifest, selected, extra_records, plan) -> list[dict]:
    """Remap both accepted and rejected inventories; never resurrect rejected pixels."""
    dataset = work / "reconstruction-repair"
    all_records = [(r, "primary", i) for i, r in enumerate(selected, 1)] + [
        (r, "new", i) for i, r in enumerate(extra_records, 1)
    ]
    all_records.sort(key=lambda item: item[0]["timestamp_seconds"])
    mapping = {(kind, old): i for i, (_, kind, old) in enumerate(all_records, 1)}
    records = [
        {**r, "file": f"frame_{i:06d}.jpg", "origin": kind, "origin_frame": old}
        for i, (r, kind, old) in enumerate(all_records, 1)
    ]
    identity = dict(plan_sha256=canonical_hash(plan), records_sha256=canonical_hash(records), sources={})
    inventories = {}
    for label, source in (("primary", work / "reconstruction-primary"), ("new", work / "repair-new")):
        inventories[label] = validate_mask_filter(source, verify_hashes=True)
        identity["sources"][label] = sha256_file(source / "mask-filter.json")
    receipt = dataset / "repair-input.json"
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Repair input identity changed")
        validate_mask_filter(dataset, verify_hashes=True)
        return records
    dataset.mkdir(exist_ok=True)
    (dataset / "images").mkdir(exist_ok=True)
    (dataset / "masks").mkdir(exist_ok=True)
    merged = dict(
        schema_version=1,
        status="complete",
        threshold=run.config.masking.mask_discard_threshold,
        comparison="masked_fraction > threshold",
        original_image_count=0,
        accepted=[],
        rejected=[],
    )
    excluded: set[str] = set()
    storage = []
    for label, source in (("primary", work / "reconstruction-primary"), ("new", work / "repair-new")):
        filtered = inventories[label]
        original_excluded: set[str] = set()
        if (source / "mask-final.json").exists():
            final = validate_mask_finalization(source, {r["image"] for r in filtered["accepted"]})
            original_excluded = set(final["excluded_images"])

        def remap(name, label=label):
            index = mapping[label, frame_id(name)]
            return re.sub(r"frame_\d+", f"frame_{index:06d}", name)

        for kind in ("accepted", "rejected"):
            for row in filtered[kind]:
                changed = {**row, "image": remap(row["image"]), "mask": remap(row["mask"])}
                merged[kind].append(changed)
                if kind == "accepted":
                    for folder, key in (("images", "image"), ("masks", "mask")):
                        storage.append(
                            link_or_copy(source / folder / row[key], dataset / folder / changed[key])
                        )
                    if row["image"] in original_excluded:
                        excluded.add(changed["image"])
        merged["original_image_count"] += filtered["original_image_count"]
    for kind in ("accepted", "rejected"):
        merged[kind].sort(key=lambda r: r["image"])
    json_write(dataset / "mask-filter.json", merged)
    validate_mask_filter(dataset, verify_hashes=True)
    if excluded:
        review = MaskReviewDataset(dataset)
        for item in review.items:
            if item.source_image in excluded:
                review.update_review(item.image_id, "exclude", "Inherited from finalized source attempt")
    if merged["accepted"] and not run.config.masking.mask_review_required:
        finalize_mask_dataset(
            dataset, {r["image"] for r in merged["accepted"]}, run.config.masking.mask_discard_threshold
        )
    json_write(
        dataset / "storage.json",
        dict(files=storage, duplicate_bytes=sum(r["duplicate_bytes"] for r in storage)),
    )
    write_records(work / "selected-repair-metrics.jsonl", records)
    json_write(receipt, identity)
    return records


def record_repair_metrics(run: RunManifest, filtered: dict) -> None:
    total = filtered["original_image_count"]
    run.metrics.setdefault("masking", {})["repair"] = dict(
        projected_planar_images=total,
        planar_images=total,
        reconstruction_input_images=len(filtered["accepted"]),
        automatic_rejected_images=len(filtered["rejected"]),
        automatic_rejected_fraction=len(filtered["rejected"]) / total if total else 0.0,
    )


def prepare_repair(scene_path: Path, run: RunManifest, selected, report) -> list[dict] | None:
    work = scene_path / run.id
    candidates = read_records(work / "frame-metrics.jsonl")
    plan = repair_plan(run, selected, candidates, report)
    path = work / "repair-plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("Repair plan identity changed; create a new Run")
    else:
        json_write(path, plan)
    if not plan["frames"]:
        return None
    dataset = work / "reconstruction-repair"
    if (dataset / "repair-input.json").exists():
        receipt = json.loads((dataset / "repair-input.json").read_text(encoding="utf-8"))
        records = read_records(work / "selected-repair-metrics.jsonl")
        if (
            receipt["plan_sha256"] != canonical_hash(plan)
            or receipt["records_sha256"] != canonical_hash(records)
        ):
            raise RuntimeError("Committed repair dataset identity changed")
        record_repair_metrics(run, validate_mask_filter(dataset, verify_hashes=True))
        return records
    outputs = _export_repair(scene_path, run, plan)
    extras = analyze_frames(
        outputs,
        run.config.input.selection.end_seconds - run.config.input.selection.start_seconds,
        work / "repair-frame-metrics.jsonl",
        timestamps_seconds=[r["timestamp_seconds"] for r in plan["frames"]],
    )
    extras = [
        {**r, **identity, "source_file": r["file"], "source_sha256": sha256_file(p)}
        for r, identity, p in zip(extras, plan["frames"], outputs)
    ]
    metrics = _prepare_masked_dataset(
        scene_path,
        work,
        outputs[0].parent,
        work / "repair-new",
        run.config.reconstruction.primary,
        run,
        "repair-new",
    )
    run.metrics.setdefault("masking", {})["repair-new"] = metrics
    records = merge_repair_inputs(work, run, selected, extras, plan)
    write_records(work / "selected-repair-metrics.jsonl", records)
    record_repair_metrics(run, validate_mask_filter(dataset, verify_hashes=True))
    return records


@gpu_locked
@run_locked
def reconstruct_run(scene_path: Path, run: RunManifest, resume: bool = False) -> RunManifest:
    if not begin_stage(run, "reconstruct", resume=resume):
        auto_cleanup(scene_path, run)
        return run
    work = scene_path / run.id
    save_run(scene_path, run)
    try:
        selected = read_records(work / "selected-primary-metrics.jsonl")
        attempts = run.metrics.setdefault("reconstruction", {})
        viable = []
        report: dict[str, Any] | None = None
        for label, records in (("primary", selected), ("repair", None)):
            if label == "repair":
                if report is None:
                    break
                records = prepare_repair(scene_path, run, selected, report)
                if records is None:
                    break
                save_run(scene_path, run)
                auto_cleanup(scene_path, run, checkpoint=True)
            dataset = work / f"reconstruction-{label}"
            filtered = validate_mask_filter(dataset, verify_hashes=False)
            included = {r["image"] for r in filtered["accepted"]}
            if included:
                _, included = _attempt_inventory(dataset, run)
            if len(included) >= 2:
                try:
                    alignment = run_realityscan_alignment
                    if run.config.reconstruction.backend == 'colmap':
                        from gsstudio.pipeline.reconstruction.colmap import run_colmap_alignment
                        alignment = run_colmap_alignment
                    metrics = alignment(
                        dataset,
                        run.config.reconstruction.primary,
                        work / "logs" / f"reconstruct-{label}",
                        run.config.reconstruction,
                        included_images=included,
                        projected_image_count=filtered["original_image_count"],
                    )
                except CommandError as error:
                    if label != "repair" or not viable:
                        raise
                    attempts[label] = {
                        "status": "failed",
                        "error": str(error),
                        "retained_valid_primary": True,
                    }
                    break
                attempts[label] = metrics
                report = write_segments(
                    dataset,
                    records,
                    included,
                    run.config.reconstruction.primary,
                    run.config.segment_qa,
                )
            else:
                report = analyze_segments([], records, included, run.config.segment_qa, [])
                attempts[label] = {
                    "registered_images": 0,
                    "reason": "fewer than two included images",
                }
                report["lineage"] = {
                    "records_sha256": canonical_hash(records),
                    "included_sha256": canonical_hash(sorted(included)),
                }
                temporary = dataset / "segments.json.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                temporary.replace(dataset / "segments.json")
            run.metrics["segment_qa"] = {
                "attempt": label,
                "training_status": report["training_status"],
                "coverage_status": report["coverage_status"],
                "report": f"{run.id}/reconstruction-{label}/segments.json",
            }
            if report["training_status"] == "passed":
                # Keep independently aligned models separate. Prefer covered time,
                # then usable images; do not discard valid primary segments when the
                # single repair attempt yields a worse component.
                passed = [s for s in report["segments"] if s["status"] == "passed"]
                score = (
                    sum(s["end_seconds"] - s["start_seconds"] for s in passed),
                    sum(len(s["images"]) for s in passed),
                )
                viable.append((score, dataset, report, label))
            save_run(scene_path, run)
        if not viable:
            raise RuntimeError("No valid training segment after the bounded repair attempt")
        _, dataset, report, label = max(viable, key=lambda item: item[0])
        run.metrics["segment_qa"] = {
            "attempt": label,
            "training_status": report["training_status"],
            "coverage_status": report["coverage_status"],
            "report": f"{run.id}/reconstruction-{label}/segments.json",
        }
        run.selected_dataset = dataset.relative_to(scene_path).as_posix()
        run.metrics["selected_attempt"] = dataset.name.removeprefix("reconstruction-")
        complete_stage(
            run, "reconstruct", message=f"Valid segments available; route coverage {report['coverage_status']}"
        )
        save_run(scene_path, run)
        auto_cleanup(scene_path, run)
        return run
    except Exception as error:
        fail_stage(run, "reconstruct", str(error))
        if isinstance(error, MaskFinalizationMissingError):
            run.status = RunStatus.WAITING_REVIEW
        save_run(scene_path, run)
        raise

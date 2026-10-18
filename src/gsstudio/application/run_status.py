"""Stage, evidence and log views for one Run.

Status still comes from the run manifest: stage records decide success. The
evidence rows only say whether a manifest, report or result file exists, so a
present artifact can never be read as a finished stage (AC-20). Nothing here
writes to the data tree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gsstudio.domain.models import RunConfig
from gsstudio.application.browse import Problem, RunEntry, absolute_root, load_run_entry, status_label


@dataclass
class StageView:
    name: str
    status: str
    started_at: str | None
    completed_at: str | None
    elapsed_seconds: float | None
    message: str | None
    log_path: str | None
    log_present: bool


@dataclass
class LogView:
    name: str
    path: str
    size_bytes: int
    modified_at: float


@dataclass
class EvidenceView:
    """One file that documents work. ``present`` is never a success judgement."""

    name: str
    path: str
    present: bool


@dataclass
class ExperimentView:
    segment: str | None
    backend: str | None
    status: str | None
    output: str | None
    error: str | None
    dispatch_path: str | None
    dispatch_present: bool


@dataclass
class RunDetail:
    entry: RunEntry
    status_label: str
    selected_attempt: str | None
    review_notes: str | None
    config_summary: dict[str, Any]
    stages: list[StageView]
    evidence: list[EvidenceView]
    logs: list[LogView]
    experiments: list[ExperimentView]
    training_status: str | None = None
    coverage_status: str | None = None
    qa_report_text: str | None = None
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_summary(config: RunConfig) -> dict[str, Any]:
    primary = config.reconstruction.primary
    panorama = primary.panorama
    return {
        "capture_id": config.capture_id,
        "source_kind": config.input.source_kind,
        "source_files": len(config.input.source_sha256),
        "candidate_fps": config.preprocess.candidate_fps,
        "selected_per_second": config.preprocess.selected_per_second,
        "temporal_rank_limit": primary.temporal_rank_limit,
        "projection_fov_degrees": primary.projection_fov_degrees,
        "projection": (
            "perspective"
            if panorama is None
            else {"views": panorama.views, "size": panorama.size, "crop_bottom": panorama.crop_bottom}
        ),
        "mask_classes": list(config.masking.classes),
        "mask_discard_threshold": config.masking.mask_discard_threshold,
        "mask_review_required": config.masking.mask_review_required,
        "vision_qa": config.vision_qa.enabled,
        "retention": config.retention.mode,
        "input_relative_path": config.input_relative_path,
        "input_dataset_sha256": config.input_dataset_sha256,
    }


def read_run_detail(root: Path, location_id: str, scene_id: str, run_id: str) -> RunDetail:
    entry, run = load_run_entry(root, location_id, scene_id, run_id)
    if run is None:
        return RunDetail(
            entry=entry,
            status_label=status_label(entry),
            selected_attempt=None,
            review_notes=None,
            config_summary={},
            stages=[],
            evidence=[],
            logs=[],
            experiments=[],
            problems=list(entry.problems),
        )

    base = absolute_root(root) / location_id / scene_id
    work = base / run.id
    stages = _stage_views(run, base)
    evidence = _evidence_views(run, base, work)
    return RunDetail(
        entry=entry,
        status_label=status_label(entry),
        selected_attempt=run.metrics.get("selected_attempt"),
        review_notes=run.review_notes,
        config_summary=_config_summary(run.config),
        stages=stages,
        evidence=evidence,
        logs=_log_views(work / "logs"),
        experiments=_experiment_views(run, work),
        training_status=run.metrics.get("segment_qa", {}).get("training_status"),
        coverage_status=run.metrics.get("segment_qa", {}).get("coverage_status"),
        qa_report_text=(
            (base / "qa" / f"{run.id}.md").read_text(encoding="utf-8")[:200_000]
            if (base / "qa" / f"{run.id}.md").is_file() else None
        ),
        # The entry already reports manifest-level defects such as a missing
        # prepared input; the detail adds none of its own.
        problems=list(entry.problems),
    )


def _stage_views(run, scene_path: Path) -> list[StageView]:
    views: list[StageView] = []
    for name, record in run.stages.items():
        log_path = None
        log_present = False
        if record.log_path:
            resolved = scene_path / record.log_path
            log_path = str(resolved)
            log_present = resolved.is_file()
        views.append(
            StageView(
                name=name,
                status=record.status.value,
                started_at=record.started_at.isoformat() if record.started_at else None,
                completed_at=record.completed_at.isoformat() if record.completed_at else None,
                elapsed_seconds=record.elapsed_seconds,
                message=record.message,
                log_path=log_path,
                log_present=log_present,
            )
        )
    return views


def _evidence_views(run, scene_path: Path, work: Path) -> list[EvidenceView]:
    views: list[EvidenceView] = []

    def add(name: str, path: Path) -> None:
        views.append(EvidenceView(name=name, path=str(path), present=path.is_file()))

    prepared = work / run.config.input_relative_path / "dataset.json"
    add("prepared-input", prepared)
    for label in ("primary", "repair"):
        dataset = work / f"reconstruction-{label}"
        if not dataset.is_dir():
            continue
        for name in ("mask-filter.json", "mask-final.json", "segments.json"):
            add(f"{label}-{name}", dataset / name)
        add(f"selected-{label}-metrics", work / f"selected-{label}-metrics.jsonl")
    add("qa-report", scene_path / "qa" / f"{run.id}.md")
    return views


def _log_views(directory: Path) -> list[LogView]:
    if not directory.is_dir():
        return []
    views: list[LogView] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        stat = path.stat()
        views.append(
            LogView(
                name=path.relative_to(directory).as_posix(),
                path=str(path),
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
            )
        )
    return views


def _experiment_views(run, work: Path) -> list[ExperimentView]:
    from gsstudio.infrastructure.runtime.run_lock import is_run_active

    active = is_run_active(work)
    views: list[ExperimentView] = []
    for item in run.metrics.get("training_experiments", []):
        output = item.get("output")
        dispatch = Path(output) / "dispatch.json" if output else None
        views.append(
            ExperimentView(
                segment=item.get("segment"),
                backend=item.get("backend"),
                status=("needs_inspection" if item.get("status") == "preparing" and
                        not active else item.get("status")),
                output=output,
                error=item.get("error"),
                dispatch_path=str(dispatch) if dispatch else None,
                dispatch_present=bool(dispatch and dispatch.is_file()),
            )
        )
    return views

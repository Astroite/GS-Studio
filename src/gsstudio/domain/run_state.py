"""Authoritative Run stage state transitions."""
from __future__ import annotations
from gsstudio.domain.models import (
    STAGES, RunManifest, RunStatus, StageRecord, StageStatus, utc_now,
)

STAGE_ORDER = STAGES


def require_previous_stages(run: RunManifest, stage: str) -> None:
    index = STAGE_ORDER.index(stage)
    for previous in STAGE_ORDER[:index]:
        if previous == "qa":
            continue
        status = run.stages[previous].status
        if status != StageStatus.SUCCEEDED:
            raise RuntimeError(
                f"Stage {stage} requires {previous}=succeeded; current status is {status.value}"
            )

def begin_stage(
    run: RunManifest, stage: str, resume: bool = False, force: bool = False
) -> bool:
    """Open a stage for work, or report that a completed one can be reused."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown stage: {stage}")
    if stage not in ("preprocess", "qa"):
        require_previous_stages(run, stage)
    current = run.stages.get(stage, StageRecord())
    if current.status == StageStatus.SUCCEEDED and not force:
        if resume:
            return False
        raise RuntimeError(f"Stage {stage} already succeeded; use --resume to reuse it")
    if (
        current.status in (StageStatus.PROCESSING, StageStatus.FAILED)
        and not resume
        and not force
    ):
        raise RuntimeError(f"Stage {stage} was already attempted; use --resume or create a new run")
    run.stages[stage] = StageRecord(status=StageStatus.PROCESSING, started_at=utc_now())
    run.status = RunStatus.PROCESSING
    run.active_stage = stage
    return True

def complete_stage(
    run: RunManifest, stage: str, message: str | None = None, log_path: str | None = None
) -> None:
    record = run.stages[stage]
    record.status = StageStatus.SUCCEEDED
    record.completed_at = utc_now()
    if record.started_at:
        record.elapsed_seconds = (record.completed_at - record.started_at).total_seconds()
    record.message = message
    record.log_path = log_path
    run.active_stage = None
    if stage == "qa":
        run.status = RunStatus.NEEDS_REVIEW

def fail_stage(
    run: RunManifest, stage: str, message: str, log_path: str | None = None
) -> None:
    record = run.stages[stage]
    record.status = StageStatus.FAILED
    record.completed_at = utc_now()
    if record.started_at:
        record.elapsed_seconds = (record.completed_at - record.started_at).total_seconds()
    record.message = message
    record.log_path = log_path
    run.status = RunStatus.FAILED
    run.active_stage = None

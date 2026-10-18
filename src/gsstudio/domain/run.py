from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import Field
from gsstudio.domain.common import StrictModel, RunStatus, StageStatus, SLUG_PATTERN
from gsstudio.domain.config import RunConfig


class StageRecord(StrictModel):
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None
    message: str | None = None
    log_path: str | None = None


STAGES = ("preprocess", "mask", "reconstruct", "qa")


def default_stages() -> dict[str, StageRecord]:
    return {name: StageRecord() for name in STAGES}


class RunManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str
    location_id: str = Field(pattern=SLUG_PATTERN)
    scene_id: str = Field(pattern=SLUG_PATTERN)
    status: RunStatus = RunStatus.DRAFT
    config_hash: str
    config: RunConfig
    created_at: datetime
    updated_at: datetime
    active_stage: str | None = None
    stages: dict[str, StageRecord] = Field(default_factory=default_stages)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Reconstruction dataset this run trains from: reconstruction-primary|repair.
    selected_dataset: str | None = None
    review_notes: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import Field
from gsstudio.domain.common import StrictModel, RunStatus, SLUG_PATTERN


class Region(StrictModel):
    country_code: str = "CN"
    province: str | None = None
    city: str | None = None


class Rights(StrictModel):
    status: str = "needs_review"
    notes: str | None = None


class LocationManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=SLUG_PATTERN)
    display_name: str = Field(min_length=1)
    status: RunStatus = RunStatus.DRAFT
    capture_date: date | None = None
    region: Region = Field(default_factory=Region)
    tags: list[str] = Field(default_factory=list)
    rights: Rights = Field(default_factory=Rights)
    notes: str | None = None


class SceneManifest(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=SLUG_PATTERN)
    location_id: str = Field(pattern=SLUG_PATTERN)
    display_name: str = Field(min_length=1)
    status: RunStatus = RunStatus.DRAFT
    description: str | None = None
    metric_scale: bool = False
    quality_target: Literal["pipeline_proof", "reference_quality"] = "pipeline_proof"
    default_capture_id: str | None = None
    notes: str | None = None

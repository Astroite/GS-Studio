from __future__ import annotations

import ntpath
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"

# One capture shape and one run shape. Historical run schemas are not read.
CAPTURE_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 6


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StringEnum(str, Enum):
    """Python 3.10-compatible equivalent of enum.StrEnum."""

    def __str__(self) -> str:
        return self.value


class RunStatus(StringEnum):
    DRAFT = "draft"
    PROCESSING = "processing"
    FAILED = "failed"
    WAITING_REVIEW = "waiting_review"
    NEEDS_REVIEW = "needs_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class StageStatus(StringEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

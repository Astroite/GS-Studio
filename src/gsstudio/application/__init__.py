"""Shared GS-Studio status, operation and error interfaces.

Both front ends use the same pipeline through :mod:`operations`; the Qt worker
emits structured events and never interprets CLI text as Run state.
"""
from __future__ import annotations

from gsstudio.application.browse import (
    CaptureEntry,
    LocationEntry,
    Problem,
    RunEntry,
    SceneEntry,
    list_captures,
    list_locations,
    list_runs,
    list_scenes,
    load_run_entry,
    read_run_entry,
    require_location,
    require_scene,
    status_label,
)
from gsstudio.application.deps import CheckView, collect_checks, format_checks
from gsstudio.application.errors import (
    AlreadyExistsError,
    ExternalToolError,
    IntegrityError,
    InvalidInputError,
    LockedError,
    NotFoundError,
    PreconditionError,
    ServiceError,
    StorageError,
    UnsupportedError,
    classify,
)
from gsstudio.application.run_status import (
    EvidenceView,
    ExperimentView,
    LogView,
    RunDetail,
    StageView,
    read_run_detail,
)

__all__ = [
    "AlreadyExistsError",
    "CaptureEntry",
    "CheckView",
    "EvidenceView",
    "ExperimentView",
    "ExternalToolError",
    "IntegrityError",
    "InvalidInputError",
    "LocationEntry",
    "LockedError",
    "LogView",
    "NotFoundError",
    "PreconditionError",
    "Problem",
    "RunDetail",
    "RunEntry",
    "SceneEntry",
    "ServiceError",
    "StageView",
    "StorageError",
    "UnsupportedError",
    "classify",
    "collect_checks",
    "format_checks",
    "list_captures",
    "list_locations",
    "list_runs",
    "list_scenes",
    "load_run_entry",
    "read_run_detail",
    "read_run_entry",
    "require_location",
    "require_scene",
    "status_label",
]

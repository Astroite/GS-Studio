"""Error classes shared by the GS-Studio GUI and the ``gsstudio`` CLI.

Classification is for display, routing and scripting only: it never changes
which operation failed, its message, its exit status or how a stage is
recorded. Stage failure and recovery semantics stay in :mod:`gsstudio.infrastructure.persistence.run_repository` and
:mod:`gsstudio.pipeline.stages`.
"""
from __future__ import annotations

import os
from typing import Any, ClassVar


class ServiceError(Exception):
    """A classified operation error carrying a stable code for both front ends."""

    code: ClassVar[str] = "error"

    def __init__(self, message: str, *, path: str | None = None, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            payload["path"] = self.path
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class NotFoundError(ServiceError):
    code = "not_found"


class AlreadyExistsError(ServiceError):
    code = "already_exists"


class InvalidInputError(ServiceError):
    code = "invalid_input"


class PreconditionError(ServiceError):
    """A stage or command precondition is not met yet (not a failure of the work)."""

    code = "precondition_failed"


class IntegrityError(ServiceError):
    """A manifest, hash or identity check failed. Never hand-edit to clear it."""

    code = "integrity"


class UnsupportedError(ServiceError):
    code = "unsupported"


class ExternalToolError(ServiceError):
    code = "external_tool"


class LockedError(ServiceError):
    code = "locked"


class StorageError(ServiceError):
    code = "storage"


def _path_of(error: BaseException) -> str | None:
    filename = getattr(error, "filename", None)
    if filename:
        return str(filename)
    arguments = getattr(error, "args", ())
    if len(arguments) == 1 and isinstance(arguments[0], (str, os.PathLike)):
        return str(arguments[0])
    return None


def classify(error: BaseException) -> ServiceError:
    """Label ``error`` for display, preserving its message and original cause.

    Existing modules keep raising the exception types their callers and tests
    rely on; this maps those types onto the shared codes without rewriting them.
    """
    if isinstance(error, ServiceError):
        return error
    message = str(error) or type(error).__name__
    from gsstudio.pipeline.masks.finalize import MaskFinalizationMissingError
    from gsstudio.infrastructure.runtime.processes import CommandError

    if isinstance(error, MaskFinalizationMissingError):
        result: ServiceError = PreconditionError(message, detail="mask_review")
    elif isinstance(error, CommandError):
        result = ExternalToolError(message, path=str(error.log_path))
    elif isinstance(error, RuntimeError) and "Run is busy" in message:
        result = LockedError(message)
    elif isinstance(error, RuntimeError) and any(marker in message for marker in (
        "hash mismatch", "identity changed", "does not match", "has changed",
        "incomplete", "corrupt",
    )):
        result = IntegrityError(message, detail=type(error).__name__)
    elif isinstance(error, FileNotFoundError):
        result: ServiceError = NotFoundError(message, path=_path_of(error))
    elif isinstance(error, FileExistsError):
        result = AlreadyExistsError(message, path=_path_of(error))
    elif isinstance(error, (PermissionError, OSError)):
        result = StorageError(message, detail=type(error).__name__)
    elif isinstance(error, ValueError):
        result = InvalidInputError(message, detail=type(error).__name__)
    else:
        result = ServiceError(message, detail=type(error).__name__)
    result.__cause__ = error
    return result

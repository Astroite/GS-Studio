"""Dependency check views shared by the GS-Studio GUI and the ``gsstudio`` CLI.

:mod:`gsstudio.application.doctor` owns the checks themselves. This module only turns their
result into the PASS / FAIL / UNAVAILABLE labelling both front ends show, so a
check never changes meaning between the two, and so a GUI can reuse the same
optional-versus-required rule the CLI prints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gsstudio.application.doctor import run_doctor

# Optional unless the caller asks for them explicitly (``--require``) or picks
# the backend that needs them.
OPTIONAL_CHECKS = frozenset({"mediasdk"})


@dataclass
class CheckView:
    name: str
    ok: bool
    required: bool
    label: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "required": self.required,
            "label": self.label,
            "detail": self.detail,
        }


def format_checks(
    checks: dict[str, Any], *, require: Iterable[str] = (), backend: str = "gsplat"
) -> list[CheckView]:
    """Label each doctor result the way both front ends present it."""
    from gsstudio.pipeline.training.runner import require_backend
    if backend!='all':
        require_backend(backend)
    required = set(require)
    views: list[CheckView] = []
    for name, result in checks.items():
        if name == "ok":
            continue
        ok, detail = result
        optional = name in OPTIONAL_CHECKS and name not in required
        views.append(
            CheckView(
                name=name,
                ok=bool(ok),
                required=not optional,
                label="PASS" if ok else ("UNAVAILABLE" if optional else "FAIL"),
                detail=str(detail),
            )
        )
    return views


def collect_checks(
    data_root: Path,
    *,
    minimum_free_gib: float = 20.0,
    require: Iterable[str] = (),
    backend: str = "gsplat",
) -> tuple[bool, list[CheckView]]:
    """Run ``doctor`` and label it. Uses the GPU lock and may write temp files."""
    checks = run_doctor(
        data_root,
        minimum_free_gib=minimum_free_gib,
        require=set(require),
        backend=backend,
    )
    return bool(checks["ok"]), format_checks(checks, require=require, backend=backend)

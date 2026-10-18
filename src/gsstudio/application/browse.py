"""Read-only browsing of the Location / Scene / Capture / Run tree.

Every entry is derived from the manifest that owns it and from the evidence it
references. A directory, an exported file or a PLY never implies success:
unreadable manifests, missing sources and half-written runs are returned as
problems so both front ends can point at the file that shows the defect
(AC-20). Nothing here writes to the data tree.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gsstudio.infrastructure.persistence.manifests import load_capture_manifest, load_model
from gsstudio.domain.models import LocationManifest, RunManifest, SceneManifest
from gsstudio.infrastructure.paths import location_dir, scene_dir
from gsstudio.infrastructure.persistence.run_repository import load_run
from gsstudio.application.errors import NotFoundError


MISSING_MANIFEST = "missing_manifest"
UNREADABLE_MANIFEST = "unreadable_manifest"
CONFIG_HASH_MISMATCH = "config_hash_mismatch"
MISSING_SOURCE = "missing_source"
MISSING_INPUT = "missing_input"
STRAY_RUN_DIRECTORY = "stray_run_directory"

# Scene children that are never a run directory.
SCENE_DIRECTORIES = frozenset({"captures", "stitched", "exports", "qa", "logs", "prepared"})


@dataclass
class Problem:
    """One visible defect, tied to the manifest, log or result that shows it."""

    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path:
            payload["path"] = self.path
        return payload


@dataclass
class LocationEntry:
    id: str
    display_name: str | None
    status: str | None
    capture_date: str | None
    province: str | None
    city: str | None
    tags: list[str]
    manifest_path: str
    scene_count: int
    readable: bool = True
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneEntry:
    id: str
    location_id: str
    display_name: str | None
    status: str | None
    quality_target: str | None
    metric_scale: bool
    manifest_path: str
    readable: bool = True
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptureEntry:
    id: str
    location_id: str
    scene_id: str
    status: str | None
    source_kind: str | None
    projection: str | None
    camera: str | None
    files_total: int
    files_present: int
    media: dict[str, Any]
    selection: dict[str, float] | None
    manifest_path: str
    readable: bool = True
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunEntry:
    id: str
    location_id: str
    scene_id: str
    readable: bool
    status: str | None
    active_stage: str | None
    stages: dict[str, str]
    selected_dataset: str | None
    capture_id: str | None
    config_hash: str | None
    created_at: str | None
    updated_at: str | None
    manifest_path: str
    problems: list[Problem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def absolute_root(root: Path) -> Path:
    return Path(os.path.abspath(root))


def require_location(root: Path, location_id: str) -> Path:
    path = location_dir(absolute_root(root), location_id)
    if not (path / "location.yaml").is_file():
        raise NotFoundError(f"Location does not exist: {location_id}", path=str(path))
    return path


def require_scene(root: Path, location_id: str, scene_id: str) -> Path:
    path = scene_dir(absolute_root(root), location_id, scene_id)
    if not (path / "scene.yaml").is_file():
        raise NotFoundError(
            f"Scene does not exist: {location_id}/{scene_id}", path=str(path)
        )
    return path


def list_locations(root: Path) -> list[LocationEntry]:
    base = absolute_root(root)
    entries: list[LocationEntry] = []
    for manifest_path in sorted(base.glob("*/location.yaml")):
        if manifest_path.parent.name == ".external-imports":
            continue
        path = manifest_path.parent
        scenes = sorted(manifest_path.parent.glob("*/scene.yaml"))
        try:
            manifest = load_model(manifest_path, LocationManifest)
        except Exception as error:
            entries.append(
                LocationEntry(
                    id=path.name,
                    display_name=None,
                    status=None,
                    capture_date=None,
                    province=None,
                    city=None,
                    tags=[],
                    manifest_path=str(manifest_path),
                    scene_count=len(scenes),
                    readable=False,
                    problems=[
                        Problem(UNREADABLE_MANIFEST, str(error), str(manifest_path))
                    ],
                )
            )
            continue
        entries.append(
            LocationEntry(
                id=manifest.id,
                display_name=manifest.display_name,
                status=manifest.status.value,
                capture_date=manifest.capture_date.isoformat() if manifest.capture_date else None,
                province=manifest.region.province,
                city=manifest.region.city,
                tags=list(manifest.tags),
                manifest_path=str(manifest_path),
                scene_count=len(scenes),
            )
        )
    return entries


def list_scenes(root: Path, location_id: str) -> list[SceneEntry]:
    base = require_location(root, location_id)
    entries: list[SceneEntry] = []
    for manifest_path in sorted(base.glob("*/scene.yaml")):
        try:
            manifest = load_model(manifest_path, SceneManifest)
        except Exception as error:
            entries.append(
                SceneEntry(
                    id=manifest_path.parent.name,
                    location_id=location_id,
                    display_name=None,
                    status=None,
                    quality_target=None,
                    metric_scale=False,
                    manifest_path=str(manifest_path),
                    readable=False,
                    problems=[Problem(UNREADABLE_MANIFEST, str(error), str(manifest_path))],
                )
            )
            continue
        entries.append(
            SceneEntry(
                id=manifest.id,
                location_id=manifest.location_id,
                display_name=manifest.display_name,
                status=manifest.status.value,
                quality_target=manifest.quality_target,
                metric_scale=manifest.metric_scale,
                manifest_path=str(manifest_path),
            )
        )
    return entries


def list_captures(root: Path, location_id: str, scene_id: str) -> list[CaptureEntry]:
    path = require_scene(root, location_id, scene_id)
    entries: list[CaptureEntry] = []
    for manifest_path in sorted((path / "captures").glob("*.yaml")):
        entries.append(_capture_entry(manifest_path))
    return entries


def _capture_entry(manifest_path: Path) -> CaptureEntry:
    try:
        capture = load_capture_manifest(manifest_path)
    except Exception as error:
        return CaptureEntry(
            id=manifest_path.stem,
            location_id="",
            scene_id="",
            status=None,
            source_kind=None,
            projection=None,
            camera=None,
            files_total=0,
            files_present=0,
            media={},
            selection=None,
            manifest_path=str(manifest_path),
            readable=False,
            problems=[Problem(UNREADABLE_MANIFEST, str(error), str(manifest_path))],
        )
    problems: list[Problem] = []
    present = 0
    for record in capture.source.files:
        source = Path(record.windows_path)
        if source.is_file():
            present += 1
        else:
            problems.append(
                Problem(
                    MISSING_SOURCE,
                    f"Registered source file is missing: {record.windows_path}",
                    str(manifest_path),
                )
            )
    probe = capture.source.probe
    return CaptureEntry(
        id=capture.id,
        location_id=capture.location_id,
        scene_id=capture.scene_id,
        status=capture.status.value,
        source_kind=capture.source.kind,
        projection=capture.source.projection,
        camera=f"{capture.camera.make} {capture.camera.model}".strip(),
        files_total=len(capture.source.files),
        files_present=present,
        media={
            "width": probe.width,
            "height": probe.height,
            "fps": probe.fps,
            "duration_seconds": probe.duration_seconds,
        },
        selection={
            "start_seconds": capture.selection.start_seconds,
            "end_seconds": capture.selection.end_seconds,
        },
        manifest_path=str(manifest_path),
        problems=problems,
    )


def list_runs(root: Path, location_id: str, scene_id: str) -> list[RunEntry]:
    path = require_scene(root, location_id, scene_id)
    entries: list[RunEntry] = []
    for child in sorted(item for item in path.iterdir() if item.is_dir()):
        if child.name in SCENE_DIRECTORIES:
            continue
        if (child / "manifest.yaml").is_file():
            entries.append(read_run_entry(root, location_id, scene_id, child.name))
        elif _looks_like_run(child):
            entries.append(
                RunEntry(
                    id=child.name,
                    location_id=location_id,
                    scene_id=scene_id,
                    readable=False,
                    status=None,
                    active_stage=None,
                    stages={},
                    selected_dataset=None,
                    capture_id=None,
                    config_hash=None,
                    created_at=None,
                    updated_at=None,
                    manifest_path=str(child / "manifest.yaml"),
                    problems=[
                        Problem(
                            STRAY_RUN_DIRECTORY,
                            "Run directory has prepared work but no manifest; it is not a readable run",
                            str(child),
                        )
                    ],
                )
            )
    return entries


def _looks_like_run(path: Path) -> bool:
    """A directory that holds run work but lost or never got its manifest."""
    return (
        (path / "inputs").is_dir()
        or (path / "training-data").is_dir()
        or (path / "logs").is_dir()
        or any(path.glob("reconstruction-*"))
    )


def read_run_entry(root: Path, location_id: str, scene_id: str, run_id: str) -> RunEntry:
    return load_run_entry(root, location_id, scene_id, run_id)[0]


def load_run_entry(
    root: Path, location_id: str, scene_id: str, run_id: str
) -> tuple[RunEntry, RunManifest | None]:
    """Entry for one run together with its manifest (``None`` when unreadable)."""
    scene_path = absolute_root(root) / location_id / scene_id
    manifest_path = scene_path / run_id / "manifest.yaml"

    def broken(problems: list[Problem]) -> tuple[RunEntry, None]:
        return (
            RunEntry(
                id=run_id,
                location_id=location_id,
                scene_id=scene_id,
                readable=False,
                status=None,
                active_stage=None,
                stages={},
                selected_dataset=None,
                capture_id=None,
                config_hash=None,
                created_at=None,
                updated_at=None,
                manifest_path=str(manifest_path),
                problems=problems,
            ),
            None,
        )

    if not manifest_path.is_file():
        return broken([Problem(MISSING_MANIFEST, "Run manifest is missing", str(manifest_path))])
    try:
        run = load_run(scene_path, run_id)
    except RuntimeError as error:
        # load_run only raises RuntimeError for its stored-identity check.
        return broken([Problem(CONFIG_HASH_MISMATCH, str(error), str(manifest_path))])
    except FileNotFoundError as error:
        return broken([Problem(MISSING_MANIFEST, str(error), str(manifest_path))])
    except Exception as error:
        return broken([Problem(UNREADABLE_MANIFEST, str(error), str(manifest_path))])
    problems: list[Problem] = []
    prepared = scene_path / run_id / run.config.input_relative_path
    if not (prepared / "dataset.json").is_file():
        problems.append(
            Problem(
                MISSING_INPUT,
                f"Prepared input manifest is missing: {run.config.input_relative_path}/dataset.json",
                str(prepared),
            )
        )
    from gsstudio.infrastructure.runtime.run_lock import is_run_active

    visible_status = run.status.value
    if visible_status == "processing" and run.active_stage and not is_run_active(scene_path / run_id):
        visible_status = "needs_inspection"
        problems.append(Problem(
            "interrupted_stage",
            f"Stage {run.active_stage} has no active Run lock; inspect its evidence before resuming",
            str(manifest_path),
        ))
    return (
        RunEntry(
            id=run.id,
            location_id=run.location_id,
            scene_id=run.scene_id,
            readable=True,
            status=visible_status,
            active_stage=run.active_stage,
            stages={name: record.status.value for name, record in run.stages.items()},
            selected_dataset=run.selected_dataset,
            capture_id=run.config.capture_id,
            config_hash=run.config_hash,
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
            manifest_path=str(manifest_path),
            problems=problems,
        ),
        run,
    )


def status_label(entry: LocationEntry | SceneEntry | CaptureEntry | RunEntry) -> str:
    """Row text for any entry list. Unreadable work never reads as a real status."""
    return entry.status if entry.readable and entry.status else "unreadable"

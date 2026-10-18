from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gsstudio.infrastructure.persistence.manifests import canonical_hash, load_model, save_yaml
from gsstudio.domain.models import (
    STAGES,
    RunConfig,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
    utc_now,
)
from gsstudio.infrastructure.paths import ensure_run_dir



def run_manifest_path(scene_path: Path, run_id: str) -> Path:
    return scene_path / run_id / "manifest.yaml"


def load_run(scene_path: Path, run_id: str) -> RunManifest:
    run = load_model(run_manifest_path(scene_path, run_id), RunManifest)
    current_hash = canonical_hash(run.config)
    if current_hash != run.config_hash:
        raise RuntimeError(
            f"Run config hash mismatch for {run_id}; configuration changes require a new run"
        )
    return run


def save_run(scene_path: Path, run: RunManifest) -> None:
    run.updated_at = utc_now()
    save_yaml(run_manifest_path(scene_path, run.id), run)


def create_run(
    scene_path: Path,
    location_id: str,
    scene_id: str,
    config: RunConfig,
    now: datetime | None = None,
    run_id: str | None = None,
) -> RunManifest:
    created_at = now or utc_now()
    digest = canonical_hash(config)
    identifier = run_id or f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-{digest[:8]}"
    work = scene_path / identifier
    if (work / "manifest.yaml").exists():
        raise FileExistsError(f"Run already exists: {identifier}")
    if work.exists():
        # New Runs prepare their owned, content-addressed input before the
        # manifest can pin its hash. Only that verified staging layout may be
        # adopted; an arbitrary stray directory must remain visible as a defect.
        from gsstudio.infrastructure.persistence.prepared_input import load_prepared_input

        if {child.name for child in work.iterdir()} != {"inputs"}:
            raise FileExistsError(f"Unrecognized Run staging directory: {work}")
        prepared = load_prepared_input(work / config.input_relative_path)
        if prepared.dataset_sha256 != config.input_dataset_sha256:
            raise RuntimeError("Prepared input does not match the new Run config")
    run = RunManifest(
        id=identifier,
        location_id=location_id,
        scene_id=scene_id,
        config_hash=digest,
        config=config,
        created_at=created_at,
        updated_at=created_at,
    )
    ensure_run_dir(scene_path, identifier, exist_ok=True)
    save_run(scene_path, run)
    return run

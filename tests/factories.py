"""CPU-only builders shared by the gsstudio tests: a synthetic data tree with one
location, scene, capture and run. Nothing here touches the real Data tree."""
from __future__ import annotations

from pathlib import Path

from gsstudio.infrastructure.persistence.manifests import save_yaml
from gsstudio.domain.models import (
    Camera,
    CaptureManifest,
    CaptureSource,
    LocationManifest,
    NormalizationSettings,
    PreparedInputConfig,
    Region,
    RunConfig,
    SceneManifest,
    SourceFile,
    SourceProbe,
    TimeSelection,
)
from gsstudio.infrastructure.persistence.run_repository import create_run

DIGEST = "0" * 64
RUN_ID = "20260922T010203Z-deadbeef"
STRAY_ID = "20260922T000000Z-stray00"


def build_capture(tmp_path: Path) -> CaptureManifest:
    source = tmp_path / "raw" / "clip.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"0" * 10)
    return CaptureManifest(
        id="capture",
        location_id="loc",
        scene_id="scene",
        source=CaptureSource(
            kind="equirect_video",
            files=[
                SourceFile(
                    windows_path=str(source),
                    sha256=DIGEST,
                    byte_size=source.stat().st_size,
                )
            ],
            projection="equirectangular",
            probe=SourceProbe(width=3840, height=1920, fps=30.0, duration_seconds=10.0),
        ),
        camera=Camera(make="Insta360", model="X5"),
        selection=TimeSelection(start_seconds=0.0, end_seconds=10.0),
        normalization=NormalizationSettings(width=3840, height=1920),
    )


def build_config(capture: CaptureManifest) -> RunConfig:
    return RunConfig(
        capture_id=capture.id,
        input_dataset_sha256=DIGEST,
        input_relative_path=f"inputs/primary/{DIGEST}",
        input=PreparedInputConfig(
            source_kind=capture.source.kind,
            source_sha256=[item.sha256 for item in capture.source.files],
            source_probe=capture.source.probe,
            normalization=capture.normalization,
            selection=capture.selection,
            candidate_frame_indices=[0, 1],
            candidate_fps=5.0,
        ),
        preprocess={"candidate_fps": 5.0, "selected_per_second": 1},
        reconstruction={
            "primary": {
                "temporal_rank_limit": 1,
                "projection_fov_degrees": 110.0,
                "panorama": {"views": 14, "size": 1746, "crop_bottom": 0.15},
            }
        },
    )


def build_tree(tmp_path: Path) -> dict:
    root = tmp_path / "Data"
    capture = build_capture(tmp_path)
    save_yaml(
        root / "loc" / "location.yaml",
        LocationManifest(id="loc", display_name="燕园", region=Region(province="北京")),
    )
    save_yaml(
        root / "loc" / "scene" / "scene.yaml",
        SceneManifest(id="scene", location_id="loc", display_name="夜游"),
    )
    save_yaml(root / "loc" / "scene" / "captures" / "capture.yaml", capture)
    run = create_run(root / "loc" / "scene", "loc", "scene", build_config(capture), run_id=RUN_ID)
    prepared = root / "loc" / "scene" / RUN_ID / "inputs" / "primary" / DIGEST
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "dataset.json").write_text("{}", encoding="utf-8")
    return {"root": root, "capture": capture, "run": run}

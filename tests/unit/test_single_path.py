"""CPU tests for the single-path pipeline: chunked extraction, source validation,
perspective staging counts, retention whitelist and the select-sharp workbench."""
from __future__ import annotations

from pathlib import Path

import pytest

from gsstudio.infrastructure.adapters.media import MAX_SELECT_TERMS, extract_indexed_frames
from gsstudio.domain.models import (
    Camera,
    CaptureManifest,
    CaptureSource,
    NormalizationSettings,
    SourceFile,
    TimeSelection,
)
from gsstudio.pipeline.reconstruction.core import expected_planar_images, projection_view_specs
from gsstudio.pipeline.retention import _disposable
from gsstudio.pipeline.input.sharp_select import parse_window, reconcile, unique_names


DIGEST = "0" * 64


def _capture(kind: str, *, size: tuple[int, int] | None = None, horizontal_fov: float | None = None):
    projection = {
        "insta360_insv": "dual_fisheye",
        "perspective_video": "perspective",
    }.get(kind, "equirectangular")
    suffix = {"equirect_sequence": ".jpg"}.get(kind, ".insv" if kind == "insta360_insv" else ".mp4")
    return CaptureManifest(
        id="capture",
        location_id="loc",
        scene_id="scene",
        source=CaptureSource(
            kind=kind,
            files=[SourceFile(windows_path=f"D:\\raw\\clip{suffix}", sha256=DIGEST, byte_size=10)],
            projection=projection,
        ),
        camera=Camera(make="m", model="m"),
        selection=TimeSelection(start_seconds=0.0, end_seconds=1.0),
        normalization=NormalizationSettings(width=size[0], height=size[1]) if size else NormalizationSettings(),
        horizontal_fov_degrees=horizontal_fov,
    )


def test_perspective_capture_rejects_normalization_and_requires_fov():
    with pytest.raises(ValueError, match="horizontal_fov_degrees"):
        _capture("perspective_video")
    with pytest.raises(ValueError, match="normalization dimensions must be unset"):
        _capture("perspective_video", size=(3840, 1920), horizontal_fov=84.0)
    capture = _capture("perspective_video", horizontal_fov=84.0)
    assert capture.source.is_perspective
    assert capture.normalization.width is None


def test_panorama_capture_rejects_fov_and_keeps_2_to_1():
    with pytest.raises(ValueError, match="applies only to perspective_video"):
        _capture("equirect_video", size=(3840, 1920), horizontal_fov=84.0)
    assert _capture("equirect_video", size=(3840, 1920)).normalization.height == 1920
    with pytest.raises(ValueError, match="exactly 2:1"):
        _capture("insta360_insv", size=(3840, 2160))
    assert _capture("insta360_insv").source.projection == "dual_fisheye"


def test_extract_indexed_frames_chunks_at_the_ffmpeg_limit(tmp_path: Path, monkeypatch):
    """100 select terms parse; the 101st does not, so requests must be chunked."""
    from gsstudio.infrastructure.adapters import media

    calls: list[list[str]] = []

    def fake_run_logged(command, log_path):
        calls.append(command)
        destination = Path(command[-1]).parent
        start = int(command[command.index("-start_number") + 1])
        count = int(command[command.index("-frames:v") + 1])
        for ordinal in range(start, start + count):
            (destination / f"frame_{ordinal:06d}.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr(media, "run_logged", fake_run_logged)
    indices = list(range(0, 205))
    frames = extract_indexed_frames(tmp_path / "clip.mp4", tmp_path / "out", indices, 95, tmp_path / "log.txt")

    assert [int(call[call.index("-start_number") + 1]) for call in calls] == [1, 101, 201]
    assert [int(call[call.index("-frames:v") + 1]) for call in calls] == [100, 100, 5]
    for call in calls:
        select = call[call.index("-vf") + 1]
        assert select.count("eq(n") <= MAX_SELECT_TERMS
    assert [path.name for path in frames] == [f"frame_{i:06d}.jpg" for i in range(1, 206)]


def test_retention_whitelist_covers_both_projection_layouts():
    assert _disposable("inputs/primary/" + DIGEST + "/frames/frame_000001.jpg")
    assert _disposable("frames-primary/frame_000042.jpg")
    assert _disposable("reconstruction-primary/images/view_13/frame_000007.jpg")
    assert _disposable("reconstruction-primary/masks/view_13/frame_000007.jpg.png")
    # A perspective source stages one frame per view under its own name.
    assert _disposable("reconstruction-primary/images/frame_000007.jpg")
    assert _disposable("reconstruction-repair/masks/frame_000007.jpg.png")
    assert _disposable("reconstruction-primary/alignment-images/gsdb_003.jpg.mask.png")
    assert not _disposable("reconstruction-primary/segments.json")
    assert not _disposable("inputs/primary/" + DIGEST + "/dataset.json")


def test_select_sharp_window_and_reconcile(tmp_path: Path):
    assert parse_window("") == (0.0, None)
    assert parse_window("5..30") == (5.0, 30.0)
    assert parse_window("..12.5") == (0.0, 12.5)
    with pytest.raises(ValueError):
        parse_window("30")
    with pytest.raises(ValueError):
        parse_window("30..10")
    assert unique_names([tmp_path / "a.mp4", tmp_path / "A.mp4", tmp_path / "b.mp4"]) == [
        "a",
        "a-2",
        "b",
    ]

    # Re-running filter with a new threshold reconciles instead of appending.
    kept = tmp_path / "kept"
    (kept / "view_00").mkdir(parents=True)
    (kept / "view_00" / "frame_000001.jpg").write_bytes(b"a")
    (kept / "view_00" / "frame_000002.jpg").write_bytes(b"b")
    (kept / "frame_000003.jpg").write_bytes(b"c")
    removed = reconcile(kept, {"view_00/frame_000001.jpg", "frame_000003.jpg"})
    assert removed == 1
    assert sorted(p.relative_to(kept).as_posix() for p in kept.rglob("*") if p.is_file()) == [
        "frame_000003.jpg",
        "view_00/frame_000001.jpg",
    ]


def test_perspective_attempt_stages_one_view_per_frame():
    from gsstudio.domain.models import ReconstructionAttempt

    perspective = ReconstructionAttempt(temporal_rank_limit=1, projection_fov_degrees=84.0)
    assert expected_planar_images(perspective, 37) == 37
    with pytest.raises(ValueError, match="no projected view layout"):
        projection_view_specs(perspective)


def test_run_config_rejects_inconsistent_temporal_sampling():
    from gsstudio.domain.models import PreparedInputConfig, RunConfig, SourceProbe

    def config(candidate_fps: float, selected: int, rank: int):
        return RunConfig(
            capture_id="capture",
            input_dataset_sha256=DIGEST,
            input_relative_path=f"inputs/primary/{DIGEST}",
            input=PreparedInputConfig(
                source_kind="perspective_video",
                source_sha256=[DIGEST],
                source_probe=SourceProbe(width=3840, height=2160, fps=30.0),
                normalization=NormalizationSettings(),
                selection=TimeSelection(start_seconds=0.0, end_seconds=10.0),
                candidate_frame_indices=[0, 1],
                candidate_fps=candidate_fps,
            ),
            preprocess={"candidate_fps": candidate_fps, "selected_per_second": selected},
            reconstruction={"primary": {"temporal_rank_limit": rank, "projection_fov_degrees": 84.0}},
        )

    assert config(5.0, 2, 1).reconstruction.primary.panorama is None
    with pytest.raises(ValueError, match="temporal sampling"):
        config(1.0, 1, 2)
    mismatched = config(5.0, 1, 1)
    with pytest.raises(ValueError, match="must match preprocess.candidate_fps"):
        RunConfig.model_validate(
            {**mismatched.model_dump(mode="json"), "preprocess": {"candidate_fps": 4.0, "selected_per_second": 1}}
        )


def test_run_manifest_round_trips_without_changing_its_config_hash(tmp_path: Path):
    from gsstudio.domain.models import PreparedInputConfig, RunConfig, SourceProbe
    from gsstudio.infrastructure.persistence.run_repository import create_run, load_run, save_run

    config = RunConfig(
        capture_id="capture",
        input_dataset_sha256=DIGEST,
        input_relative_path=f"inputs/primary/{DIGEST}",
        input=PreparedInputConfig(
            source_kind="perspective_video",
            source_sha256=[DIGEST],
            source_probe=SourceProbe(width=3840, height=2160, fps=30.0),
            normalization=NormalizationSettings(),
            selection=TimeSelection(start_seconds=0.0, end_seconds=10.0),
            candidate_frame_indices=[0, 1],
            candidate_fps=5.0,
        ),
        preprocess={"candidate_fps": 5.0, "selected_per_second": 1},
        reconstruction={
            "primary": {"temporal_rank_limit": 1, "projection_fov_degrees": 84.0},
        },
    )
    run = create_run(tmp_path, "loc", "scene", config, run_id="20260922T000000Z-deadbeef")
    assert set(run.stages) == {"preprocess", "mask", "reconstruct", "qa"}
    assert run.config.reconstruction.primary.panorama is None

    reloaded = load_run(tmp_path, run.id)
    assert reloaded.config_hash == run.config_hash
    assert reloaded.config == config
    save_run(tmp_path, reloaded)
    assert load_run(tmp_path, run.id).config_hash == run.config_hash

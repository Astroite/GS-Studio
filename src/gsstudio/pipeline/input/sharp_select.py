"""Workbench: pull the sharpest frames out of explicit sources and drop person-heavy views.

Sits outside the Run pipeline on purpose -- no capture manifest, no run directory,
no training -- so a single clip can be inspected before it is worth registering.
It reuses the pipeline's own extraction, scoring, projection and masking so the two
paths cannot drift: the same MediaSDK/FFmpeg adapters, the same composite frame
score, the same 14x110 deg panorama split, the same Mask R-CNN person filter.

Every stage is idempotent. An interrupted run is resumed by repeating the same
command: prepared candidates are content-addressed and verified, projection skips
views that already pass their size check, masking reuses existing masks, and the
filter reconciles kept/rejected instead of appending to them.

This is not a quality acceptance path.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.pipeline.masks.masking import (
    MaskingConfig,
    TorchvisionPersonSegmenter,
    generate_person_masks,
    image_files,
    validate_mask_set,
)
from gsstudio.infrastructure.adapters.media import analyze_frames, create_temporal_subset
from gsstudio.domain.models import (
    Camera,
    CaptureManifest,
    CaptureSource,
    NormalizationSettings,
    PanoramaProjection,
    ReconstructionAttempt,
    SourceFile,
    SourceProbe,
    TimeSelection,
)
from gsstudio.pipeline.reconstruction.core import project_equirectangular_frames, projection_view_specs
from gsstudio.pipeline.input.sources import load_prepared_input, prepare_capture_input
from gsstudio.infrastructure.persistence.storage import link_or_copy

STAGES = ("probe", "extract", "select", "project", "mask", "filter")
EQUIRECT_KINDS = ("equirect_video", "equirect_sequence", "insta360_insv")

# Mirrors the panorama defaults recorded in docs/CURRENT-WORKFLOW.md.
DEFAULT_VIEWS = 14
DEFAULT_VIEW_FOV = 110.0
DEFAULT_VIEW_SIZE = 1746
DEFAULT_CROP_BOTTOM = 0.15


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass
class Settings:
    sources: list[Path]
    out: Path
    kind: str | None
    stage: str
    start_seconds: float
    end_seconds: float | None
    candidate_fps: float
    selected_per_second: int
    mask_threshold: float
    views: int
    view_fov: float
    view_size: int
    crop_bottom: float
    target: str
    keep_candidates: bool
    jpeg_quality: int
    device: str

    @property
    def stages(self) -> tuple[str, ...]:
        return STAGES if self.stage == "all" else (self.stage,)

    @property
    def attempt(self) -> ReconstructionAttempt:
        return ReconstructionAttempt(
            temporal_rank_limit=1,
            projection_fov_degrees=self.view_fov,
            panorama=PanoramaProjection(
                views=self.views, size=self.view_size, crop_bottom=self.crop_bottom
            ),
        )


def parse_window(value: str) -> tuple[float, float | None]:
    if not value.strip():
        return 0.0, None
    text = value.strip()
    if ".." not in text:
        raise ValueError("--window must look like START..END (either side may be empty)")
    raw_start, raw_end = text.split("..", 1)
    start = float(raw_start) if raw_start.strip() else 0.0
    end = float(raw_end) if raw_end.strip() else None
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"Invalid --window: {value}")
    return start, end


def unique_names(paths: list[Path]) -> list[str]:
    names: list[str] = []
    used: dict[str, int] = {}
    for path in paths:
        base = path.stem.strip().lower().replace(" ", "-") or "source"
        used[base] = used.get(base, 0) + 1
        names.append(base if used[base] == 1 else f"{base}-{used[base]}")
    return names


def infer_kind(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".insv":
        return "insta360_insv"
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        return "equirect_sequence"
    if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        raise ValueError(f"Unsupported source extension: {path}")
    # A video is either a 2:1 panorama or a landscape perspective clip; the
    # aspect ratio decides, since both are plain video files.
    from gsstudio.infrastructure.adapters.media import probe_video

    probe = probe_video(path)
    width, height = int(probe["width"]), int(probe["height"])
    if abs(width / height - 2.0) <= 0.01:
        return "equirect_video"
    if width > height:
        return "perspective_video"
    raise ValueError(f"Cannot classify {path.name}: {width}x{height} is neither 2:1 nor landscape")


def build_capture(name: str, kind: str, path: Path, settings: Settings) -> CaptureManifest:
    """A throwaway capture document so the pipeline's adapters can be reused."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    projections = {
        "insta360_insv": "dual_fisheye",
        "perspective_video": "perspective",
    }
    perspective = kind == "perspective_video"
    return CaptureManifest(
        id=name,
        location_id="standalone",
        scene_id="standalone",
        source=CaptureSource(
            kind=kind,
            files=[
                SourceFile(
                    windows_path=str(path.absolute()),
                    sha256=digest.hexdigest(),
                    byte_size=path.stat().st_size,
                )
            ],
            projection=projections.get(kind, "equirectangular"),
            probe=SourceProbe(),
        ),
        camera=Camera(make="unknown", model="unknown"),
        selection=TimeSelection(start_seconds=0.0, end_seconds=1.0),
        normalization=NormalizationSettings(jpeg_quality=settings.jpeg_quality),
        # Focal prior for RealityScan only; unused here beyond model validation.
        horizontal_fov_degrees=84.0 if perspective else None,
    )


def source_dir(settings: Settings, name: str) -> Path:
    return settings.out / name


def manifest_path(settings: Settings) -> Path:
    return settings.out / "sources.json"


def load_captures(settings: Settings) -> list[tuple[str, CaptureManifest]]:
    stored = manifest_path(settings)
    if not stored.is_file():
        raise RuntimeError(f"{stored} is missing; run --stage probe first")
    payload = json.loads(stored.read_text(encoding="utf-8"))
    captures = []
    for record in payload["sources"]:
        path = Path(record["path"])
        stat = path.stat()
        if stat.st_size != int(record["byte_size"]) or stat.st_mtime_ns != int(record["mtime_ns"]):
            raise RuntimeError(f"{path} changed since probe (size/mtime differ); re-run --stage probe")
        kind = str(record["kind"])
        name = str(record["name"])
        capture = build_capture(name, kind, path, settings)
        capture.source.probe = SourceProbe.model_validate(record["probe"])
        capture.selection = TimeSelection(
            start_seconds=float(record["selection"]["start_seconds"]),
            end_seconds=float(record["selection"]["end_seconds"]),
        )
        captures.append((name, capture))
    return captures


# --------------------------------------------------------------------------- probe


def probe_stage(settings: Settings) -> list[tuple[str, CaptureManifest]]:
    settings.out.mkdir(parents=True, exist_ok=True)
    names = unique_names(settings.sources)
    records = []
    captures: list[tuple[str, CaptureManifest]] = []
    for name, path in zip(names, settings.sources):
        if not path.is_file():
            raise FileNotFoundError(path)
        kind = settings.kind or infer_kind(path)
        capture = build_capture(name, kind, path, settings)
        root = source_dir(settings, name)
        root.mkdir(parents=True, exist_ok=True)
        from gsstudio.pipeline.input.sources import probe_capture_source

        probe = probe_capture_source(capture, root / "protocol")
        capture.source.probe = SourceProbe.model_validate(
            {key: value for key, value in probe.items() if key in SourceProbe.model_fields}
        )
        duration = float(probe["duration_seconds"])
        end = min(settings.end_seconds if settings.end_seconds is not None else duration, duration)
        if end <= settings.start_seconds:
            raise ValueError(f"{name}: window [{settings.start_seconds}, {end}] is empty")
        capture.selection = TimeSelection(start_seconds=settings.start_seconds, end_seconds=end)
        stat = path.stat()
        records.append(
            {
                "name": name,
                "kind": kind,
                "path": str(path.absolute()),
                "byte_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": capture.source.files[0].sha256,
                "probe": capture.source.probe.model_dump(mode="json"),
                "selection": {"start_seconds": settings.start_seconds, "end_seconds": end},
            }
        )
        captures.append((name, capture))
        log(
            f"{name}: {kind} {int(probe['width'])}x{int(probe['height'])} @ {probe['fps']} fps, "
            f"{duration:.3f}s, window [{settings.start_seconds:.3f}, {end:.3f}]"
        )
    write_json(
        manifest_path(settings),
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": "standalone (no Run, no capture manifest)",
            "parameters": {
                "candidate_fps": settings.candidate_fps,
                "selected_per_second": settings.selected_per_second,
                "selection_algorithm": "composite_v1",
                "mask_threshold": settings.mask_threshold,
                "jpeg_quality": settings.jpeg_quality,
                "keep_candidates": settings.keep_candidates,
            },
            "sources": records,
        },
    )
    return captures


# ------------------------------------------------------------------------- extract


def extract_stage(settings: Settings) -> None:
    for name, capture in load_captures(settings):
        root = source_dir(settings, name)
        candidates = root / "prepared"
        log(f"{name}: preparing candidates at {settings.candidate_fps:g} fps ...")
        prepared = prepare_capture_input(
            settings.out,
            capture,
            candidates,
            settings.candidate_fps,
            selection_end_seconds=capture.selection.end_seconds,
            resume=True,
        )
        log(f"{name}: {len(prepared.frame_paths)} candidates verified")


# -------------------------------------------------------------------------- select


def prepared_candidates(settings: Settings, name: str):
    root = source_dir(settings, name)
    directories = (
        sorted(item for item in (root / "prepared").iterdir() if item.is_dir() and len(item.name) == 64)
        if (root / "prepared").is_dir()
        else []
    )
    if not directories:
        raise RuntimeError(f"{name}: no prepared candidates; run --stage extract first")
    return load_prepared_input(directories[-1])


def select_stage(settings: Settings) -> None:
    for name, capture in load_captures(settings):
        root = source_dir(settings, name)
        prepared = prepared_candidates(settings, name)
        duration = capture.selection.end_seconds - capture.selection.start_seconds
        log(f"{name}: scoring {len(prepared.frame_paths)} candidates (composite_v1) ...")
        metrics = analyze_frames(
            list(prepared.frame_paths),
            duration,
            root / "candidates.jsonl",
            start_seconds=capture.selection.start_seconds,
            timestamps_seconds=prepared.timestamps_seconds,
        )
        outputs, rewritten = create_temporal_subset(
            prepared.path / "frames",
            metrics,
            root / "selected",
            capture.selection.start_seconds,
            settings.selected_per_second,
            rank_limit=None,
            selection_metric="selection_score",
        )
        write_jsonl(root / "selected.jsonl", rewritten)
        buckets = [int(item["time_bucket"]) for item in rewritten]
        if len(set(buckets)) != len(buckets):
            raise RuntimeError(f"{name}: a one-second bucket holds two frames")
        log(f"{name}: selected {len(outputs)} frames ({len(set(buckets))} seconds)")
        if not settings.keep_candidates:
            # selected/ entries are hard links, so dropping the candidate names
            # leaves the data intact.
            log(f"{name}: removing prepared candidates (hard links keep selected/ intact)")
            shutil.rmtree(prepared.path, ignore_errors=True)


# ------------------------------------------------------------------------- project


def project_stage(settings: Settings) -> None:
    for name, capture in load_captures(settings):
        if capture.source.kind not in EQUIRECT_KINDS:
            log(f"{name}: {capture.source.kind} is already one frame per view; nothing to project")
            continue
        root = source_dir(settings, name)
        frames = image_files(root / "selected")
        if not frames:
            raise RuntimeError(f"{name}: no selected frames in {root / 'selected'}")
        expected_names = [f"frame_{index:06d}.jpg" for index in range(1, len(frames) + 1)]
        if [item.name for item in frames] != expected_names:
            raise RuntimeError(f"{name}: selected/ must be a contiguous frame_000001.. sequence")
        specs = projection_view_specs(settings.attempt)
        log(
            f"{name}: projecting {len(frames)} panoramas into {len(specs)} views of "
            f"{settings.view_size}px at {settings.view_fov:g} deg ..."
        )
        with gpu_session():
            produced = project_equirectangular_frames(
                root / "selected", root / "planar", settings.attempt
            )
        if len(produced) != len(frames) * len(specs):
            raise RuntimeError(
                f"{name}: projection produced {len(produced)} images; "
                f"expected {len(frames) * len(specs)}"
            )
        buckets = {
            str(record.get("file")): record.get("time_bucket")
            for record in read_jsonl(root / "selected.jsonl")
        }
        index = []
        for position, frame in enumerate(frames, start=1):
            for view, (yaw, pitch) in enumerate(specs):
                index.append(
                    {
                        "image": f"view_{view:02d}/frame_{position:06d}.jpg",
                        "view": view,
                        "yaw": yaw,
                        "pitch": pitch,
                        "equirect_image": frame.name,
                        "time_bucket": buckets.get(frame.name),
                    }
                )
        write_jsonl(root / "planar" / "selected.jsonl", index)
        log(f"{name}: {len(produced)} views under {root / 'planar' / 'images'}")


# ---------------------------------------------------------------------------- mask


def mask_targets(settings: Settings, name: str, capture: CaptureManifest) -> list[str]:
    """Which image set mask/filter act on.

    A person beside a 360 rig occupies a small slice of the sphere but a large
    share of the whole equirectangular image, so a full-frame fraction rejects
    every frame. Splitting into views localises them to a few of the fourteen.
    """
    root = source_dir(settings, name)
    planar = (root / "planar" / "images").is_dir()
    if settings.target == "flat":
        return ["flat"]
    if settings.target == "planar":
        return ["planar"] if planar else []
    if settings.target == "all":
        return ["flat", "planar"] if planar else ["flat"]
    return ["planar"] if (planar and capture.source.kind in EQUIRECT_KINDS) else ["flat"]


def target_paths(settings: Settings, name: str, label: str) -> dict[str, Path]:
    root = source_dir(settings, name)
    base = root / "planar" if label == "planar" else root
    return {
        "images": (base / "images") if label == "planar" else root / "selected",
        "masks": base / "masks",
        "metrics": base / "mask-metrics.jsonl",
        "kept": base / "kept",
        "rejected": base / "rejected",
    }


def mask_stage(settings: Settings) -> None:
    for name, capture in load_captures(settings):
        for label in mask_targets(settings, name, capture):
            paths = target_paths(settings, name, label)
            images = image_files(paths["images"])
            if not images:
                raise RuntimeError(f"{name}/{label}: no images in {paths['images']}")
            config = MaskingConfig(
                enabled=True,
                device=settings.device,
                classes=["person"],
                mask_discard_threshold=settings.mask_threshold,
            )
            log(f"{name}/{label}: person masks for {len(images)} images on {settings.device} ...")
            with gpu_session():
                records = generate_person_masks(
                    paths["images"], paths["masks"], config, paths["metrics"],
                    TorchvisionPersonSegmenter(config),
                )
            summary = validate_mask_set(paths["images"], paths["masks"], 0.99)
            log(
                f"{name}/{label}: {len(records)} masks, "
                f"mean {summary['mean_masked_fraction']:.5f}, "
                f"max {summary['max_masked_fraction']:.5f}"
            )


# -------------------------------------------------------------------------- filter


def reconcile(directory: Path, desired: set[str]) -> int:
    """Drop stale names so re-running with a new threshold is not additive."""
    if not directory.is_dir():
        return 0
    removed = 0
    for item in sorted(directory.rglob("*"), reverse=True):
        if item.is_file():
            if item.relative_to(directory).as_posix() not in desired:
                item.unlink()
                removed += 1
        elif item.is_dir() and not any(item.iterdir()):
            item.rmdir()
    return removed


def filter_stage(settings: Settings) -> dict:
    summary_path = settings.out / "summary.json"
    summary: dict = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    summary.update(
        {
            "schema_version": 1,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "mask_threshold": settings.mask_threshold,
            "comparison": "masked_fraction > threshold is rejected; equal is kept",
        }
    )
    sources = summary.setdefault("sources", {})

    for name, capture in load_captures(settings):
        entry = sources.setdefault(name, {})
        entry.update(
            {
                "source_path": str(Path(capture.source.files[0].windows_path)),
                "source_sha256": capture.source.files[0].sha256,
                "kind": capture.source.kind,
                "window_seconds": [capture.selection.start_seconds, capture.selection.end_seconds],
            }
        )
        for label in mask_targets(settings, name, capture):
            paths = target_paths(settings, name, label)
            records = read_jsonl(paths["metrics"])
            images = image_files(paths["images"])
            names = {item.relative_to(paths["images"]).as_posix() for item in images}
            if {str(item["image"]) for item in records} != names:
                raise RuntimeError(
                    f"{name}/{label}: mask records do not cover the images exactly; "
                    f"run --stage mask"
                )
            kept: list[dict] = []
            rejected: list[dict] = []
            storage = {"hardlink": 0, "copy": 0, "existing": 0}
            for record in records:
                image = str(record["image"])
                fraction = float(record["masked_fraction"])
                over = fraction > settings.mask_threshold
                destination = paths["rejected"] if over else paths["kept"]
                storage[link_or_copy(paths["images"] / image, destination / image)] += 1
                if (paths["masks"] / f"{image}.png").is_file():
                    storage[link_or_copy(paths["masks"] / f"{image}.png", destination / f"{image}.png")] += 1
                row = {
                    "image": image,
                    "masked_fraction": fraction,
                    "detections": record.get("detections"),
                    "classes": record.get("classes"),
                }
                (rejected if over else kept).append(row)
            for directory, desired in (
                (paths["kept"], {str(item["image"]) for item in kept}),
                (paths["rejected"], {str(item["image"]) for item in rejected}),
            ):
                stale = reconcile(directory, desired)
                if stale:
                    log(f"{name}/{label}: removed {stale} stale entries")
            section = {
                "images": len(images),
                "kept": len(kept),
                "rejected": len(rejected),
                "storage": storage,
                "masks_dir": str(paths["masks"]),
            }
            if label == "flat":
                entry.update(section)
                entry["rejected_frames"] = rejected
            else:
                entry["planar"] = {
                    **section,
                    "views": settings.views,
                    "view_fov_degrees": settings.view_fov,
                    "view_size": settings.view_size,
                    "crop_bottom": settings.crop_bottom,
                    "index": str(source_dir(settings, name) / "planar" / "selected.jsonl"),
                    "rejected_frames": rejected,
                }
            log(
                f"{name}/{label}: kept {len(kept)} / rejected {len(rejected)} "
                f"(threshold {settings.mask_threshold:.2%})"
            )
    write_json(summary_path, summary)
    return summary


# ---------------------------------------------------------------------------- main


HANDLERS = {
    "probe": probe_stage,
    "extract": extract_stage,
    "select": select_stage,
    "project": project_stage,
    "mask": mask_stage,
    "filter": filter_stage,
}


def run_select_sharp(
    *,
    sources: list[Path],
    out: Path,
    kind: str | None = None,
    stage: str = "all",
    window: str = "",
    candidate_fps: float = 5.0,
    selected_per_second: int = 1,
    mask_threshold: float = 0.01,
    views: int = DEFAULT_VIEWS,
    view_fov: float = DEFAULT_VIEW_FOV,
    view_size: int = DEFAULT_VIEW_SIZE,
    crop_bottom: float = DEFAULT_CROP_BOTTOM,
    target: str = "auto",
    keep_candidates: bool = False,
    jpeg_quality: int = 95,
    device: str = "cuda",
) -> dict | None:
    if not sources:
        raise ValueError("At least one --source is required")
    if stage not in (*STAGES, "all"):
        raise ValueError(f"Unknown stage: {stage}")
    if kind is not None and kind not in (*EQUIRECT_KINDS, "perspective_video"):
        raise ValueError(f"Unknown source kind: {kind}")
    if target not in ("auto", "flat", "planar", "all"):
        raise ValueError(f"Unknown mask target: {target}")
    if views not in (8, 14):
        raise ValueError("--views must be 8 or 14")
    if not 0.0 < mask_threshold < 1.0:
        raise ValueError("--mask-threshold must be between 0 and 1")
    if candidate_fps <= 0 or selected_per_second < 1:
        raise ValueError("Sampling rates must be positive")
    if not 0.0 <= crop_bottom < 0.5:
        raise ValueError("--crop-bottom must be in [0, 0.5)")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    start_seconds, end_seconds = parse_window(window)

    settings = Settings(
        sources=[Path(item) for item in sources],
        out=Path(out),
        kind=kind,
        stage=stage,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        candidate_fps=float(candidate_fps),
        selected_per_second=int(selected_per_second),
        mask_threshold=float(mask_threshold),
        views=int(views),
        view_fov=float(view_fov),
        view_size=int(view_size),
        crop_bottom=float(crop_bottom),
        target=target,
        keep_candidates=bool(keep_candidates),
        jpeg_quality=int(jpeg_quality),
        device=device,
    )
    started = time.monotonic()
    result = None
    for name in settings.stages:
        log(f"=== stage {name} ===")
        result = HANDLERS[name](settings)
        log(f"=== stage {name} done ({time.monotonic() - started:.0f} s total) ===")
    return result

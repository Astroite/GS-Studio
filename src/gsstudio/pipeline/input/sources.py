from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image, write_cv_image

from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.adapters.media import (
    extract_indexed_frames,
    probe_video,
    sha256_file,
    validate_equirectangular,
    validate_perspective_video,
)
from gsstudio.domain.models import IMAGE_SUFFIXES, CaptureManifest, SourceProbe
from gsstudio.infrastructure.paths import host_path
from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.domain.prepared import CandidateFrameSet
from gsstudio.infrastructure.persistence.prepared_input import _validate_frame, load_prepared_input
from gsstudio.infrastructure.adapters.mediasdk import (
    MediaHelperError, PUBLIC_SUPPORTED_CAMERAS, _camera_key, _helper_visible,
    invoke_media_helper, _atomic_json, _utc_now,
)
























def _sequence_index(path: Path) -> int | None:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else None


def ordered_sequence_files(paths: Iterable[Path], fps: float | None = None) -> list[Path]:
    files = [path for path in paths if path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise RuntimeError("Image sequence contains no supported images")
    indexed = [(_sequence_index(path), path) for path in files]
    if any(index is None for index, _ in indexed):
        if fps is None:
            raise ValueError("Image filenames have no numeric index; provide --fps")
        return sorted(files, key=lambda path: path.name.casefold())
    indices = [int(index) for index, _ in indexed if index is not None]
    if len(set(indices)) != len(indices):
        raise ValueError("Image sequence contains duplicate numeric indices")
    return [path for _, path in sorted(indexed, key=lambda item: (int(item[0]), item[1].name))]


def probe_image_sequence(paths: Iterable[Path], fps: float | None = None) -> dict[str, Any]:
    ordered = ordered_sequence_files(paths, fps=fps)
    dimensions: set[tuple[int, int]] = set()
    for path in ordered:
        image = read_cv_image(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode image sequence member: {path}")
        dimensions.add((int(image.shape[1]), int(image.shape[0])))
    if len(dimensions) != 1:
        raise ValueError(f"Image sequence dimensions are inconsistent: {sorted(dimensions)}")
    width, height = next(iter(dimensions))
    effective_fps = float(fps or 1.0)
    probe = {
        "width": width,
        "height": height,
        "fps": effective_fps,
        "frame_count": len(ordered),
        "duration_seconds": len(ordered) / effective_fps,
        "codec": "image-sequence",
    }
    validate_equirectangular(probe)
    return probe


def source_paths(capture: CaptureManifest) -> list[Path]:
    paths = [host_path(item.windows_path) for item in capture.source.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Capture source files are missing: {missing}")
    return paths


def validate_source_fingerprints(capture: CaptureManifest) -> list[Path]:
    paths = source_paths(capture)
    for record, path in zip(capture.source.files, paths):
        if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
            raise RuntimeError(f"Capture source changed after registration: {path}")
    return paths


class SourceAdapter:
    """Normalize one capture source into a deterministic CandidateFrameSet."""

    kind: str

    def probe(
        self, capture: CaptureManifest, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        raise NotImplementedError

    def export(
        self,
        capture: CaptureManifest,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        raise NotImplementedError


class EquirectVideoAdapter(SourceAdapter):
    kind = "equirect_video"

    def probe(
        self, capture: CaptureManifest, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        probe = probe_video(paths[0])
        validate_equirectangular(probe)
        frame_count = int(
            probe.get("frame_count")
            or round(float(probe["duration_seconds"]) * float(probe["fps"] or 0))
        )
        return {**probe, "frame_count": max(1, frame_count)}

    def export(
        self,
        capture: CaptureManifest,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        valid: set[Path] = set()
        for path in output_paths:
            if not path.is_file():
                continue
            try:
                _validate_frame(path, width, height)
                valid.add(path)
            except RuntimeError:
                pass
        missing = [path for path in output_paths if path not in valid]
        if not missing:
            return
        if valid and not resume:
            raise RuntimeError("Partial video frame extraction exists; use --resume")

        # FFmpeg's legacy extraction command emits a complete, uniformly sampled
        # sequence.  During resume, stage that sequence separately and replace only
        # the absent or invalid destinations so already-verified candidates remain
        # byte-for-byte untouched.
        with tempfile.TemporaryDirectory(prefix=".video-extract-", dir=building) as temporary_dir:
            extracted_dir = Path(temporary_dir) / "frames"
            extract_indexed_frames(
                paths[0],
                extracted_dir,
                indices,
                capture.normalization.jpeg_quality,
                building / "export.log",
            )
            for destination in missing:
                extracted = extracted_dir / destination.name
                if (int(probe["width"]), int(probe["height"])) == (width, height):
                    os.replace(extracted, destination)
                    continue
                normalized = extracted.with_suffix(".normalized.jpg")
                _normalize_image(
                    extracted,
                    normalized,
                    width,
                    height,
                    capture.normalization.jpeg_quality,
                )
                os.replace(normalized, destination)


class PerspectiveVideoAdapter(EquirectVideoAdapter):
    """Landscape perspective video: frames are training views, nothing is projected."""

    kind = "perspective_video"

    def probe(
        self, capture: CaptureManifest, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        probe = probe_video(paths[0])
        validate_perspective_video(probe)
        frame_count = int(
            probe.get("frame_count")
            or round(float(probe["duration_seconds"]) * float(probe["fps"] or 0))
        )
        return {**probe, "frame_count": max(1, frame_count)}


class EquirectSequenceAdapter(SourceAdapter):
    kind = "equirect_sequence"

    def probe(
        self, capture: CaptureManifest, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        return probe_image_sequence(paths, fps=capture.source.probe.fps)

    def export(
        self,
        capture: CaptureManifest,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        ordered = ordered_sequence_files(paths, fps=float(probe["fps"]))
        for source_index, destination in zip(indices, output_paths):
            if destination.is_file():
                try:
                    _validate_frame(destination, width, height)
                    continue
                except RuntimeError:
                    destination.unlink()
            _normalize_image(
                ordered[source_index],
                destination,
                width,
                height,
                capture.normalization.jpeg_quality,
            )


class PerspectiveImagesAdapter(SourceAdapter):
    """Ordered native photos. The external originals are never altered or removed."""
    kind = "perspective_images"

    def probe(self, capture, paths, protocol_dir):
        from PIL import Image, ImageOps
        dimensions = []
        for path in paths:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                dimensions.append(image.size)
        fps = float(capture.source.probe.fps or 1.0)
        return dict(width=dimensions[0][0], height=dimensions[0][1], fps=fps,
                    frame_count=len(paths), duration_seconds=len(paths)/fps,
                    codec="perspective-images")

    def export(self, capture, paths, probe, indices, output_paths, building,
               protocol_dir, width, height, resume):
        from PIL import Image, ImageOps
        for index, destination in zip(indices, output_paths):
            source = paths[index]
            with Image.open(source) as image:
                orientation = image.getexif().get(274, 1)
                image = ImageOps.exif_transpose(image).convert("RGB")
                if destination.is_file():
                    try:
                        _validate_frame(destination, *image.size)
                        continue
                    except RuntimeError:
                        destination.unlink()
                if source.suffix.lower() in {".jpg", ".jpeg"} and orientation == 1:
                    # Copy, not a writable hardlink to immutable user media.
                    shutil.copyfile(source, destination)
                else:
                    image.save(destination, quality=capture.normalization.jpeg_quality,
                               exif=image.getexif().tobytes())


class Insta360InsvAdapter(SourceAdapter):
    kind = "insta360_insv"

    def probe(
        self, capture: CaptureManifest, paths: list[Path], protocol_dir: Path
    ) -> dict[str, Any]:
        response = invoke_media_helper(
            "probe",
            {"source_files": [_helper_visible(path) for path in paths]},
            protocol_dir,
        )
        media = dict(response.get("media") or {})
        camera_model = str(media.get("camera_model") or capture.camera.model).strip()
        supported = {
            _camera_key(str(item))
            for item in response.get("supported_camera_models", [])
        }
        public_supported = {_camera_key(item) for item in PUBLIC_SUPPORTED_CAMERAS}
        declared_key = _camera_key(capture.camera.model)
        detected_key = _camera_key(camera_model)
        if detected_key not in supported or detected_key not in public_supported:
            raise RuntimeError(
                f"MediaSDK unsupported_camera: {camera_model}; use Insta360 Studio to export "
                "a 2:1 equirectangular video"
            )
        if declared_key not in {"", "unknown"} and declared_key != detected_key:
            raise RuntimeError(
                f"MediaSDK camera_mismatch: capture declares {capture.camera.model}, "
                f"but helper detected {camera_model}"
            )
        required = ("width", "height", "fps", "frame_count", "duration_seconds")
        if any(media.get(name) is None for name in required):
            raise RuntimeError(f"MediaSDK probe omitted required fields: {required}")
        media["helper_version"] = response.get("helper_version")
        media["sdk_version"] = response.get("sdk_version")
        media["camera_model"] = camera_model
        return media

    def export(
        self,
        capture: CaptureManifest,
        paths: list[Path],
        probe: dict[str, Any],
        indices: list[int],
        output_paths: list[Path],
        building: Path,
        protocol_dir: Path,
        width: int,
        height: int,
        resume: bool,
    ) -> None:
        missing = []
        requested_outputs: list[Path] = []
        for index, destination in zip(indices, output_paths):
            if destination.is_file():
                try:
                    _validate_frame(destination, width, height)
                    continue
                except RuntimeError:
                    destination.unlink()
            missing.append(
                {
                    "source_frame_index": index,
                    "output_file": _helper_visible(destination),
                }
            )
            requested_outputs.append(destination)
        if not missing:
            return
        response = invoke_media_helper(
            "export_frames",
            {
                "source_files": [_helper_visible(path) for path in paths],
                "frames": missing,
                "output": {
                    "width": width,
                    "height": height,
                    "format": "jpeg",
                    "jpeg_quality": capture.normalization.jpeg_quality,
                },
                "stitching": {
                    "mode": "dynamic",
                    "flowstate": True,
                    "direction_lock": False,
                    "ai_denoise": False,
                    "sharpening": False,
                    "color_processing": False,
                },
            },
            protocol_dir,
            timeout_seconds=3600,
        )
        validation_error: str | None = None
        if int(response.get("exported_count", -1)) != len(missing):
            validation_error = "MediaSDK helper exported an unexpected number of frames"
        elif response.get("helper_version") != probe.get("helper_version"):
            validation_error = "MediaSDK helper version changed between probe and export"
        elif response.get("sdk_version") != probe.get("sdk_version"):
            validation_error = "MediaSDK version changed between probe and export"
        if validation_error is not None:
            # The response cannot prove which bytes were produced under the
            # probed implementation. Never let a later --resume bless them.
            for path in requested_outputs:
                path.unlink(missing_ok=True)
            raise RuntimeError(validation_error)


SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    adapter.kind: adapter
    for adapter in (
        EquirectVideoAdapter(),
        PerspectiveVideoAdapter(),
        PerspectiveImagesAdapter(),
        EquirectSequenceAdapter(),
        Insta360InsvAdapter(),
    )
}


def source_adapter(capture: CaptureManifest) -> SourceAdapter:
    return SOURCE_ADAPTERS[capture.source.kind]


def probe_capture_source(capture: CaptureManifest, protocol_dir: Path) -> dict[str, Any]:
    if capture.source.kind == "insta360_insv":
        with gpu_session():
            return _probe_capture_source(capture, protocol_dir)
    return _probe_capture_source(capture, protocol_dir)


def _probe_capture_source(capture: CaptureManifest, protocol_dir: Path) -> dict[str, Any]:
    paths = validate_source_fingerprints(capture)
    result = source_adapter(capture).probe(capture, paths, protocol_dir)
    validate_source_fingerprints(capture)
    return result


def fixed_rate_frame_indices(
    frame_count: int,
    source_fps: float,
    start_seconds: float,
    end_seconds: float,
    candidate_fps: float,
) -> tuple[list[int], list[float]]:
    """Sample unique source frames at the centres of fixed-rate time cells."""

    if candidate_fps <= 0:
        raise ValueError("candidate_fps must be positive")
    if source_fps + 1e-9 < candidate_fps:
        raise ValueError(
            f"Source frame rate {source_fps:.6g} is below requested candidate rate "
            f"{candidate_fps:.6g}; duplicate source frames are not allowed"
        )
    if end_seconds <= start_seconds:
        raise ValueError("Selection end must be after its start")
    step = 1.0 / candidate_fps
    requested: list[float] = []
    sample = 0
    while True:
        timestamp = start_seconds + (sample + 0.5) * step
        if timestamp >= end_seconds - 1e-12:
            break
        requested.append(timestamp)
        sample += 1
    indices = [
        # Nearest source frame, resolving exact half-frame ties toward the
        # earlier timestamp so a source running exactly at candidate_fps keeps
        # each sample inside its intended one-second bucket.
        min(
            frame_count - 1,
            max(0, int(math.floor(timestamp * source_fps + 0.5 - 1e-12))),
        )
        for timestamp in requested
    ]
    if len(indices) < 2:
        raise ValueError("Selection contains fewer than two fixed-rate candidate frames")
    if len(set(indices)) != len(indices):
        raise ValueError(
            "Fixed-rate sampling mapped multiple timestamps to the same source frame"
        )
    return indices, [index / source_fps for index in indices]




def anchored_frame_indices(frame_count: int, source_fps: float, start: float,
                           end: float, rate: float) -> tuple[list[int], list[float]]:
    """Nested grids anchored at Capture start; unlike historical cell centres."""
    if not all(math.isfinite(v) for v in (source_fps, start, end, rate)) or not 0 < rate <= source_fps or end <= start:
        raise ValueError('Invalid anchored sampling range/rate')
    indices = sorted({min(frame_count - 1, max(math.ceil(start * source_fps),
        int(math.floor((start + i / rate) * source_fps + 0.5 - 1e-12))))
        for i in range(math.ceil((end - start) * rate))
        if start + i / rate < end - 1e-12})
    indices = [i for i in indices if 0 <= i < frame_count and i / source_fps < end]
    return indices, [i / source_fps for i in indices]


def _normalize_image(source: Path, destination: Path, width: int, height: int, quality: int) -> None:
    image = read_cv_image(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode source image: {source}")
    source_height, source_width = image.shape[:2]
    if width > source_width or height > source_height:
        raise ValueError(
            f"Normalization may not upscale {source_width}x{source_height} to {width}x{height}"
        )
    if (source_width, source_height) != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not write_cv_image(
        str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ):
        raise RuntimeError(f"Cannot write normalized image: {destination}")




def _trusted_cached_frame_hashes(path: Path, preparation_hash: str) -> dict[str, str]:
    """Read expected frame hashes without accepting any current frame bytes."""

    manifest_path = path / "dataset.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        lineage = payload["lineage"]
        records = payload["frames"]
        if (
            payload.get("schema_version") != 2
            or payload.get("integrity") != "complete"
            or payload.get("preparation_hash") != preparation_hash
            or payload.get("dataset_sha256") != canonical_hash(lineage)
            or records != lineage.get("frames")
        ):
            return {}
        trusted: dict[str, str] = {}
        for record in records:
            relative = Path(str(record["file"]))
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
                return {}
            digest = str(record["sha256"])
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                return {}
            trusted[relative.as_posix()] = digest
        return trusted
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def prepare_capture_input(
    scene_path: Path,
    capture: CaptureManifest,
    target_root: Path,
    candidate_fps: float,
    selection_end_seconds: float | None = None,
    resume: bool = False,
) -> CandidateFrameSet:
    """Prepare CPU media directly; serialize MediaSDK's GPU work with all GPU stages."""
    if capture.source.kind == "insta360_insv":
        with gpu_session():
            return _prepare_capture_input(scene_path, capture, target_root, candidate_fps,
                                          selection_end_seconds, resume)
    return _prepare_capture_input(scene_path, capture, target_root, candidate_fps,
                                  selection_end_seconds, resume)


def _prepare_capture_input(
    scene_path: Path,
    capture: CaptureManifest,
    target_root: Path,
    candidate_fps: float,
    selection_end_seconds: float | None = None,
    resume: bool = False,
) -> CandidateFrameSet:
    """Extract rate-sampled candidate frames from the registered sources.

    Sampling is anchored to the capture start so a given second always yields the
    same source frame. The verified set lands at ``target_root/<preparation hash>``,
    which the Run owns.
    """
    protocol_dir = target_root / ".protocol"
    probe = probe_capture_source(capture, protocol_dir)
    paths = source_paths(capture)
    effective_end = min(
        capture.selection.end_seconds,
        selection_end_seconds
        if selection_end_seconds is not None
        else capture.selection.end_seconds,
    )
    if effective_end > float(probe["duration_seconds"]) + 0.5:
        raise ValueError(
            f"Selected range ends at {effective_end:.3f}s but source is "
            f"{float(probe['duration_seconds']):.3f}s"
        )
    indices, timestamps = anchored_frame_indices(
        int(probe["frame_count"]),
        float(probe["fps"]),
        capture.selection.start_seconds,
        effective_end,
        candidate_fps,
    )
    if len(indices) < 2:
        raise ValueError("At least two candidate frames are required")
    if capture.source.is_perspective:
        width, height = int(probe["width"]), int(probe["height"])
    else:
        width = int(capture.normalization.width or probe["width"])
        height = int(capture.normalization.height or probe["height"])
        if width != height * 2:
            raise ValueError(f"Prepared panorama must be 2:1, got {width}x{height}")
        if width > int(probe["width"]) or height > int(probe["height"]):
            raise ValueError("Normalization dimensions may not exceed source panorama dimensions")
    preparation = {
        "capture_id": capture.id,
        "source_kind": capture.source.kind,
        "source_files": [item.model_dump(mode="json") for item in capture.source.files],
        "selection": {
            "start_seconds": capture.selection.start_seconds,
            "end_seconds": effective_end,
        },
        "normalization": capture.normalization.model_dump(mode="json"),
        "source_probe": SourceProbe(
            **{
                name: value
                for name, value in probe.items()
                if name in SourceProbe.model_fields
            }
        ).model_dump(mode="json"),
        "source_frame_indices": indices,
        "helper_version": probe.get("helper_version"),
        "sdk_version": probe.get("sdk_version"),
        "sampling": {
            "mode": "anchored_rate_v1",
            "candidate_fps": candidate_fps,
        },
        "candidate_frame_count": len(indices),
    }
    preparation_hash = canonical_hash(preparation)
    target = target_root / preparation_hash
    if (target / "dataset.json").is_file():
        try:
            return load_prepared_input(target)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            if not resume:
                raise RuntimeError(
                    f"Prepared input cache is invalid; use --resume to repair it: {target}"
                ) from error
    required_bytes = len(indices) * width * height + (1 << 30)
    disk_path = scene_path
    while not disk_path.exists() and disk_path.parent != disk_path:
        disk_path = disk_path.parent
    free_bytes = shutil.disk_usage(disk_path).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Insufficient Windows-project disk space for prepared input: "
            f"{free_bytes / 1024**3:.1f} GiB free, "
            f"{required_bytes / 1024**3:.1f} GiB conservatively required"
        )
    building = target.with_name(target.name + ".building")
    if target.exists():
        if not resume:
            raise RuntimeError(f"Partial prepared input exists; use --resume: {target}")
        if building.exists():
            raise RuntimeError(
                f"Both invalid prepared input and partial repair exist; inspect them before resuming: "
                f"{target}, {building}"
            )
        target.replace(building)
    if building.exists() and not resume:
        raise RuntimeError(f"Partial prepared input exists; use --resume: {building}")
    frames_dir = building / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [frames_dir / f"frame_{index + 1:06d}.jpg" for index in range(len(indices))]

    # A completed cache carries trusted hashes.  On resume, invalidate only bytes
    # that no longer match that lineage; adapters then regenerate those candidates.
    if (building / "dataset.json").is_file():
        trusted_hashes = _trusted_cached_frame_hashes(building, preparation_hash)
        for output_path in output_paths:
            relative = output_path.relative_to(building).as_posix()
            if (
                output_path.is_file()
                and trusted_hashes.get(relative) != sha256_file(output_path)
            ):
                output_path.unlink()

    try:
        source_adapter(capture).export(
            capture,
            paths,
            probe,
            indices,
            output_paths,
            building,
            protocol_dir,
            width,
            height,
            resume,
        )
        # A native helper is an external process. Re-hash the registered source
        # before publishing the cache so even an accidental write to an INSV is
        # detected and can never enter prepared-input lineage unnoticed.
        validate_source_fingerprints(capture)

        frame_records = []
        for index, timestamp, path in zip(indices, timestamps, output_paths):
            frame_width, frame_height = width, height
            if capture.source.kind == "perspective_images":
                from PIL import Image
                with Image.open(path) as photo:
                    frame_width, frame_height = photo.size
            frame_records.append(
                {
                    "file": path.relative_to(building).as_posix(),
                    "source_frame_index": index,
                    "timestamp_seconds": round(timestamp, 6),
                    "width": frame_width,
                    "height": frame_height,
                    "sha256": _validate_frame(path, frame_width, frame_height),
                }
            )
    except Exception as error:
        # Only an explicit helper interruption is trusted as resumable. Protocol
        # drift, malformed media, source mutation, and generic adapter failures
        # must not leave bytes that a later --resume could silently bless.
        if not isinstance(error, MediaHelperError) or error.code != "interrupted_export":
            shutil.rmtree(building, ignore_errors=True)
        raise
    lineage = {**preparation, "frames": frame_records}
    dataset_sha256 = canonical_hash(lineage)
    manifest = {
        "schema_version": 2,
        "generated_at": _utc_now(),
        "preparation_hash": preparation_hash,
        "dataset_sha256": dataset_sha256,
        "lineage": lineage,
        "frames": frame_records,
        "integrity": "complete",
    }
    _atomic_json(building / "dataset.json", manifest)
    load_prepared_input(building)
    target.parent.mkdir(parents=True, exist_ok=True)
    building.replace(target)
    return load_prepared_input(target)

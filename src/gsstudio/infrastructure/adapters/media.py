from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image
import numpy as np

from gsstudio.infrastructure.runtime.processes import run_logged

# FFmpeg parses a select chain into a fixed-size expression stack and rejects the
# 101st eq() term. The failure surfaces only as "Cannot allocate memory" while
# opening the output, so a single oversized chain looks like a disk problem. The
# boundary is exactly 100 terms on ffmpeg 9.0.1. Requests are split into chunks of
# at most this many terms; -frames:v ends each pass at its own last selected frame,
# which keeps the repeated decoding bounded by the chunk's own position in the clip.
MAX_SELECT_TERMS = 100


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _rational(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(result.stdout)
    video_streams = [
        item
        for item in payload.get("streams", [])
        if item.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ValueError("Source contains no video stream")
    # Camera files may carry extra low-resolution video tracks (previews or
    # thumbnails); the primary stream is the largest one.
    stream = max(
        video_streams,
        key=lambda item: (int(item.get("width") or 0), int(item.get("height") or 0)),
    )
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    frame_count = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": _rational(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "frame_count": int(frame_count) if frame_count and str(frame_count).isdigit() else None,
        "duration_seconds": float(duration),
        "codec": str(stream.get("codec_name") or stream.get("codec_tag_string") or "unknown"),
        "pixel_format": stream.get("pix_fmt"),
        "byte_size": path.stat().st_size,
    }


def validate_equirectangular(probe: dict[str, Any], tolerance: float = 0.01) -> None:
    width = int(probe["width"])
    height = int(probe["height"])
    if height <= 0 or abs(width / height - 2.0) > tolerance:
        raise ValueError(f"Expected a 2:1 equirectangular video, got {width}x{height}")
    if float(probe["duration_seconds"]) <= 0:
        raise ValueError("Video duration must be positive")


def validate_perspective_video(probe: dict[str, Any]) -> None:
    width = int(probe["width"])
    height = int(probe["height"])
    if width <= height:
        raise ValueError(f"Expected a landscape perspective video, got {width}x{height}")
    if float(probe["duration_seconds"]) <= 0 or not float(probe.get("fps") or 0) > 0:
        raise ValueError("Video duration and frame rate must be positive")


def required_free_bytes(video_size: int, reserve_gib: float) -> int:
    reserve = int(reserve_gib * 1024**3)
    estimated_work = max(video_size * 6, 20 * 1024**3)
    return reserve + estimated_work


def check_disk_budget(path: Path, video_size: int, reserve_gib: float) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    required = required_free_bytes(video_size, reserve_gib)
    if usage.free < required:
        raise RuntimeError(
            f"Insufficient disk space: {usage.free / 1024**3:.1f} GiB free, "
            f"{required / 1024**3:.1f} GiB required by the pilot estimate"
        )
    return {"free_bytes": usage.free, "required_bytes": required}


def extract_indexed_frames(
    video_path: Path,
    output_dir: Path,
    frame_indices: list[int],
    jpeg_quality: int,
    log_path: Path,
) -> list[Path]:
    """Decode the exact zero-based source frames, in chunks FFmpeg can parse."""

    if not frame_indices or frame_indices != sorted(set(frame_indices)):
        raise ValueError("Frame indices must be a non-empty, strictly increasing list")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        if len(existing) != len(frame_indices):
            raise RuntimeError(
                f"Found {len(existing)} existing frames but expected {len(frame_indices)}"
            )
        return existing
    for offset in range(0, len(frame_indices), MAX_SELECT_TERMS):
        chunk = frame_indices[offset : offset + MAX_SELECT_TERMS]
        selection = "+".join(f"eq(n\\,{index})" for index in chunk)
        # A chunk that fails aborts the loop, so the shared log always ends up
        # holding the run that failed even though run_logged truncates it.
        run_logged(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-vf",
                f"select={selection}",
                "-fps_mode",
                "vfr",
                "-frames:v",
                str(len(chunk)),
                "-start_number",
                str(offset + 1),
                "-q:v",
                str(max(1, min(31, round((101 - jpeg_quality) * 31 / 100)))),
                "-pix_fmt",
                "yuvj420p",
                str(output_dir / "frame_%06d.jpg"),
            ],
            log_path,
        )
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(frames) != len(frame_indices):
        raise RuntimeError(
            f"FFmpeg produced {len(frames)} indexed frames; expected {len(frame_indices)}"
        )
    return frames


def _frame_statistics(path: Path, timestamp: float) -> dict[str, Any]:
    image = read_cv_image(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {path}")
    height, width = image.shape[:2]
    scale = min(1.0, 1024.0 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "file": path.name,
        "timestamp_seconds": round(timestamp, 6),
        "width": width,
        "height": height,
        "blur_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "tenengrad": float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y)),
        "luma_mean": float(gray.mean()),
        "black_fraction": float(np.mean(gray <= 8)),
        "highlight_fraction": float(np.mean(gray >= 247)),
    }


def add_selection_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach deterministic 70/15/15 quality scores without discarding legacy metrics."""

    if not records:
        return records
    sharpness = np.asarray([float(item["tenengrad"]) for item in records])
    low = float(np.min(sharpness))
    high = float(np.max(sharpness))
    span = high - low
    for item in records:
        if span > 0:
            normalized_sharpness = (float(item["tenengrad"]) - low) / span
            sharpness_metric = "tenengrad"
        else:
            normalized_sharpness = 1.0
            sharpness_metric = "tenengrad-degenerate-equal"
        black_score = 1.0 - float(item["black_fraction"])
        highlight_score = 1.0 - float(item["highlight_fraction"])
        score = (
            0.70 * normalized_sharpness
            + 0.15 * black_score
            + 0.15 * highlight_score
        )
        item["selection_score"] = round(float(score), 12)
        item["selection_reason"] = {
            "method": "tenengrad-exposure-v1",
            "sharpness_metric": sharpness_metric,
            "tenengrad_weight": 0.70,
            "black_fraction_inverse_weight": 0.15,
            "highlight_fraction_inverse_weight": 0.15,
            "tie_break": "earlier_timestamp",
        }
    return records


def analyze_frames(
    frames: list[Path],
    duration_seconds: float,
    output_jsonl: Path,
    start_seconds: float = 0.0,
    timestamps_seconds: Sequence[float] | None = None,
    selection_algorithm: str = "composite_v1",
) -> list[dict[str, Any]]:
    if selection_algorithm != "composite_v1":
        raise ValueError(f"Unknown frame selection algorithm: {selection_algorithm}")
    if timestamps_seconds is not None and len(timestamps_seconds) != len(frames):
        raise ValueError("Explicit timestamps must match the candidate-frame count")
    expected_timestamps = (
        [float(value) for value in timestamps_seconds]
        if timestamps_seconds is not None
        else [
            start_seconds + (index + 0.5) * duration_seconds / len(frames)
            for index in range(len(frames))
        ]
    )
    if output_jsonl.is_file():
        records = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line]
        required = ("tenengrad", "selection_score")
        if len(records) == len(frames) and all(
            all(field in item for field in required) for item in records
        ) and all(
            abs(float(item["timestamp_seconds"]) - expected) < 1e-6
            for item, expected in zip(records, expected_timestamps)
        ):
            return records
    raw_records = [
        _frame_statistics(frame, expected_timestamps[index])
        for index, frame in enumerate(frames)
    ]
    records = add_selection_scores(raw_records)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    return records


def summarize_frame_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    def median(field: str) -> float:
        return float(np.median([float(item[field]) for item in records]))

    summary = {
        "frame_count": len(records),
        "blur_median": median("blur_laplacian_variance"),
        "luma_mean_median": median("luma_mean"),
        "black_fraction_median": median("black_fraction"),
        "highlight_fraction_median": median("highlight_fraction"),
    }
    if all("tenengrad" in item for item in records):
        summary["tenengrad_median"] = median("tenengrad")
    if all("selection_score" in item for item in records):
        summary["selection_score_median"] = median("selection_score")
    return summary


def select_temporal_records(
    metrics: list[dict[str, Any]],
    start_seconds: float,
    selected_per_second: int,
    selection_metric: str = "selection_score",
) -> list[dict[str, Any]]:
    """Keep the best N candidates in each one-second bucket."""

    if selected_per_second < 1:
        raise ValueError("selected_per_second must be positive")
    buckets: dict[int, list[dict[str, Any]]] = {}
    for item in metrics:
        timestamp = float(item["timestamp_seconds"])
        bucket = max(0, int(math.floor(timestamp - start_seconds + 1e-9)))
        buckets.setdefault(bucket, []).append(item)
    selected: list[dict[str, Any]] = []
    for bucket_index in sorted(buckets):
        ranked = sorted(
            buckets[bucket_index],
            key=lambda item: (
                -float(item[selection_metric]),
                float(item.get("timestamp_seconds", 0.0)),
            ),
        )
        for rank, item in enumerate(ranked[:selected_per_second], start=1):
            selected.append(
                {
                    **item,
                    "time_bucket": bucket_index,
                    "temporal_rank": rank,
                }
            )
    return sorted(selected, key=lambda item: float(item["timestamp_seconds"]))


def create_temporal_subset(
    source_dir: Path,
    metrics: list[dict[str, Any]],
    output_dir: Path,
    start_seconds: float,
    selected_per_second: int,
    rank_limit: int | None = None,
    selection_metric: str = "selection_score",
) -> tuple[list[Path], list[dict[str, Any]]]:
    selected = select_temporal_records(
        metrics,
        start_seconds,
        selected_per_second,
        selection_metric=selection_metric,
    )
    if rank_limit is not None:
        selected = [
            item for item in selected if int(item["temporal_rank"]) <= rank_limit
        ]
    if not selected:
        raise ValueError("Temporal selection contains no frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and len(existing) != len(selected):
        raise RuntimeError("Partial temporal frame directory exists; create a new run")
    outputs: list[Path] = []
    rewritten: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        source = source_dir / str(item["file"])
        destination = output_dir / f"frame_{index:06d}.jpg"
        if not destination.is_file():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        outputs.append(destination)
        rewritten.append(
            {
                **item,
                "source_file": item["file"],
                "file": destination.name,
            }
        )
    return outputs, rewritten

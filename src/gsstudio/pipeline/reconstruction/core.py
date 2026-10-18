"""Equirectangular view projection and the COLMAP/Nerfstudio hand-off helpers.

RealityScan writes one COLMAP model per attempt under ``colmap/``; this module
supplies the projection, the model readers and the per-frame mask attachment it
needs.
"""
from __future__ import annotations

import json
import math
import struct
import tempfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image

from gsstudio.pipeline.masks.masking import (
    atomic_imwrite,
    image_dimensions,
    image_files,
    io_worker_count,
    run_image_tasks,
)
from gsstudio.domain.models import ReconstructionAttempt, ReconstructionConfig


AttemptConfig = ReconstructionAttempt
ReconstructionSettings = ReconstructionConfig

class _WriteQueue:
    """Bounded background encode/write queue that keeps the GPU loop moving.

    Each pending 2048x2048 BGR frame holds about 12 MiB, so the queue is capped
    rather than unbounded.
    """

    def __init__(self) -> None:
        self._workers = io_worker_count()
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pending: deque[Future[Any]] = deque()
        self._limit = max(2, self._workers * 4)

    def submit(self, action: Callable[..., Any], *arguments: Any) -> None:
        while len(self._pending) >= self._limit:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(action, *arguments))

    def drain(self) -> None:
        while self._pending:
            self._pending.popleft().result()

    def __enter__(self) -> "_WriteQueue":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: Any) -> None:
        try:
            if exc_type is None:
                self.drain()
            else:
                # Let the original failure surface instead of a queued write's error.
                for future in self._pending:
                    future.cancel()
        finally:
            self._pool.shutdown(wait=True)


def expected_planar_images(attempt: AttemptConfig, frame_count: int) -> int:
    """Training views written for ``frame_count`` prepared frames."""
    if attempt.panorama is None:
        return int(frame_count)
    return int(frame_count) * attempt.panorama.views


def projection_fov_degrees(attempt: AttemptConfig) -> float:
    return float(attempt.projection_fov_degrees)


def _crop_vertical_bounds(
    bounds: list[float], fov_degrees: float, crop_bottom: float
) -> list[float | None]:
    """Mirror Nerfstudio 1.1.5's bottom-crop pitch adjustment without private imports."""
    adjusted: list[float | None] = list(bounds)
    if crop_bottom <= 0:
        return adjusted
    degrees_chopped = 180.0 * crop_bottom
    new_top_start = -90.0 + degrees_chopped + fov_degrees / 2.0
    for index, value in enumerate(adjusted):
        assert value is not None
        if value < new_top_start - fov_degrees / 2.0:
            adjusted[index] = None
        elif value < new_top_start:
            difference = new_top_start - value
            adjusted[index] = new_top_start
            for later in range(index + 1, len(adjusted)):
                assert adjusted[later] is not None
                adjusted[later] += difference / (2 ** (later - index))
            break
    return adjusted


def projection_view_specs(attempt: AttemptConfig) -> list[tuple[float, float]]:
    """Return deterministic ``(yaw, pitch)`` pairs in Nerfstudio's view order."""
    panorama = attempt.panorama
    if panorama is None:
        raise ValueError("A perspective source has no projected view layout")
    bounds = _crop_vertical_bounds(
        [-45.0, 0.0, 45.0], projection_fov_degrees(attempt), panorama.crop_bottom
    )
    middle_step, outer_step = (90, 180) if panorama.views == 8 else (60, 90)
    pairs: list[tuple[float, float]] = []
    for bound_index, step in ((1, middle_step), (2, outer_step), (0, outer_step)):
        pitch = bounds[bound_index]
        if pitch is not None:
            pairs.extend((float(yaw), float(pitch)) for yaw in range(-180, 180, step))
    if len(pairs) != panorama.views:
        raise RuntimeError(
            f"Projection crop leaves {len(pairs)} views; expected {panorama.views}"
        )
    return pairs


def _angular_separation_degrees(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle angle between two (yaw, pitch) optical axes, in degrees."""
    import math

    def unit(yaw: float, pitch: float) -> tuple[float, float, float]:
        cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        return (sy * cp, sp, cy * cp)

    ax, ay, az = unit(*a)
    bx, by, bz = unit(*b)
    dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
    return math.degrees(math.acos(dot))


def projection_overlap_summary(attempt: AttemptConfig) -> dict:
    """Report adjacent-view angular overlap for the 360→pinhole split.

    Overlap uses great-circle axis separation (not raw yaw delta): at pitch
    ``φ`` a yaw step ``Δλ`` separates axes by ``arccos(sin²φ + cos²φ·cosΔλ)``.
    Two views share ``fov − separation`` degrees of solid coverage along the
    arc between them. Only *adjacent* pairs matter for cross-positioning;
    antipodal views on a ring are not required to overlap.
    """
    specs = projection_view_specs(attempt)
    fov = projection_fov_degrees(attempt)
    by_pitch: dict[float, list[tuple[float, float]]] = {}
    for view in specs:
        by_pitch.setdefault(round(view[1], 6), []).append(view)
    adjacent: list[tuple[float, float, tuple[float, float], tuple[float, float], str]] = []
    for pitch, group in sorted(by_pitch.items()):
        group = sorted(group, key=lambda item: item[0])
        for index in range(len(group)):
            a = group[index]
            b = group[(index + 1) % len(group)] if len(group) > 1 else group[index]
            if a is b:
                continue
            sep = _angular_separation_degrees(a, b)
            adjacent.append((fov - sep, sep, a, b, f'yaw@pitch={pitch:g}'))
    # Nearest different-pitch neighbor for each view (undirected edges).
    pitch_edges: dict[tuple[tuple[float, float], tuple[float, float]], tuple[float, float, str]] = {}
    for a in specs:
        candidates = [b for b in specs if abs(a[1] - b[1]) > 1e-6]
        if not candidates:
            continue
        b = min(candidates, key=lambda other: (_angular_separation_degrees(a, other), other))
        key = tuple(sorted([a, b]))
        sep = _angular_separation_degrees(a, b)
        pitch_edges[key] = (fov - sep, sep, 'pitch')
    for (a, b), (overlap, sep, kind) in pitch_edges.items():
        adjacent.append((overlap, sep, a, b, kind))
    adjacent.sort(key=lambda item: item[0])
    min_overlap, min_sep, min_a, min_b, min_kind = adjacent[0]
    yaw_only = [item[0] for item in adjacent if item[4].startswith('yaw@')]
    min_yaw = min(yaw_only) if yaw_only else None
    return dict(
        views=int(attempt.panorama.views),
        fov_degrees=fov,
        view_count=len(specs),
        adjacent_pairs=len(adjacent),
        min_adjacent_overlap_degrees=float(min_overlap),
        min_adjacent_separation_degrees=float(min_sep),
        min_adjacent_yaw_overlap_degrees=float(min_yaw) if min_yaw is not None else None,
        worst_pair=dict(overlap_kind=min_kind, a=list(min_a), b=list(min_b)),
        meets_cross_view_rule=bool(min_overlap >= 15.0),
        rule='相邻视图光轴角距 < FOV 且 FOV − 角距 ≥ 15°（球面角距，只计环内相邻与最近跨环）',
    )


def validate_existing_projection_set(
    target: Path,
    frame_count: int,
    view_count: int,
    projection_size: int,
) -> set[Path]:
    """Validate and return reusable relative paths from a partial v2 projection."""
    if not target.exists():
        return set()
    if not target.is_dir():
        raise RuntimeError(f"Planar image target is not a directory: {target}")
    expected = {
        Path(f"view_{view_index:02d}") / f"frame_{frame_index:06d}.jpg"
        for frame_index in range(1, frame_count + 1)
        for view_index in range(view_count)
    }
    actual = {path.relative_to(target) for path in image_files(target)}
    unexpected = sorted(actual - expected)
    if unexpected:
        raise RuntimeError(
            "Existing planar projection contains unexpected images: "
            + ", ".join(path.as_posix() for path in unexpected[:5])
        )

    def check(relative: Path) -> None:
        try:
            shape = image_dimensions(target / relative)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Existing planar projection is invalid: {relative.as_posix()}; {error}"
            ) from error
        if shape != (projection_size, projection_size):
            raise RuntimeError(
                f"Existing planar projection is invalid: {relative.as_posix()}; "
                f"expected {projection_size}x{projection_size}"
            )

    run_image_tasks(check, sorted(actual))
    return actual


def project_equirectangular_frames(
    source: Path,
    dataset: Path,
    attempt: AttemptConfig,
) -> list[Path]:
    if attempt.panorama is None:
        raise ValueError("A perspective source is never projected; stage the frames instead")
    frames = image_files(source)
    frame_count = len(frames)
    expected = expected_planar_images(attempt, frame_count)
    target = dataset / "images"
    if target.is_dir():
        existing = image_files(target)
        if len(existing) == expected:
            return existing
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"Planar image target is not a directory: {target}")

    import torch
    from gsstudio._vendor.nerfstudio.equirect_utils import equirect2persp

    device = torch.device("cuda")
    size = attempt.panorama.size
    specs = projection_view_specs(attempt)
    reusable = validate_existing_projection_set(
        target, frame_count, len(specs), size
    )
    target.mkdir(parents=True, exist_ok=True)
    for view_index in range(len(specs)):
        (target / f"view_{view_index:02d}").mkdir(parents=True, exist_ok=True)

    def decode(frame_path: Path) -> np.ndarray:
        image = read_cv_image(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode equirectangular frame: {frame_path}")
        return image

    with ThreadPoolExecutor(max_workers=1) as reader, _WriteQueue() as writes:
        prefetched: Future[np.ndarray] | None = None
        for frame_index, frame in enumerate(frames, start=1):
            frame_targets = [
                Path(f"view_{view_index:02d}") / f"frame_{frame_index:06d}.jpg"
                for view_index in range(len(specs))
            ]
            if all(relative in reusable for relative in frame_targets):
                print(
                    f"Planar projection: {frame_index}/{len(frames)} (reused)",
                    flush=True,
                )
                continue
            image = prefetched.result() if prefetched is not None else decode(frame)
            prefetched = None
            # Overlap the next equirectangular decode with this frame's GPU work.
            for later_index in range(frame_index, len(frames)):
                later_targets = [
                    Path(f"view_{view_index:02d}") / f"frame_{later_index + 1:06d}.jpg"
                    for view_index in range(len(specs))
                ]
                if not all(relative in reusable for relative in later_targets):
                    prefetched = reader.submit(decode, frames[later_index])
                    break
            tensor = torch.tensor(image, dtype=torch.float32, device=device)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            for view_index, (yaw, pitch) in enumerate(specs):
                relative = frame_targets[view_index]
                if relative in reusable:
                    continue
                perspective = equirect2persp(
                    tensor,
                    attempt.projection_fov_degrees,
                    yaw,
                    pitch,
                    size,
                    size,
                )
                output = (
                    (perspective * 255.0)
                    .clamp(0, 255)
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                writes.submit(
                    atomic_imwrite,
                    target / relative,
                    output,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
            print(f"Planar projection: {frame_index}/{len(frames)}", flush=True)
    produced = image_files(target)
    if len(produced) != expected:
        raise RuntimeError(f"Projection produced {len(produced)} images; expected {expected}")
    return image_files(target)



# COLMAP 3.8 ships ``ba_global_function_tolerance`` at 0, which switches off Ceres'
# cost-based convergence test, so every global bundle adjustment runs the full 50
# iterations and reports ``No convergence``. On the 5,128-image Yunxiu run that was
# 6.5 of the 8 hours the mapper survived, and the extra iterations bought nothing
# measurable: its last global BA spent 4.2 minutes moving the cost by 0.01%. Replaying
# that log's Ceres tables against a 1e-5 tolerance puts the same 48 expensive passes at
# 1.7 hours instead of 6.5. The trigger knobs matter as much as the tolerance -- at
# ``images_ratio`` 1.1 the mapper called global BA 166 times for 2,755 registrations,
# three or four of them per trigger because ``max_refinements`` is 5 and the 0.0005
# change threshold is essentially never met. ``images_freq``/``points_freq`` are a
# second, independent trigger and have to move with the ratio, or past roughly 1,700
# registered images they become the binding one and the ratio change does nothing.
#
# Local BA deliberately keeps COLMAP's defaults: it was 7.8% of BA time at 0.24 s per
# call, and it is what keeps incremental registration honest.


def excluded_images(dataset: Path) -> set[str]:
    path = dataset / "mask-final.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item) for item in payload.get("excluded_images", [])}




def _rodrigues(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = math.radians(angle_degrees)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def projection_world_from_camera(yaw: float, pitch: float) -> np.ndarray:
    """Rotation from COLMAP camera axes to the equirectangular world axes."""
    yaw_rotation = _rodrigues(np.array((0.0, 0.0, 1.0)), yaw)
    pitch_axis = yaw_rotation @ np.array((0.0, 1.0, 0.0))
    pitch_rotation = _rodrigues(pitch_axis, -pitch)
    # Nerfstudio rays are [forward, right, up]; COLMAP uses [right, down, forward].
    nerfstudio_from_colmap = np.array(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    )
    return pitch_rotation @ yaw_rotation @ nerfstudio_from_colmap




def _write_colmap_binary_model(
    output: Path,
    cameras: dict[int, Any],
    images: dict[int, Any],
    points: dict[int, Any],
) -> None:
    """Write the COLMAP 3.8 binary subset used by the rig merge."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(cameras)))
        for camera_id, camera in sorted(cameras.items()):
            if camera.model != "PINHOLE" or len(camera.params) != 4:
                raise RuntimeError(
                    f"Rig merge supports only four-parameter PINHOLE cameras, got {camera.model}"
                )
            stream.write(
                struct.pack(
                    "<iiQQ4d",
                    int(camera_id),
                    1,
                    int(camera.width),
                    int(camera.height),
                    *(float(value) for value in camera.params),
                )
            )
    with (output / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(images)))
        for image_id, image in sorted(images.items()):
            stream.write(
                struct.pack(
                    "<i7di",
                    int(image_id),
                    *(float(value) for value in image.qvec),
                    *(float(value) for value in image.tvec),
                    int(image.camera_id),
                )
            )
            stream.write(image.name.encode("utf-8") + b"\0")
            stream.write(struct.pack("<Q", len(image.xys)))
            for xy, point_id in zip(image.xys, image.point3D_ids):
                stream.write(
                    struct.pack(
                        "<ddq", float(xy[0]), float(xy[1]), int(point_id)
                    )
                )
    with (output / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_id, point in sorted(points.items()):
            error = float(np.asarray(point.error).reshape(-1)[0])
            stream.write(
                struct.pack(
                    "<Q3d3BdQ",
                    int(point_id),
                    *(float(value) for value in point.xyz),
                    *(int(value) for value in point.rgb),
                    error,
                    len(point.image_ids),
                )
            )
            for image_id, point2d_index in zip(point.image_ids, point.point2D_idxs):
                stream.write(struct.pack("<ii", int(image_id), int(point2d_index)))




def _selected_model_dir(dataset: Path) -> Path:
    # RealityScan (the live backend) always writes one model directly into
    # colmap/, recorded as model="." -- no attempt/component indirection like the
    # dead COLMAP-mapper cluster below, which still writes its own
    # colmap/attempt-NNN/sparse/N convention and is unreachable from the CLI.
    selection = dataset / "colmap" / "selected-attempt.json"
    payload = json.loads(selection.read_text(encoding="utf-8"))
    model = dataset / "colmap" / str(payload["model"])
    required = [model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Selected COLMAP model is incomplete: {model}")
    return model


def _relative_image_path(file_path: str) -> Path:
    normalized = file_path.replace("\\", "/").removeprefix("./")
    if normalized.startswith("images/"):
        normalized = normalized.removeprefix("images/")
    return Path(normalized)


def validate_frame_masks(transforms_path: Path, masks_dir: Path) -> int:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_relative = _relative_image_path(str(frame["file_path"]))
        mask_relative = image_relative.parent / f"{image_relative.name}.png"
        mask = masks_dir / mask_relative
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_relative.as_posix()}")
        expected = f"./masks/{mask_relative.as_posix()}"
        if frame.get("mask_path") != expected:
            raise RuntimeError(
                f"Registered image has an invalid mask_path: {image_relative.as_posix()}"
            )
    return len(frames)


def attach_frame_masks(transforms_path: Path, masks_dir: Path) -> None:
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    if not frames:
        raise RuntimeError(f"No registered frames in {transforms_path}")
    for frame in frames:
        image_relative = _relative_image_path(str(frame["file_path"]))
        mask_relative = image_relative.parent / f"{image_relative.name}.png"
        mask = masks_dir / mask_relative
        if not mask.is_file():
            raise RuntimeError(f"Registered image has no person mask: {image_relative.as_posix()}")
        frame["mask_path"] = f"./masks/{mask_relative.as_posix()}"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=transforms_path.parent,
        prefix=f".{transforms_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(transforms_path)
    validate_frame_masks(transforms_path, masks_dir)




def _recorded_cross_view_pairs(dataset: Path) -> int:
    """Pairs imported by the within-frame view matcher for the selected attempt."""
    selection_path = dataset / "colmap" / "selected-attempt.json"
    attempt_name = None
    if selection_path.is_file():
        attempt_name = json.loads(selection_path.read_text(encoding="utf-8")).get("attempt")
    candidates = (
        [dataset / "colmap" / str(attempt_name)]
        if attempt_name
        else sorted((dataset / "colmap").glob("attempt-*"))
    )
    for candidate in candidates:
        marker = candidate / ".cross-view-matching-complete"
        if marker.is_file():
            return int(marker.read_text(encoding="utf-8").strip() or 0)
    return 0

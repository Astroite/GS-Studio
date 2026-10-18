from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence, TypeVar

import cv2
import numpy as np

from gsstudio.domain.models import COCO_DYNAMIC_CLASS_IDS, MaskingConfig
from gsstudio.infrastructure.adapters.image_io import atomic_imwrite, read_cv_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_T = TypeVar("_T")


IO_WORKERS_ENV = "GSSTUDIO_IO_WORKERS"


def io_worker_count() -> int:
    """Threads used for JPEG/PNG decode, encode and write.

    OpenCV releases the GIL around codec and resize calls, so these stages scale
    with threads while the projection keeps its single CUDA stream serialized.
    """
    override = os.environ.get(IO_WORKERS_ENV, "").strip()
    if override:
        workers = int(override)
        if workers < 1:
            raise ValueError(f"{IO_WORKERS_ENV} must be at least 1, got {workers}")
        return workers
    return max(1, min(8, (os.cpu_count() or 1)))


def run_image_tasks(action: Callable[[_T], Any], items: Sequence[_T]) -> list[Any]:
    """Run ``action`` over ``items``, surfacing the first failure deterministically."""
    workers = min(io_worker_count(), len(items))
    if workers <= 1:
        return [action(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [future.result() for future in [pool.submit(action, item) for item in items]]


@dataclass(frozen=True)
class PersonPrediction:
    mask: np.ndarray
    detections: int
    detections_by_class: dict[str, int] = field(default_factory=dict)
    masks_by_class: dict[str, np.ndarray] = field(default_factory=dict)


PersonPredictor = Callable[[np.ndarray], PersonPrediction]


def image_files(path: Path) -> list[Path]:
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda item: item.relative_to(path).as_posix(),
    )


_JPEG_SIZE_MARKERS = frozenset(
    # SOF0-SOF15 carry the frame dimensions; DHT/JPG/DAC share the range and do not.
    value for value in range(0xC0, 0xD0) if value not in (0xC4, 0xC8, 0xCC)
)


def image_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` from the file header without decoding pixels.

    Validating a cached image set with ``cv2.imread`` costs a full JPEG decode per
    file — about 50 ms each over the 9p mount — which dominates every resume. The
    header carries the only property those checks actually test.
    """
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            header = stream.read(17)
            if len(header) < 17 or header[4:8] != b"IHDR":
                raise RuntimeError(f"Truncated PNG header: {path}")
            width, height = struct.unpack(">II", header[8:16])
            return int(height), int(width)
        if not prefix.startswith(b"\xff\xd8"):
            raise RuntimeError(f"Unsupported image header: {path}")
        stream.seek(2)
        while True:
            marker = stream.read(2)
            if len(marker) < 2 or marker[0] != 0xFF:
                raise RuntimeError(f"Truncated JPEG header: {path}")
            if marker[1] in _JPEG_SIZE_MARKERS:
                frame = stream.read(7)
                if len(frame) < 7:
                    raise RuntimeError(f"Truncated JPEG frame header: {path}")
                height, width = struct.unpack(">HH", frame[3:7])
                return int(height), int(width)
            length = stream.read(2)
            if len(length) < 2:
                raise RuntimeError(f"Truncated JPEG segment: {path}")
            stream.seek(struct.unpack(">H", length)[0] - 2, os.SEEK_CUR)


def mask_path_for_image(
    masks_dir: Path, image: Path, images_dir: Path | None = None
) -> Path:
    """COLMAP appends .png to the complete relative image name."""
    images_dir = images_dir or masks_dir.parent / "images"
    try:
        relative = image.relative_to(images_dir)
    except ValueError:
        relative = Path(image.name)
    return masks_dir / relative.parent / f"{relative.name}.png"


def postprocess_person_mask(mask: np.ndarray, config: MaskingConfig) -> np.ndarray:
    ignored = np.asarray(mask, dtype=np.uint8)
    if ignored.ndim != 2:
        raise ValueError("Person mask must be a 2D array")
    ignored = (ignored > 0).astype(np.uint8)
    if config.closing_pixels > 1:
        kernel = np.ones((config.closing_pixels, config.closing_pixels), np.uint8)
        ignored = cv2.morphologyEx(ignored, cv2.MORPH_CLOSE, kernel)
    if config.dilation_pixels > 0:
        diameter = config.dilation_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        ignored = cv2.dilate(ignored, kernel, iterations=1)
    return ignored.astype(bool)


def colmap_mask_from_ignored(ignored: np.ndarray) -> np.ndarray:
    """Return a mask where white is usable and black is ignored by COLMAP/Nerfstudio."""
    return np.where(np.asarray(ignored, dtype=bool), 0, 255).astype(np.uint8)


def ignored_from_colmap_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Stored mask must be single-channel")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        raise ValueError(f"Stored mask is not binary: {sorted(values)}")
    return mask == 0


class TorchvisionPersonSegmenter:
    """Lazy Mask R-CNN wrapper; importing this module never downloads model weights."""

    def __init__(self, config: MaskingConfig):
        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Masking requested CUDA, but torch.cuda.is_available() is false")
        self._torch = torch
        self._config = config
        self._device = torch.device(config.device)
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = maskrcnn_resnet50_fpn_v2(weights=weights).to(self._device).eval()

    def __call__(self, image_bgr: np.ndarray) -> PersonPrediction:
        torch = self._torch
        try:
            return self._predict(image_bgr)
        finally:
            # torchvision pastes every detection's mask at the source resolution,
            # so one 8K equirect frame allocates a handful of ~118 MB blocks whose
            # count varies with the number of people found. The caching allocator
            # keeps those blocks for reuse and cannot defragment them, so
            # memory_reserved climbs past physical VRAM within about twenty frames;
            # WDDM then starts paging and per-frame time goes from ~1.5 s to
            # 100-250 s. Live allocations stay flat throughout, so releasing the
            # cached blocks is the whole fix. Measured over 20 native 8K frames:
            # 498 s and 41 GiB reserved without this, 37 s and 0.19 GiB with it.
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    def _predict(self, image_bgr: np.ndarray) -> PersonPrediction:
        torch = self._torch
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self._config.inference_gamma != 1.0:
            rgb = np.power(rgb, self._config.inference_gamma)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self._device)
        with torch.inference_mode():
            output = self._model([tensor])[0]
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        class_names = list(getattr(self._config, "classes", ["person"]))
        class_ids = {
            name: (
                self._config.person_class_id
                if name == "person"
                else COCO_DYNAMIC_CLASS_IDS[name]
            )
            for name in class_names
        }
        selected = np.flatnonzero(
            np.isin(labels, list(class_ids.values()))
            & (scores >= self._config.score_threshold)
        )
        if len(selected) == 0:
            return PersonPrediction(
                mask=np.zeros(image_bgr.shape[:2], dtype=bool),
                detections=0,
                detections_by_class={name: 0 for name in class_names},
                masks_by_class={
                    name: np.zeros(image_bgr.shape[:2], dtype=bool)
                    for name in class_names
                },
            )
        probabilities = output["masks"][selected, 0].detach().cpu().numpy()
        selected_labels = labels[selected]
        masks_by_class = {}
        detections_by_class = {}
        for name, class_id in class_ids.items():
            positions = np.flatnonzero(selected_labels == class_id)
            detections_by_class[name] = len(positions)
            masks_by_class[name] = (
                np.any(
                    probabilities[positions] >= self._config.probability_threshold,
                    axis=0,
                )
                if len(positions)
                else np.zeros(image_bgr.shape[:2], dtype=bool)
            )
        mask = np.any(list(masks_by_class.values()), axis=0)
        return PersonPrediction(
            mask=mask,
            detections=len(selected),
            detections_by_class=detections_by_class,
            masks_by_class=masks_by_class,
        )


def _read_existing_mask(mask_path: Path, shape: tuple[int, int]) -> np.ndarray:
    stored = read_cv_image(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if stored is None:
        raise RuntimeError(f"Cannot read existing mask: {mask_path}")
    if stored.shape != shape:
        raise RuntimeError(
            f"Existing mask size mismatch for {mask_path.name}: {stored.shape} != {shape}"
        )
    return ignored_from_colmap_mask(stored)


def generate_person_masks(
    images_dir: Path,
    masks_dir: Path,
    config: MaskingConfig,
    metrics_path: Path,
    predictor: PersonPredictor | None = None,
) -> list[dict[str, object]]:
    images = image_files(images_dir)
    if not images:
        raise RuntimeError(f"No planar images found in {images_dir}")
    masks_dir.mkdir(parents=True, exist_ok=True)
    if config.enabled and predictor is None:
        predictor = TorchvisionPersonSegmenter(config)

    previous_records: dict[str, dict[str, object]] = {}
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(previous, dict) and "image" in previous:
                previous_records[str(previous["image"])] = previous
    records: list[dict[str, object]] = []
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_stream:
        for index, image_path in enumerate(images, start=1):
            image_name = image_path.relative_to(images_dir).as_posix()
            target = mask_path_for_image(masks_dir, image_path, images_dir)
            if target.exists():
                # Only the shape is needed to check a cached mask, so skip the decode.
                ignored = _read_existing_mask(target, image_dimensions(image_path))
                detections = previous_records.get(image_name, {}).get("detections")
                class_metrics = previous_records.get(image_name, {}).get("classes", {})
                reused = True
            else:
                image = read_cv_image(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Cannot decode planar image: {image_path}")
                prediction = (
                    predictor(image)
                    if predictor is not None
                    else PersonPrediction(np.zeros(image.shape[:2], dtype=bool), 0)
                )
                if prediction.mask.shape != image.shape[:2]:
                    raise RuntimeError(
                        f"Segmenter mask size mismatch for {image_path.name}: "
                        f"{prediction.mask.shape} != {image.shape[:2]}"
                    )
                ignored = postprocess_person_mask(prediction.mask, config)
                detections = prediction.detections
                class_metrics = {
                    name: {
                        "detections": int(
                            prediction.detections_by_class.get(name, 0)
                        ),
                        "masked_fraction": round(
                            float(
                                np.count_nonzero(
                                    postprocess_person_mask(class_mask, config)
                                )
                                / class_mask.size
                            ),
                            8,
                        ),
                    }
                    for name, class_mask in prediction.masks_by_class.items()
                }
                if not class_metrics:
                    class_metrics = {
                        "person": {
                            "detections": int(prediction.detections),
                            "masked_fraction": round(
                                float(np.count_nonzero(ignored) / ignored.size), 8
                            ),
                        }
                    }
                reused = False
                atomic_imwrite(target, colmap_mask_from_ignored(ignored))
            fraction = float(np.count_nonzero(ignored) / ignored.size)
            record: dict[str, object] = {
                "image": image_name,
                "mask": target.relative_to(masks_dir).as_posix(),
                "detections": detections,
                "classes": class_metrics,
                "masked_fraction": round(fraction, 8),
                "reused": reused,
            }
            records.append(record)
            metrics_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index == 1 or index % 50 == 0 or index == len(images):
                metrics_stream.flush()
                print(f"Person masks: {index}/{len(images)}", flush=True)
    return records


def validate_mask_set(
    images_dir: Path,
    masks_dir: Path,
    max_masked_fraction: float,
) -> dict[str, object]:
    images = image_files(images_dir)
    expected = {
        mask_path_for_image(masks_dir, image, images_dir).relative_to(masks_dir).as_posix(): image
        for image in images
    }
    actual = {
        item.relative_to(masks_dir).as_posix(): item
        for item in masks_dir.rglob("*.png")
        if item.is_file()
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise RuntimeError(f"Mask/image mismatch; missing={missing}, extra={extra}")
    def measure(item: tuple[str, Path]) -> float:
        name, image_path = item
        # The mask is decoded because its pixels drive the ratio; the image only has
        # to agree on shape, which the header already carries.
        mask = read_cv_image(str(actual[name]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot decode image/mask pair: {image_path.name}")
        if mask.shape != image_dimensions(image_path):
            raise RuntimeError(f"Mask size mismatch for {image_path.name}")
        ignored = ignored_from_colmap_mask(mask)
        fraction = float(np.count_nonzero(ignored) / ignored.size)
        if fraction > max_masked_fraction:
            raise RuntimeError(
                f"Mask for {image_path.name} hides {fraction:.1%}, above "
                f"the configured {max_masked_fraction:.1%} ceiling"
            )
        return fraction

    fractions: list[float] = run_image_tasks(measure, list(expected.items()))
    return {
        "image_count": len(images),
        "mask_count": len(actual),
        "masked_images": sum(value > 0 for value in fractions),
        "zero_mask_images": sum(value == 0 for value in fractions),
        "mean_masked_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "max_masked_fraction": max(fractions, default=0.0),
        "deterministic_qa": "passed",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _safe_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value:
        raise RuntimeError(f"Unsafe mask-filter path: {value}")
    return relative


def validate_mask_filter(
    dataset: Path, *, verify_hashes: bool = True
) -> dict[str, Any]:
    path = dataset / "mask-filter.json"
    if not path.is_file():
        raise RuntimeError(f"Mask filter manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise RuntimeError(f"Mask filter manifest is incomplete: {path}")
    accepted = payload.get("accepted")
    rejected = payload.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise RuntimeError("Mask filter manifest has invalid inventories")
    accepted_images = {str(item["image"]) for item in accepted}
    accepted_masks = {str(item["mask"]) for item in accepted}
    rejected_images = {str(item["image"]) for item in rejected}
    rejected_masks = {str(item["mask"]) for item in rejected}
    if len(accepted_images) != len(accepted) or len(accepted_masks) != len(accepted):
        raise RuntimeError("Mask filter accepted inventory contains duplicates")
    if len(rejected_images) != len(rejected) or len(rejected_masks) != len(rejected):
        raise RuntimeError("Mask filter rejected inventory contains duplicates")
    if accepted_images & rejected_images or accepted_masks & rejected_masks:
        raise RuntimeError("Mask filter accepted/rejected inventories overlap")
    if int(payload.get("original_image_count", -1)) != len(accepted) + len(rejected):
        raise RuntimeError("Mask filter original inventory count is invalid")
    threshold = float(payload.get("threshold", -1.0))
    for item in accepted + rejected:
        _safe_relative(str(item["image"]))
        _safe_relative(str(item["mask"]))
        for field in ("image_sha256", "mask_sha256"):
            if not isinstance(item.get(field), str) or len(str(item[field])) != 64:
                raise RuntimeError(f"Mask filter entry has invalid {field}")
    if any(float(item["masked_fraction"]) > threshold for item in accepted):
        raise RuntimeError("Mask filter accepted inventory violates its threshold")
    if any(float(item["masked_fraction"]) <= threshold for item in rejected):
        raise RuntimeError("Mask filter rejected inventory violates its threshold")
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    actual_images = {
        item.relative_to(images_dir).as_posix() for item in image_files(images_dir)
    }
    actual_masks = {
        item.relative_to(masks_dir).as_posix() for item in image_files(masks_dir)
    }
    if actual_images != accepted_images or actual_masks != accepted_masks:
        raise RuntimeError("Filtered image/mask inventory no longer matches mask-filter.json")
    for item in rejected:
        image = images_dir / _safe_relative(str(item["image"]))
        mask = masks_dir / _safe_relative(str(item["mask"]))
        if image.exists() or mask.exists():
            raise RuntimeError("Rejected mask-filter files still exist")
    if verify_hashes:
        review_path = dataset / 'mask-review.json'
        review = json.loads(review_path.read_text(encoding='utf-8')) if review_path.is_file() else {}
        reviewed = {str(name): entry for name, entry in review.get('reviews', {}).items()}
        for item in accepted:
            image = images_dir / _safe_relative(str(item["image"]))
            mask = masks_dir / _safe_relative(str(item["mask"]))
            if _sha256(image) != item["image_sha256"]:
                raise RuntimeError("Accepted mask-filter file hash changed")
            digest = _sha256(mask)
            if digest != item['mask_sha256']:
                matching = [key for key in reviewed
                            if review.get('image_path',{}).get(key)==item['image']
                            and review.get('mask_sha256',{}).get(key)==digest
                            and reviewed[key].get('status') in {'ok','exclude','false_positive','false_negative'}]
                if review.get('schema_version')!=1 or len(matching)!=1:
                    raise RuntimeError('Accepted mask-filter mask changed without current identity-bound review')
                _read_existing_mask(mask, image_dimensions(image))
    return payload


def _recover_mask_filter(
    dataset: Path, threshold: float, *, review_required: bool = False,
) -> dict[str, Any] | None:
    """Preflight all retained media before resuming authoritative deletions.

    This explicit recovery operation does not relax the read-only validator.
    A completed record is immutable; pending intent is completed atomically.
    """
    import math
    import stat
    from pathlib import PureWindowsPath
    from gsstudio.infrastructure.persistence.manifests import canonical_hash

    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Mask discard threshold must be between 0 and 1")

    # These caches belong only to this invocation's read-only preflight. Never
    # reuse them after cleanup starts or across recovery attempts.
    verified_parts: set[Path] = set()
    resolved_paths: dict[Path, Path] = {}
    safe_paths: dict[tuple[Path, str], Path] = {}

    def resolved(path: Path) -> Path:
        if path not in resolved_paths:
            resolved_paths[path] = path.resolve()
        return resolved_paths[path]

    def safe(root: Path, name: object) -> Path:
        if isinstance(name, str) and (root, name) in safe_paths:
            return safe_paths[root, name]
        if (not isinstance(name, str) or not name or "\\" in name
                or PureWindowsPath(name).drive or ":" in name
                or any(part in {"", ".", ".."} or part.endswith((".", " "))
                       or PureWindowsPath(part).is_reserved() for part in name.split("/"))):
            raise RuntimeError(f"Unsafe mask-filter path: {name}")
        target = root / name
        for part in [target, *target.parents]:
            if part in verified_parts:
                # A verified part includes every ancestor above it.
                break
            try:
                info = part.lstat()
            except FileNotFoundError:
                # Missing rejected files and optional records are allowed;
                # their existing ancestors still need to be checked.
                continue
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise RuntimeError(f"Unsafe mask-filter link: {part}")
        if (resolved(target) != target.absolute()
                or not resolved(target).is_relative_to(resolved(root))):
            raise RuntimeError(f"Escaping mask-filter path: {target}")
        verified_parts.update(target.parents)
        verified_parts.add(target)
        safe_paths[root, name] = target
        return target

    def read(path: Path, status: str) -> dict[str, Any]:
        print(f"Mask-filter recovery: preflight record {path.name}", flush=True)
        safe(dataset, path.name)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
                    or value.get("schema_version") != 1
                    or value.get("status") != status):
                raise ValueError("invalid schema/status")
            limit = value["threshold"]
            if type(limit) not in (float, int) or not math.isfinite(limit) or not 0 < limit < 1:
                raise ValueError("invalid threshold")
            if abs(limit - threshold) > 1e-12:
                raise RuntimeError(f"Mask-filter threshold conflict: {path}")
            inventories = []
            for kind in ("accepted", "rejected"):
                entries = value[kind]
                if not isinstance(entries, list):
                    raise ValueError(f"invalid {kind} inventory")
                print(f"Mask-filter recovery: validating {len(entries)} {kind} entries", flush=True)
                names: list[set[str]] = [set(), set()]
                for entry_index, entry in enumerate(entries, 1):
                    for index, field in enumerate(("image", "mask")):
                        target = safe(dataset / ("images" if index == 0 else "masks"), entry[field])
                        key = entry[field].casefold()
                        if key in names[index]:
                            raise RuntimeError(f"Duplicate mask-filter path: {target}")
                        names[index].add(key)
                        digest = entry[field + "_sha256"]
                        if (not isinstance(digest, str) or len(digest) != 64
                                or any(c not in "0123456789abcdef" for c in digest)):
                            raise RuntimeError(f"Invalid mask-filter hash: {target}")
                    fraction = entry["masked_fraction"]
                    if (Path(entry["image"]).suffix.lower() not in IMAGE_SUFFIXES
                            or entry["mask"] != entry["image"] + ".png"
                            or type(fraction) not in (int, float)
                            or not math.isfinite(fraction) or not 0 <= fraction <= 1
                            or (fraction <= limit) != (kind == "accepted")):
                        raise RuntimeError(f"Invalid mask-filter entry: {path}: {entry['image']}")
                    if entry_index % 1000 == 0:
                        print(f"Mask-filter recovery: validated {entry_index}/{len(entries)} {kind} entries",
                              flush=True)
                inventories.append(names)
            for index in (0, 1):
                overlap = inventories[0][index] & inventories[1][index]
                if overlap:
                    raise RuntimeError(f"Overlapping mask-filter paths: {path}: {sorted(overlap)}")
            if (type(value["original_image_count"]) is not int
                    or value["original_image_count"] != len(value["accepted"]) + len(value["rejected"])):
                raise ValueError("invalid original count")
            return value
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Invalid mask-filter record: {path}: {error}") from error

    final = dataset / "mask-filter.json"
    pending = dataset / ".mask-filter.pending.json"
    complete = read(final, "complete") if final.exists() or final.is_symlink() else None
    intent = read(pending, "pending") if pending.exists() or pending.is_symlink() else None
    if complete is None and intent is None:
        return None
    if complete is not None and intent is not None:
        def comparable(value: dict[str, Any]) -> dict[str, Any]:
            return {**value, "status": "complete",
                    "accepted": sorted(value["accepted"], key=lambda item: item["image"]),
                    "rejected": sorted(value["rejected"], key=lambda item: item["image"])}
        if comparable(intent) != comparable(complete):
            raise RuntimeError(f"Conflicting mask-filter records: {final}, {pending}")
    payload = complete if complete is not None else intent
    accepted, rejected = payload["accepted"], payload["rejected"]
    for folder, field in (("images", "image"), ("masks", "mask")):
        print(f"Mask-filter recovery: beginning {folder} inventory "
              f"({len(accepted)} accepted, {len(rejected)} rejected)", flush=True)
        root = safe(dataset, folder)
        known = {item[field] for item in accepted + rejected}
        inventory_count = 0
        for inventory_count, path in enumerate(sorted(root.rglob("*")), 1):
            safe(root, path.relative_to(root).as_posix())
            if not path.is_dir() and path.relative_to(root).as_posix() not in known:
                raise RuntimeError(f"Unknown mask-filter file: {path}")
            if inventory_count % 1000 == 0:
                print(f"Mask-filter recovery: {folder} inventory {inventory_count} paths", flush=True)
        for entry in accepted:
            path = safe(root, entry[field])
            if not path.is_file():
                raise RuntimeError(f"Missing accepted mask-filter file: {path}")
        for entry in rejected:
            path = safe(root, entry[field])
            if path.exists() and not path.is_file():
                raise RuntimeError(f"Invalid rejected mask-filter file: {path}")
        print(f"Mask-filter recovery: {folder} inventory complete ({inventory_count} paths)", flush=True)

    review_path = safe(dataset, "mask-review.json")
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.is_file() else {}
    finalization_path = safe(dataset, "mask-final.json")
    finalization = json.loads(finalization_path.read_text(encoding="utf-8")) if finalization_path.is_file() else None
    hashes = {}
    print(f"Mask-filter recovery: checking hashes/masks for {len(accepted)} accepted pairs", flush=True)
    for index, entry in enumerate(sorted(accepted, key=lambda e: e["image"]), 1):
        image = dataset / "images" / entry["image"]
        mask = dataset / "masks" / entry["mask"]
        if _sha256(image) != entry["image_sha256"]:
            raise RuntimeError(f"Accepted mask-filter image hash changed: {image}")
        digest = _sha256(mask)
        hashes[entry["mask"]] = digest
        if digest != entry["mask_sha256"]:
            key = str(index)
            if not (review_required and review.get("schema_version") == 1
                    and review.get("reviews", {}).get(key, {}).get("status") in
                    {"ok", "exclude", "false_positive", "false_negative"}
                    and review.get("mask_sha256", {}).get(key) == digest
                    and review.get("image_path", {}).get(key, entry["image"]) == entry["image"]):
                raise RuntimeError(f"Accepted mask-filter mask hash changed without current review: {mask}")
        try:
            ignored = _read_existing_mask(mask, image_dimensions(image))
        except ValueError as error:
            raise RuntimeError(f"Invalid accepted mask: {mask}: {error}") from error
        if not review_required and float(np.count_nonzero(ignored) / ignored.size) > threshold:
            raise RuntimeError(f"Accepted mask exceeds threshold: {mask}")
        if index % 1000 == 0 or index == len(accepted):
            print(f"Mask-filter recovery: hashes/masks {index}/{len(accepted)} accepted pairs", flush=True)
    if finalization is not None:
        excluded = finalization.get("excluded_images")
        if (not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded)
                or excluded != sorted(set(excluded))
                or not set(excluded).issubset({e["image"] for e in accepted})):
            raise RuntimeError(f"Invalid exclusions: {finalization_path}")
        immutable = {
            "mask_sha256": hashes,
            "review_sha256": _sha256(review_path) if review_path.is_file() else None,
            "image_inventory_sha256": canonical_hash(sorted(e["image"] for e in accepted)),
            "excluded_images": excluded,
            "excluded_images_sha256": canonical_hash({"images": excluded}),
        }
        if (finalization.get("schema_version") != 1 or finalization.get("status") != "passed"
                or any(finalization.get(k) != v for k, v in immutable.items())
                or finalization.get("finalization_sha256") != canonical_hash(immutable)):
            raise RuntimeError(f"Stale or corrupt finalization: {finalization_path}")

    qa_review_path = safe(dataset, "mask-qa/codex-local-review.json")
    if qa_review_path.is_file():
        qa_review = json.loads(qa_review_path.read_text(encoding="utf-8"))
        bound = qa_review.get("contact_sheet_sha256")
        if not isinstance(bound, dict) or not bound:
            raise RuntimeError(f"Invalid QA sheet hashes: {qa_review_path}")
        for name, digest in bound.items():
            sheet = safe(dataset / "mask-qa", name)
            if not sheet.is_file() or _sha256(sheet) != digest:
                raise RuntimeError(f"Missing or changed reviewed QA sheet: {sheet}")

    # Everything above is read-only. Partial cleanup is safe to repeat.
    safe_paths.clear()
    resolved_paths.clear()
    verified_parts.clear()
    print(f"Mask-filter recovery: preflight complete; cleaning up {len(rejected)} rejected pairs", flush=True)
    for entry in rejected:
        for folder, field in (("images", "image"), ("masks", "mask")):
            (dataset / folder / entry[field]).unlink(missing_ok=True)
    if complete is None:
        payload = {**payload, "status": "complete"}
        _atomic_json(final, payload)
    pending.unlink(missing_ok=True)
    print("Mask-filter recovery: cleanup complete", flush=True)
    return payload


def _recover_mask_records(
    dataset: Path, payload: dict[str, Any], metrics: Path,
) -> list[dict[str, object]]:
    previous = {}
    if metrics.is_file():
        for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and isinstance(item.get("image"), str):
                previous[item["image"]] = item
    records = []
    for entry in payload["accepted"]:
        old = previous.get(entry["image"], {})
        fraction = entry["masked_fraction"]
        mask = dataset / "masks" / entry["mask"]
        if _sha256(mask) != entry["mask_sha256"]:
            ignored = _read_existing_mask(mask, image_dimensions(dataset / "images" / entry["image"]))
            fraction = float(np.count_nonzero(ignored) / ignored.size)
        records.append({"image": entry["image"], "mask": entry["mask"],
                        "masked_fraction": fraction, "detections": old.get("detections"),
                        "classes": old.get("classes", {}), "reused": True})
    return records


def filter_masked_images(
    dataset: Path,
    records: list[dict[str, object]],
    threshold: float,
) -> dict[str, Any]:
    """Delete views above the threshold using a recoverable intent manifest."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Mask discard threshold must be between 0 and 1")

    pending_path = dataset / ".mask-filter.pending.json"
    recovered = _recover_mask_filter(dataset, threshold)
    if recovered is not None:
        return recovered
    else:
        images_dir = dataset / "images"
        masks_dir = dataset / "masks"
        entries: list[dict[str, Any]] = []
        for record in records:
            image_name = str(record["image"])
            mask_name = str(record["mask"])
            image = images_dir / _safe_relative(image_name)
            mask = masks_dir / _safe_relative(mask_name)
            fraction = float(record["masked_fraction"])
            entries.append(
                {
                    "image": image_name,
                    "mask": mask_name,
                    "masked_fraction": fraction,
                    "image_sha256": _sha256(image),
                    "mask_sha256": _sha256(mask),
                }
            )
        image_names = {item.relative_to(images_dir).as_posix() for item in image_files(images_dir)}
        mask_names = {item.relative_to(masks_dir).as_posix() for item in image_files(masks_dir)}
        recorded_images = {str(item["image"]) for item in entries}
        recorded_masks = {str(item["mask"]) for item in entries}
        if (
            len(recorded_images) != len(entries)
            or len(recorded_masks) != len(entries)
            or recorded_images != image_names
            or recorded_masks != mask_names
        ):
            raise RuntimeError("Mask metrics do not exactly cover the projected image/mask inventory")
        accepted = [item for item in entries if item["masked_fraction"] <= threshold]
        rejected = [
            {**item, "reason": "masked_fraction_above_threshold"}
            for item in entries
            if item["masked_fraction"] > threshold
        ]
        payload = {
            "schema_version": 1,
            "status": "pending",
            "threshold": threshold,
            "comparison": "masked_fraction > threshold",
            "original_image_count": len(entries),
            "accepted": accepted,
            "rejected": rejected,
        }
        _atomic_json(pending_path, payload)
    return _recover_mask_filter(dataset, threshold)


def select_qa_records(
    records: list[dict[str, object]], sample_count: int
) -> list[dict[str, object]]:
    if len(records) <= sample_count:
        return records
    half = max(1, sample_count // 2)
    by_fraction = sorted(records, key=lambda item: float(item["masked_fraction"]), reverse=True)
    chosen = {str(item["image"]): item for item in by_fraction[:half]}
    remaining_slots = sample_count - len(chosen)
    if remaining_slots > 0:
        step = (len(records) - 1) / max(1, remaining_slots - 1)
        for index in range(remaining_slots):
            item = records[round(index * step)]
            chosen[str(item["image"])] = item
    if len(chosen) < sample_count:
        for item in records:
            chosen.setdefault(str(item["image"]), item)
            if len(chosen) == sample_count:
                break
    return sorted(chosen.values(), key=lambda item: str(item["image"]))[:sample_count]


def create_mask_contact_sheets(
    images_dir: Path,
    masks_dir: Path,
    records: list[dict[str, object]],
    output_dir: Path,
    sample_count: int,
    max_sheets: int,
) -> list[Path]:
    selected = select_qa_records(records, min(sample_count, max_sheets * 8))
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Path] = []
    tile_size = 320
    for sheet_index in range(0, len(selected), 8):
        batch = selected[sheet_index : sheet_index + 8]
        canvas = np.zeros((2 * (tile_size + 34), 4 * tile_size, 3), dtype=np.uint8)
        for tile_index, record in enumerate(batch):
            image_path = images_dir / Path(str(record["image"]))
            image = read_cv_image(str(image_path), cv2.IMREAD_COLOR)
            mask = read_cv_image(
                str(mask_path_for_image(masks_dir, image_path, images_dir)),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None or mask is None:
                raise RuntimeError(f"Cannot build QA tile for {image_path.name}")
            ignored = mask == 0
            overlay = image.copy()
            overlay[ignored] = (
                overlay[ignored].astype(np.float32) * 0.35
                + np.array([0, 0, 255], dtype=np.float32) * 0.65
            ).astype(np.uint8)
            overlay = cv2.resize(overlay, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            row, column = divmod(tile_index, 4)
            y = row * (tile_size + 34)
            x = column * tile_size
            canvas[y : y + tile_size, x : x + tile_size] = overlay
            label = f"{record['image']}  mask={float(record['masked_fraction']):.1%}"
            cv2.putText(
                canvas,
                label[:43],
                (x + 4, y + tile_size + 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        target = output_dir / f"mask-contact-{len(sheets) + 1:02d}.jpg"
        atomic_imwrite(target, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheets.append(target)
    return sheets


def summarize_detection_records(records: Iterable[dict[str, object]]) -> dict[str, object]:
    values = list(records)
    known_detections = [
        int(item["detections"]) for item in values if item.get("detections") is not None
    ]
    class_totals: dict[str, dict[str, float | int]] = {}
    for item in values:
        classes = item.get("classes")
        if not isinstance(classes, dict):
            continue
        for name, raw in classes.items():
            if not isinstance(raw, dict):
                continue
            totals = class_totals.setdefault(
                str(name), {"detections": 0, "masked_fraction_sum": 0.0}
            )
            totals["detections"] = int(totals["detections"]) + int(
                raw.get("detections", 0)
            )
            totals["masked_fraction_sum"] = float(
                totals["masked_fraction_sum"]
            ) + float(raw.get("masked_fraction", 0.0))
    per_class = {
        name: {
            "detections": int(totals["detections"]),
            "mean_masked_fraction": (
                float(totals["masked_fraction_sum"]) / len(values) if values else 0.0
            ),
        }
        for name, totals in sorted(class_totals.items())
    }
    return {
        "images": len(values),
        "images_with_detection_metrics": len(known_detections),
        "detection_metrics_complete": len(known_detections) == len(values),
        "images_with_detections": sum(value > 0 for value in known_detections),
        "person_instances": sum(known_detections),
        "per_class": per_class,
        "reused_masks": sum(bool(item.get("reused")) for item in values),
    }

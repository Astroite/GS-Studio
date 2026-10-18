from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image
import numpy as np

from gsstudio.pipeline.masks.masking import image_files, mask_path_for_image


REVIEW_SCHEMA_VERSION = 1
EDITABLE_REVIEW_STATUSES = frozenset(
    {"unreviewed", "ok", "false_positive", "false_negative", "exclude"}
)
REVIEW_STATUSES = (*sorted(EDITABLE_REVIEW_STATUSES), "stale")
STATIC_DIR = Path(__file__).resolve().parents[2] / "resources"
INDEX_PATH = STATIC_DIR / "mask-review.html"


@dataclass(frozen=True)
class ReviewItem:
    image_id: int
    source_image: str
    output_image: str
    output_mask: str
    width: int
    height: int
    ignored_pixels: int

    @property
    def masked_fraction(self) -> float:
        pixels = self.width * self.height
        return self.ignored_pixels / pixels if pixels else 0.0


class MaskReviewDataset:
    """Read a prepared Postshot dataset and persist non-destructive review labels."""

    def __init__(self, path: Path, *, lazy_stats: bool = False):
        self.path = path.absolute()
        self.lazy_stats = lazy_stats
        self.images_dir = self.path / "images"
        self.masks_dir = self.path / "masks"
        self.map_path = self.path / "image-map.csv"
        self.review_path = self.path / "mask-review.json"
        self.mask_semantics = self._mask_semantics()
        self._lock = threading.RLock()
        self._thumbnail_lock = threading.Lock()
        self._thumbnail_cache: OrderedDict[tuple[int, str, str], bytes] = OrderedDict()
        self.items = self._load_items()
        self._by_id = {item.image_id: item for item in self.items}
        self._review_hashes: dict[str, str] = {}
        self._review_paths: dict[str, str] = {}
        self._reviews = self._load_reviews()

    def _mask_semantics(self) -> str:
        manifest_path = self.path / "dataset.json"
        if manifest_path.is_file():
            try:
                value = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "mask_semantics"
                )
                if value in {"white_is_ignored", "black_is_ignored"}:
                    return str(value)
            except (json.JSONDecodeError, OSError):
                pass
        return "white_is_ignored" if self.map_path.is_file() else "black_is_ignored"

    def _load_items(self) -> list[ReviewItem]:
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise RuntimeError(f"Postshot images/masks are missing: {self.path}")
        if not self.map_path.is_file():
            images = image_files(self.images_dir)
            if not images:
                raise RuntimeError(f"Mask review dataset is empty: {self.path}")
            items = []
            prior = {}
            filter_path = self.path/'mask-filter.json'
            if self.lazy_stats and filter_path.is_file():
                prior = {row['image']:row for row in json.loads(filter_path.read_text(encoding='utf-8'))['accepted']}
            expected_masks: set[str] = set()
            for image_id, image_path in enumerate(images, start=1):
                relative = image_path.relative_to(self.images_dir)
                mask_path = mask_path_for_image(
                    self.masks_dir, image_path, self.images_dir
                )
                expected_masks.add(mask_path.relative_to(self.masks_dir).as_posix())
                if self.lazy_stats and relative.as_posix() in prior:
                    from gsstudio.pipeline.masks.masking import image_dimensions
                    height,width=image_dimensions(image_path)
                    ignored_pixels=round(prior[relative.as_posix()]['masked_fraction']*width*height)
                else:
                    width, height, ignored_pixels = self._media_stats(image_path, mask_path)
                items.append(
                    ReviewItem(
                        image_id=image_id,
                        source_image=relative.as_posix(),
                        output_image=relative.as_posix(),
                        output_mask=mask_path.relative_to(self.masks_dir).as_posix(),
                        width=width,
                        height=height,
                        ignored_pixels=ignored_pixels,
                    )
                )
            actual_masks = {
                item.relative_to(self.masks_dir).as_posix()
                for item in image_files(self.masks_dir)
            }
            if actual_masks != expected_masks:
                raise RuntimeError(
                    "Mask files do not exactly match reconstruction images: "
                    f"missing={sorted(expected_masks - actual_masks)}, "
                    f"extra={sorted(actual_masks - expected_masks)}"
                )
            return items
        items: list[ReviewItem] = []
        with self.map_path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                item = ReviewItem(
                    image_id=int(row["image_id"]),
                    source_image=row["source_image"],
                    output_image=row["output_image"],
                    output_mask=row["output_mask"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    ignored_pixels=int(row["ignored_pixels"]),
                )
                image_path = self.images_dir / item.output_image
                mask_path = self.masks_dir / item.output_mask
                if not image_path.is_file() or not mask_path.is_file():
                    raise RuntimeError(
                        f"Postshot image-map entry has missing media: {item.output_image}"
                    )
                width, height, ignored_pixels = self._media_stats(
                    image_path, mask_path
                )
                if (width, height) != (item.width, item.height):
                    raise RuntimeError(
                        f"Postshot image-map dimensions are stale: {item.output_image}"
                    )
                if ignored_pixels != item.ignored_pixels:
                    item = ReviewItem(
                        **{
                            **item.__dict__,
                            "ignored_pixels": ignored_pixels,
                        }
                    )
                items.append(item)
        if not items:
            raise RuntimeError(f"Postshot image map is empty: {self.map_path}")
        if len({item.image_id for item in items}) != len(items):
            raise RuntimeError(f"Postshot image map contains duplicate image IDs: {self.map_path}")
        return items

    def _media_stats(self, image_path: Path, mask_path: Path) -> tuple[int, int, int]:
        image = read_cv_image(str(image_path), cv2.IMREAD_COLOR)
        mask = read_cv_image(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Missing or invalid image/mask pair: {image_path}, {mask_path}")
        if mask.shape != image.shape[:2]:
            raise RuntimeError(
                f"Mask size mismatch: {mask_path} is {mask.shape}, image is {image.shape[:2]}"
            )
        values = set(int(value) for value in np.unique(mask))
        if not values.issubset({0, 255}):
            raise RuntimeError(f"Mask is not binary: {mask_path}; values={sorted(values)}")
        ignored = mask > 0 if self.mask_semantics == "white_is_ignored" else mask == 0
        return image.shape[1], image.shape[0], int(np.count_nonzero(ignored))

    def _load_reviews(self) -> dict[str, dict[str, str]]:
        if not self.review_path.is_file():
            return {}
        payload = json.loads(self.review_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported mask review schema: {payload.get('schema_version')}"
            )
        self._review_hashes = {
            str(int(image_id)): str(value)
            for image_id, value in payload.get("mask_sha256", {}).items()
        }
        self._review_paths = {
            str(int(image_id)): str(value)
            for image_id, value in payload.get("image_path", {}).items()
        }
        reviews: dict[str, dict[str, str]] = {}
        for image_id, review in payload.get("reviews", {}).items():
            status = str(review.get("status", "unreviewed"))
            if status not in EDITABLE_REVIEW_STATUSES:
                raise RuntimeError(f"Unsupported mask review status: {status}")
            if status != "unreviewed" or review.get("note"):
                reviews[str(int(image_id))] = {
                    "status": status,
                    "note": str(review.get("note", "")),
                }
        return reviews

    def get_item(self, image_id: int) -> ReviewItem:
        try:
            return self._by_id[image_id]
        except KeyError as error:
            raise KeyError(f"Unknown image ID: {image_id}") from error

    def orphaned_review_ids(self) -> list[str]:
        current = {str(item.image_id) for item in self.items}
        return sorted(set(self._reviews) - current)

    def media_path(self, image_id: int, kind: str) -> Path:
        item = self.get_item(image_id)
        if kind == "image":
            return self.images_dir / item.output_image
        if kind == "mask":
            return self.masks_dir / item.output_mask
        raise KeyError(f"Unknown media kind: {kind}")

    def review_for(self, image_id: int) -> dict[str, str]:
        with self._lock:
            key = str(image_id)
            review = dict(
                self._reviews.get(key, {"status": "unreviewed", "note": ""})
            )
            if key in self._reviews and self._review_hashes.get(key) != self.mask_sha256(image_id):
                review["status"] = "stale"
            if (
                key in self._reviews
                and self._review_paths.get(key, self.get_item(image_id).output_image)
                != self.get_item(image_id).output_image
            ):
                review["status"] = "stale"
            return review

    def mask_sha256(self, image_id: int) -> str:
        import hashlib

        digest = hashlib.sha256()
        with self.media_path(image_id, "mask").open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        with self._lock:
            entries = []
            for item in self.items:
                review = self.review_for(item.image_id)
                width, height, ignored_pixels = self._media_stats(
                    self.media_path(item.image_id, "image"),
                    self.media_path(item.image_id, "mask"),
                )
                entries.append(
                    {
                        "image_id": item.image_id,
                        "source_image": item.source_image,
                        "output_image": item.output_image,
                        "output_mask": item.output_mask,
                        "width": width,
                        "height": height,
                        "ignored_pixels": ignored_pixels,
                        "masked_fraction": round(ignored_pixels / (width * height), 8),
                        "mask_sha256": self.mask_sha256(item.image_id),
                        "status": review["status"],
                        "note": review["note"],
                    }
                )
            counts = Counter(entry["status"] for entry in entries)
            return {
                "dataset": str(self.path),
                "review_file": str(self.review_path),
                "counts": {
                    "total": len(entries),
                    **{status: counts.get(status, 0) for status in REVIEW_STATUSES},
                },
                "items": entries,
            }

    def update_review(self, image_id: int, status: str, note: str) -> dict[str, str]:
        self.get_item(image_id)
        if status not in EDITABLE_REVIEW_STATUSES:
            raise ValueError(f"Unsupported mask review status: {status}")
        note = note.strip()
        if len(note) > 2000:
            raise ValueError("Review note is longer than 2000 characters")
        with self._lock:
            key = str(image_id)
            if status == "unreviewed" and not note:
                self._reviews.pop(key, None)
                self._review_hashes.pop(key, None)
                self._review_paths.pop(key, None)
            else:
                self._reviews[key] = {"status": status, "note": note}
                self._review_hashes[key] = self.mask_sha256(image_id)
                self._review_paths[key] = self.get_item(image_id).output_image
            self._write_reviews()
            return {"status": status, "note": note}

    def _write_reviews(self) -> None:
        payload = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "dataset": str(self.path),
            "updated_at": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "reviews": self._reviews,
            "mask_sha256": self._review_hashes,
            "image_path": self._review_paths,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.review_path.name}.",
            suffix=".tmp",
            dir=self.review_path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.review_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def thumbnail(self, image_id: int, mode: str = "overlay") -> bytes:
        if mode not in {"overlay", "image", "mask"}:
            raise ValueError(f"Unsupported thumbnail mode: {mode}")
        cache_key = (image_id, mode, self.mask_sha256(image_id))
        with self._thumbnail_lock:
            cached = self._thumbnail_cache.pop(cache_key, None)
            if cached is not None:
                self._thumbnail_cache[cache_key] = cached
                return cached
        image = read_cv_image(str(self.media_path(image_id, "image")), cv2.IMREAD_COLOR)
        mask = read_cv_image(str(self.media_path(image_id, "mask")), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or mask.shape != image.shape[:2]:
            raise RuntimeError(f"Cannot decode image/mask pair for image ID {image_id}")
        scale = min(1.0, 320.0 / max(image.shape[:2]))
        size = (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        )
        thumb = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        ignored = (
            resized_mask > 0
            if self.mask_semantics == "white_is_ignored"
            else resized_mask == 0
        )
        if mode == "mask":
            normalized_mask = np.zeros_like(thumb)
            normalized_mask[ignored] = 255
            thumb = normalized_mask
        elif mode == "overlay":
            red = np.zeros_like(thumb)
            red[:, :, 2] = 255
            if np.any(ignored):
                thumb[ignored] = cv2.addWeighted(
                    thumb[ignored], 0.45, red[ignored], 0.55, 0
                )
        ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError(f"Cannot encode thumbnail for image ID {image_id}")
        value = encoded.tobytes()
        with self._thumbnail_lock:
            self._thumbnail_cache[cache_key] = value
            while len(self._thumbnail_cache) > 192:
                self._thumbnail_cache.popitem(last=False)
        return value

    def review_csv(self) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(
            ("image_id", "source_image", "output_image", "masked_fraction", "status", "note")
        )
        with self._lock:
            for item in self.items:
                review = self.review_for(item.image_id)
                writer.writerow(
                    (
                        item.image_id,
                        item.source_image,
                        item.output_image,
                        f"{item.masked_fraction:.8f}",
                        review["status"],
                        review["note"],
                    )
                )
        return output.getvalue().encode("utf-8-sig")


class MaskReviewHandler(BaseHTTPRequestHandler):
    server: "MaskReviewServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(
        self,
        value: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        cache: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(value)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, header_value in (headers or {}).items():
            self.send_header(name, header_value)
        self.end_headers()
        self.wfile.write(value)

    def _send_json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _image_id(self, query: dict[str, list[str]]) -> int:
        try:
            return int(query["id"][0])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("A numeric image id is required") from error

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._send_bytes(
                    INDEX_PATH.read_bytes(), "text/html; charset=utf-8"
                )
                return
            if parsed.path == "/api/items":
                self._send_json(self.server.dataset.payload())
                return
            if parsed.path == "/api/reviews.csv":
                self._send_bytes(
                    self.server.dataset.review_csv(),
                    "text/csv; charset=utf-8",
                    headers={
                        "Content-Disposition": 'attachment; filename="mask-review.csv"'
                    },
                )
                return
            image_id = self._image_id(query)
            if parsed.path == "/media/image":
                path = self.server.dataset.media_path(image_id, "image")
                self._send_bytes(path.read_bytes(), "image/jpeg", cache=True)
                return
            if parsed.path == "/media/mask":
                path = self.server.dataset.media_path(image_id, "mask")
                self._send_bytes(path.read_bytes(), "image/png")
                return
            if parsed.path == "/media/thumbnail":
                mode = query.get("mode", ["overlay"])[0]
                self._send_bytes(
                    self.server.dataset.thumbnail(image_id, mode),
                    "image/jpeg",
                )
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, FileNotFoundError):
            self._send_json({"error": "Image not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/review":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("Invalid request body size")
            payload = json.loads(self.rfile.read(length))
            review = self.server.dataset.update_review(
                int(payload["image_id"]),
                str(payload.get("status", "unreviewed")),
                str(payload.get("note", "")),
            )
            self._send_json({"ok": True, "review": review})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)


class MaskReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], dataset: MaskReviewDataset):
        self.dataset = dataset
        super().__init__(address, MaskReviewHandler)


def create_mask_review_server(
    dataset_path: Path, host: str = "127.0.0.1", port: int = 8765
) -> MaskReviewServer:
    if not INDEX_PATH.is_file():
        raise RuntimeError(f"Mask review web interface is missing: {INDEX_PATH}")
    if not (0 <= port <= 65535):
        raise ValueError("Port must be between 0 and 65535")
    return MaskReviewServer((host, port), MaskReviewDataset(dataset_path))

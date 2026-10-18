from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.pipeline.masks.review import MaskReviewDataset


MASK_FINAL_SCHEMA_VERSION = 1


class MaskFinalizationMissingError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _current_mask_hashes(dataset: MaskReviewDataset) -> dict[str, str]:
    return {
        item.output_mask: dataset.mask_sha256(item.image_id)
        for item in dataset.items
    }


def finalize_mask_dataset(
    path: Path,
    expected_images: set[str] | None = None,
    maximum_included_masked_fraction: float | None = None,
) -> dict[str, Any]:
    dataset = MaskReviewDataset(path)
    actual_images = {item.source_image for item in dataset.items}
    if expected_images is not None and actual_images != expected_images:
        raise RuntimeError(
            "Reconstruction image inventory is incomplete; images may only be removed "
            "through the exclude review status"
        )
    orphaned_reviews = dataset.orphaned_review_ids()
    if orphaned_reviews:
        raise RuntimeError(f"Mask review contains orphaned image IDs: {orphaned_reviews}")
    reviews = {
        str(item.image_id): dataset.review_for(item.image_id)
        for item in dataset.items
    }
    blocking = {
        image_id: review
        for image_id, review in reviews.items()
        if review["status"] in {"false_positive", "false_negative", "stale"}
    }
    if blocking:
        summary = ", ".join(
            f"{image_id}:{review['status']}" for image_id, review in blocking.items()
        )
        raise RuntimeError(f"Mask finalization blocked by unresolved reviews: {summary}")
    if maximum_included_masked_fraction is not None:
        over_limit = {
            str(item.image_id): item.masked_fraction
            for item in dataset.items
            if item.masked_fraction > maximum_included_masked_fraction
            and reviews[str(item.image_id)]["status"] != "exclude"
        }
        if over_limit:
            summary = ", ".join(
                f"{image_id}:{fraction:.2%}"
                for image_id, fraction in sorted(over_limit.items())
            )
            raise RuntimeError(
                "Reviewed masks above the configured discard threshold must be "
                f"explicitly excluded: {summary}"
            )
    excluded = sorted(
        item.source_image
        for item in dataset.items
        if reviews[str(item.image_id)]["status"] == "exclude"
    )
    mask_hashes = _current_mask_hashes(dataset)
    review_sha256 = _sha256(dataset.review_path) if dataset.review_path.is_file() else None
    immutable = {
        "mask_sha256": mask_hashes,
        "review_sha256": review_sha256,
        "image_inventory_sha256": canonical_hash(sorted(actual_images)),
        "excluded_images": excluded,
        "excluded_images_sha256": canonical_hash({"images": excluded}),
    }
    payload = {
        "schema_version": MASK_FINAL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **immutable,
        "finalization_sha256": canonical_hash(immutable),
        "counts": {
            "masks": len(mask_hashes),
            "excluded": len(excluded),
            "unreviewed": sum(
                review["status"] == "unreviewed" for review in reviews.values()
            ),
            "ok": sum(review["status"] == "ok" for review in reviews.values()),
        },
        "status": "passed",
    }
    target = path / "mask-final.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("finalization_sha256") != payload["finalization_sha256"]:
            raise RuntimeError(
                "Existing mask-final.json is immutable and no longer matches the masks/review"
            )
        return validate_mask_finalization(path, expected_images)
    _write_json(target, payload)
    return payload


def validate_mask_finalization(
    path: Path, expected_images: set[str] | None = None
) -> dict[str, Any]:
    target = path / "mask-final.json"
    if not target.is_file():
        raise MaskFinalizationMissingError(
            f"Mask review is not finalized for {path.name}; run gsstudio mask-finalize first"
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MASK_FINAL_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported mask-final schema: {payload.get('schema_version')}")
    dataset = MaskReviewDataset(path)
    actual_images = {item.source_image for item in dataset.items}
    if expected_images is not None and actual_images != expected_images:
        raise RuntimeError("Finalized mask image inventory no longer matches the run attempt")
    excluded = payload.get("excluded_images")
    if (
        not isinstance(excluded, list)
        or any(not isinstance(item, str) for item in excluded)
        or excluded != sorted(set(excluded))
        or not set(excluded).issubset(actual_images)
    ):
        raise RuntimeError("mask-final exclusion list is invalid")
    mask_hashes = _current_mask_hashes(dataset)
    review_sha256 = _sha256(dataset.review_path) if dataset.review_path.is_file() else None
    immutable = {
        "mask_sha256": mask_hashes,
        "review_sha256": review_sha256,
        "image_inventory_sha256": canonical_hash(sorted(actual_images)),
        "excluded_images": excluded,
        "excluded_images_sha256": canonical_hash(
            {"images": excluded}
        ),
    }
    for key, value in immutable.items():
        if payload.get(key) != value:
            raise RuntimeError(f"mask-final {key} is stale or corrupt")
    if canonical_hash(immutable) != payload.get("finalization_sha256"):
        raise RuntimeError("mask-final.json is stale or corrupt; finalize a new run")
    if immutable["excluded_images_sha256"] != payload.get("excluded_images_sha256"):
        raise RuntimeError("mask-final exclusion list hash is invalid")
    if payload.get("status") != "passed":
        raise RuntimeError("mask-final status is not passed")
    return payload

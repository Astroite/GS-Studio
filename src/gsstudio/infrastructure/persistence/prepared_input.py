"""Prepared input manifest and frame validation."""
from __future__ import annotations
import json
from pathlib import Path
import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image
from gsstudio.domain.capture import SourceProbe
from gsstudio.domain.prepared import CandidateFrameSet
from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.adapters.media import sha256_file


def _validate_frame(path: Path, width: int, height: int) -> str:
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xd8\xff":
            raise RuntimeError(f"Prepared frame is not a JPEG file: {path}")
    image = read_cv_image(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode prepared frame: {path}")
    if image.shape[1] != width or image.shape[0] != height:
        raise RuntimeError(
            f"Prepared frame has size {image.shape[1]}x{image.shape[0]}, expected {width}x{height}"
        )
    return sha256_file(path)


def load_prepared_input(path: Path) -> CandidateFrameSet:
    manifest_path = path / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("integrity") != "complete":
        raise RuntimeError(f"Prepared input is incomplete: {manifest_path}")
    records = payload.get("frames") or []
    lineage = payload["lineage"]
    if records != lineage.get("frames"):
        raise RuntimeError(f"Prepared input frame lineage is inconsistent: {manifest_path}")
    if len(records) != int(lineage.get("candidate_frame_count") or -1):
        raise RuntimeError(f"Prepared input frame count is invalid: {manifest_path}")
    preparation = {key: value for key, value in lineage.items() if key != "frames"}
    preparation_hash = canonical_hash(preparation)
    directory_hash = path.name.removesuffix(".building")
    if (
        payload.get("preparation_hash") != preparation_hash
        or directory_hash != preparation_hash
    ):
        raise RuntimeError(f"Prepared input preparation hash is invalid: {manifest_path}")
    relative_files = []
    for record in records:
        relative = Path(str(record["file"]))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise RuntimeError(f"Prepared input contains an unsafe frame path: {relative}")
        relative_files.append(str(relative))
    if len(set(relative_files)) != len(relative_files):
        raise RuntimeError(f"Prepared input contains duplicate frame paths: {manifest_path}")
    frames = tuple(path / relative_file for relative_file in relative_files)
    for frame, record in zip(frames, payload["frames"]):
        if not frame.is_file():
            raise RuntimeError(f"Prepared input frame is missing or corrupt: {frame}")
        actual_sha256 = _validate_frame(frame, int(record["width"]), int(record["height"]))
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(f"Prepared input frame is missing or corrupt: {frame}")
    expected = canonical_hash(lineage)
    if expected != payload.get("dataset_sha256"):
        raise RuntimeError("Prepared input dataset hash is invalid")
    return CandidateFrameSet(
        path=path,
        manifest_path=manifest_path,
        dataset_sha256=expected,
        frame_paths=frames,
        frame_sha256s=tuple(str(item["sha256"]) for item in records),
        frame_indices=tuple(int(item["source_frame_index"]) for item in payload["frames"]),
        timestamps_seconds=tuple(float(item["timestamp_seconds"]) for item in payload["frames"]),
        helper_version=lineage.get("helper_version"),
        sdk_version=lineage.get("sdk_version"),
        source_probe=SourceProbe.model_validate(lineage["source_probe"]),
        candidate_fps=float(lineage["sampling"]["candidate_fps"]),
    )

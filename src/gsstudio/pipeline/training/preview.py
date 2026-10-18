"""Experiment-local, atomic camera preview exchange with the desktop UI."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write

REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")


def read_request(output: Path, package_sha256: str, rows: list[dict]) -> dict | None:
    path = output / "preview-request.json"
    if not path.is_file():
        return None
    request = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(request, dict) or request.get("schema_version") != 1 or
            request.get("package_sha256") != package_sha256 or
            not isinstance(request.get("request_id"), str) or
            not REQUEST_ID.fullmatch(request["request_id"])):
        raise ValueError("Preview request identity is invalid")
    images = {row["image"]: row for row in rows}
    if request.get("image") not in images:
        raise ValueError("Preview camera is outside the verified training package")
    return {"request_id": request["request_id"], "row": images[request["image"]]}


def scaled_camera(row: dict, max_side: int = 640) -> dict:
    factor = min(1.0, max_side / max(row["width"], row["height"]))
    width = max(1, round(row["width"] * factor))
    height = max(1, round(row["height"] * factor))
    K = np.asarray(row["K"], dtype=float).copy()
    K[0] *= width / row["width"]
    K[1] *= height / row["height"]
    return {**row, "width": width, "height": height, "K": K.tolist()}


def publish(output: Path, *, request_id: str, row: dict, step: int, target: int,
            source_bgr: np.ndarray, render_rgb: np.ndarray) -> None:
    """Write the inactive bounded preview slot, then atomically switch the index."""
    if request_id != "default" and not REQUEST_ID.fullmatch(request_id):
        raise ValueError("Invalid preview request id")
    index = output / "preview.json"
    previous = None
    if index.is_file():
        try:
            previous = json.loads(index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    prior_slot = previous.get("slot") if isinstance(previous, dict) else None
    slot = 1 - prior_slot if prior_slot in (0, 1) else 0
    names = (f"preview-slot-{slot}-source.png", f"preview-slot-{slot}-render.png")
    for name, pixels in zip(names, (source_bgr, render_rgb[:, :, ::-1])):
        ok, encoded = cv2.imencode(".png", pixels)
        if not ok:
            raise RuntimeError("Could not encode training preview")
        temporary = output / f"{name}.tmp"
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(output / name)
    json_write(index, {
        "schema_version": 1, "slot": slot,
        "request_id": request_id, "image": row["image"],
        "split": row["split"], "step": step, "target": target,
        "source_file": names[0], "render_file": names[1],
        "source_sha256": sha256_file(output / names[0]),
        "render_sha256": sha256_file(output / names[1]),
    })

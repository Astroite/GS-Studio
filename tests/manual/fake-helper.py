#!/usr/bin/env python3
"""CI-only implementation of the GS Studio MediaSDK JSON protocol.

Each fake .insv needs a sibling ``.insv.probe.json``. This script never parses
INSV and must not be used as a production fallback.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


SUPPORTED = [
    "Insta360 ONE X",
    "Insta360 ONE R",
    "Insta360 ONE RS",
    "Insta360 ONE X2",
    "Insta360 X3",
    "Insta360 X4",
    "Insta360 X4 Air",
    "Insta360 X5",
    "Insta360 X6",
]


def host_path(value: str) -> Path:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if os.name != "nt" and match:
        return Path("/mnt") / match.group(1).lower() / match.group(2).replace("\\", "/")
    return Path(value)


def respond(path: Path, payload: dict[str, object]) -> int:
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if payload.get("ok") else 1


def main() -> int:
    request_path, response_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    base = {
        "schema_version": 1,
        "helper_version": "fake-1.0",
        "sdk_version": "fake-1.0",
        "supported_camera_models": SUPPORTED,
    }
    mode = os.environ.get("GSSTUDIO_FAKE_HELPER_MODE", "").strip()
    if mode == "no_response":
        return 7
    if mode == "wrong_schema":
        return respond(response_path, {**base, "schema_version": 99, "ok": True})
    if mode == "missing_versions":
        return respond(
            response_path,
            {
                "schema_version": 1,
                "ok": True,
                "supported_camera_models": SUPPORTED,
            },
        )
    if mode == "error":
        return respond(
            response_path,
            {
                **base,
                "ok": False,
                "error": {"code": "forced_failure", "message": "CI requested failure"},
            },
        )
    operation = request.get("operation")
    if operation == "capabilities":
        return respond(response_path, {**base, "ok": True})
    sources = [host_path(value) for value in request.get("source_files", [])]
    if not sources:
        return respond(
            response_path,
            {**base, "ok": False, "error": {"code": "missing_source", "message": "no source_files"}},
        )
    sidecar = sources[0].with_suffix(sources[0].suffix + ".probe.json")
    if not sidecar.is_file():
        return respond(
            response_path,
            {**base, "ok": False, "error": {"code": "fake_probe_missing", "message": str(sidecar)}},
        )
    media = json.loads(sidecar.read_text(encoding="utf-8"))
    if operation == "probe":
        return respond(response_path, {**base, "ok": True, "media": media})
    if operation == "export_frames":
        export_base = {
            **base,
            "helper_version": os.environ.get(
                "GSSTUDIO_FAKE_HELPER_EXPORT_VERSION", base["helper_version"]
            ),
        }
        output = request["output"]
        width, height = int(output["width"]), int(output["height"])
        fail_after_value = os.environ.get("GSSTUDIO_FAKE_HELPER_FAIL_AFTER", "").strip()
        fail_after = int(fail_after_value) if fail_after_value else None
        for position, item in enumerate(request["frames"]):
            if fail_after is not None and position >= fail_after:
                return respond(
                    response_path,
                    {
                        **export_base,
                        "ok": False,
                        "error": {
                            "code": "interrupted_export",
                            "message": f"stopped after {position} frames",
                        },
                    },
                )
            target = host_path(item["output_file"])
            target.parent.mkdir(parents=True, exist_ok=True)
            value = int(item["source_frame_index"]) % 255
            image = np.full((height, width, 3), value, dtype=np.uint8)
            if os.environ.get("GSSTUDIO_FAKE_HELPER_PNG_AS_JPEG"):
                encoded_ok, encoded = cv2.imencode(".png", image)
                if not encoded_ok:
                    raise RuntimeError(f"cannot encode {target}")
                target.write_bytes(encoded.tobytes())
            elif not cv2.imwrite(
                str(target),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, int(output["jpeg_quality"])],
            ):
                raise RuntimeError(f"cannot write {target}")
            if os.environ.get("GSSTUDIO_FAKE_HELPER_MUTATE_SOURCE"):
                sources[0].write_bytes(b"mutated by fake helper")
        return respond(
            response_path,
            {**export_base, "ok": True, "exported_count": len(request["frames"])},
        )
    return respond(
        response_path,
        {**base, "ok": False, "error": {"code": "unsupported_operation", "message": str(operation)}},
    )


if __name__ == "__main__":
    raise SystemExit(main())

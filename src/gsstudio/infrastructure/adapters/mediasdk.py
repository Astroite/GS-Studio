"""MediaSDK helper discovery, protocol and process adapter."""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from gsstudio.infrastructure.paths import host_path


MEDIA_PROTOCOL_VERSION = 1


MEDIA_HELPER_ENV = "GSSTUDIO_MEDIA_HELPER"


MEDIA_SDK_ROOT_ENV = "INSTA360_MEDIA_SDK_ROOT"


ALLOW_FAKE_HELPER_ENV = "GSSTUDIO_ALLOW_FAKE_MEDIA_HELPER"


PUBLIC_SUPPORTED_CAMERAS = {
    "insta360 one x",
    "insta360 one r",
    "insta360 one rs",
    "insta360 one x2",
    "insta360 x3",
    "insta360 x4",
    "insta360 x4 air",
    "insta360 x5",
    "insta360 x6",
}


class MediaHelperError(RuntimeError):
    """A structured failure reported by the MediaSDK helper itself."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"MediaSDK {code}: {message}")


def _camera_key(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    return normalized.removeprefix("insta360 ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _windows_visible(path: Path) -> str:
    value = str(path.absolute())
    if value.startswith("\\\\"):
        raise RuntimeError("MediaSDK request/output may not use a UNC path")
    return value


def _helper_visible(path: Path, helper: Path | None = None) -> str:
    selected = helper or media_helper_path()
    return str(path.absolute()) if selected.suffix.lower() == ".py" else _windows_visible(path)


def media_helper_path() -> Path:
    value = os.environ.get(MEDIA_HELPER_ENV, "").strip()
    if not value:
        raise RuntimeError(
            f"{MEDIA_HELPER_ENV} is not configured; legacy video and image-sequence "
            "inputs remain available"
        )
    path = host_path(value)
    if not path.is_file():
        raise RuntimeError(f"Configured MediaSDK helper is missing: {path}")
    if path.suffix.lower() == ".py":
        if os.environ.get(ALLOW_FAKE_HELPER_ENV) != "1":
            raise RuntimeError(
                "Python MediaSDK helpers are test-only; set GSSTUDIO_MEDIA_HELPER to the "
                "approved Windows x64 .exe"
            )
    elif path.suffix.lower() != ".exe":
        raise RuntimeError("GSSTUDIO_MEDIA_HELPER must be a Windows x64 .exe")
    return path


def invoke_media_helper(
    operation: str,
    payload: dict[str, Any],
    protocol_dir: Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    if operation not in {"capabilities", "probe", "export_frames"}:
        raise ValueError(f"Unsupported MediaSDK helper operation: {operation}")
    helper = media_helper_path()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request_path = protocol_dir / f"request-{token}.json"
    response_path = protocol_dir / f"response-{token}.json"
    request = {
        "schema_version": MEDIA_PROTOCOL_VERSION,
        "operation": operation,
        **payload,
    }
    _atomic_json(request_path, request)
    try:
        command = [
                *([sys.executable] if helper.suffix.lower() == ".py" else []),
                str(helper),
                _helper_visible(request_path, helper),
                _helper_visible(response_path, helper),
            ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        if not response_path.is_file():
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"MediaSDK helper exited with {result.returncode} without a response"
                + (f": {detail}" if detail else "")
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict):
            raise RuntimeError("MediaSDK helper response must be a JSON object")
        if response.get("schema_version") != MEDIA_PROTOCOL_VERSION:
            raise RuntimeError(
                f"Unsupported MediaSDK helper protocol: {response.get('schema_version')}"
            )
        for field in ("helper_version", "sdk_version"):
            if not isinstance(response.get(field), str) or not response[field].strip():
                raise RuntimeError(f"MediaSDK helper response omitted {field}")
        supported_models = response.get("supported_camera_models")
        if not isinstance(supported_models, list) or any(
            not isinstance(item, str) or not item.strip() for item in supported_models
        ):
            raise RuntimeError(
                "MediaSDK helper response has invalid supported_camera_models"
            )
        if response.get("ok", False) and not supported_models:
            raise RuntimeError("MediaSDK helper reports no supported camera models")
        if result.returncode != 0 or not response.get("ok", False):
            error = response.get("error") or {}
            code = error.get("code", "helper_failed")
            message = error.get("message", "MediaSDK helper failed")
            raise MediaHelperError(str(code), str(message))
        return response
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def media_capabilities(protocol_dir: Path) -> dict[str, Any]:
    return invoke_media_helper("capabilities", {}, protocol_dir)

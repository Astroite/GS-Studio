from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from gsstudio.domain.models import CaptureManifest


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_yaml(path))


def load_capture_manifest(path: Path) -> CaptureManifest:
    payload = load_yaml(path)
    if payload.get("schema_version") != 2:
        raise ValueError(f"Unsupported capture schema version: {payload.get('schema_version')}")
    return CaptureManifest.model_validate(payload)


def _yaml_data(model: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(model, BaseModel):
        return model.model_dump(mode="json", exclude_none=False)
    return model


def save_yaml(path: Path, model: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        _yaml_data(model),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def canonical_hash(value: BaseModel | dict[str, Any]) -> str:
    data = _yaml_data(value)
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

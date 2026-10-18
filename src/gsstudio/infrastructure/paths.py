from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path


def find_app_root(start: Path | None = None) -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        for candidate in (executable_dir, *executable_dir.parents):
            if (candidate / "release-manifest.json").is_file():
                return candidate
        return executable_dir
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path(__file__).resolve().parents[3]


DATA_ROOT_ENV = "GSSTUDIO_DATA_ROOT"
SETTINGS_DIR_ENV = "GSSTUDIO_SETTINGS_DIR"


def user_settings_dir() -> Path:
    configured = os.environ.get(SETTINGS_DIR_ENV, "").strip()
    if configured:
        return host_path(configured)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "GSStudio"
    return Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))) / "gsstudio"


def saved_data_root() -> Path | None:
    settings = user_settings_dir() / "settings.json"
    if not settings.is_file():
        return None
    payload = json.loads(settings.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("data_root"), str):
        raise RuntimeError(f"Invalid GS Studio settings: {settings}")
    return host_path(payload["data_root"])


def save_data_root(path: Path) -> Path:
    root = host_path(path).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    settings = user_settings_dir() / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=settings.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump({"schema_version": 1, "data_root": str(root)}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, settings)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return root


def find_data_root(start: Path | None = None) -> Path:
    """Locate Data via explicit override, saved user choice, or a sibling folder."""
    configured = os.environ.get(DATA_ROOT_ENV, "").strip()
    if configured:
        candidate = host_path(configured)
    else:
        candidate = saved_data_root()
        if candidate is None:
            from gsstudio.infrastructure.runtime.configuration import runtime_config
            candidate = Path(runtime_config()['data_root'])
    if candidate.is_dir():
        return candidate
    raise RuntimeError(f"GS Studio data root is missing: {candidate}. Select an existing Data folder in the desktop app or set {DATA_ROOT_ENV}.")


def host_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def _normalized(path: Path) -> Path:
    """Absolute path with ``..`` removed but symlinks left intact.

    ``Path.resolve`` would follow a symlink out of its containing directory and
    reject every path under it, so containment is checked lexically instead.
    Traversal escapes are still blocked because ``normpath`` collapses ``..``
    before the comparison.
    """
    return Path(os.path.normpath(os.path.abspath(path)))


def ensure_within(path: Path, parent: Path) -> Path:
    normalized = _normalized(path)
    base = _normalized(parent)
    try:
        normalized.relative_to(base)
    except ValueError as error:
        raise ValueError(f"Path must stay within {base}: {normalized}") from error
    return normalized


def ensure_run_dir(scene_path: Path, run_id: str, exist_ok: bool = True) -> Path:
    """Return ``<scene>/<run-id>``, the single directory holding a run's manifest and scratch."""
    run_dir = scene_path / run_id
    run_dir.mkdir(parents=True, exist_ok=exist_ok)
    return run_dir


def location_dir(root: Path, location_id: str) -> Path:
    return root / location_id


def scene_dir(root: Path, location_id: str, scene_id: str) -> Path:
    return location_dir(root, location_id) / scene_id

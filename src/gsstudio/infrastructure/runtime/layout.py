"""Commands and executable paths for source and frozen Windows installations."""
from __future__ import annotations

import sys
from pathlib import Path

from gsstudio.infrastructure.paths import find_app_root


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def worker_command() -> list[str]:
    if not is_frozen():
        return [sys.executable, "-m", "gsstudio.interfaces.worker.main"]
    executable = find_app_root() / "worker" / "GSStudioWorker.exe"
    if not executable.is_file():
        raise RuntimeError(f"GS Studio operation worker is missing: {executable}")
    return [str(executable)]


def trainer_executable() -> Path:
    executable = find_app_root() / "gpu-runtime" / "GSStudioTrainer.exe"
    if not executable.is_file():
        raise RuntimeError(f"GS Studio GPU runtime is missing: {executable}")
    return executable


def configure_release_path() -> None:
    """Use the release's command-line helpers and pretrained model cache."""
    if not is_frozen():
        return
    import os

    from gsstudio.infrastructure.runtime.model_cache import require_model_cache

    root = find_app_root()
    models = root / "model-cache"
    require_model_cache(models)
    os.environ["TORCH_HOME"] = str(models)
    tools = root / "tools" / "ffmpeg"
    if tools.is_dir():
        os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")

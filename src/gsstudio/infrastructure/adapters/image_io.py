"""OpenCV codecs with Python file I/O for native Windows Unicode paths."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


def read_cv_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        encoded = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, flags) if encoded.size else None


def atomic_imwrite(target: Path, image: np.ndarray, parameters: list[int] | None = None) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(target.suffix, image, parameters or [])
    if not encoded_ok:
        raise RuntimeError(f"Failed to encode image for {target}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_cv_image(path: str | Path, image: np.ndarray, parameters: list[int] | None = None) -> bool:
    """Write completely or raise; a successful codec cannot hide a lost file."""
    atomic_imwrite(Path(path), image, parameters)
    return True

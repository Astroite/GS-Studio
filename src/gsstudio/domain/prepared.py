"""Identity of one verified Run-owned prepared frame set."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from gsstudio.domain.capture import SourceProbe


@dataclass(frozen=True)
class CandidateFrameSet:
    path: Path
    manifest_path: Path
    dataset_sha256: str
    frame_paths: tuple[Path, ...]
    frame_sha256s: tuple[str, ...]
    frame_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    helper_version: str | None = None
    sdk_version: str | None = None
    source_probe: SourceProbe | None = None
    candidate_fps: float | None = None

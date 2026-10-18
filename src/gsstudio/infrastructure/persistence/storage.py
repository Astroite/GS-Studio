"""Verified immutable materialization; adapters must never edit a shared inode."""
import os
import shutil
from pathlib import Path
from gsstudio.infrastructure.adapters.media import sha256_file


def link_or_copy(source: Path, target: Path) -> dict:
    digest = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != digest:
            raise RuntimeError(f'Conflicting immutable file: {target}')
        mode = 'hardlink' if os.path.samefile(source, target) else 'copy'
    else:
        try:
            os.link(source, target)
            mode = 'hardlink'
        except OSError:
            shutil.copy2(source, target)
            mode = 'copy'
        if sha256_file(target) != digest:
            raise RuntimeError(f'Copy verification failed: {target}')
    return dict(mode=mode, bytes=target.stat().st_size,
                duplicate_bytes=target.stat().st_size if mode == 'copy' else 0)

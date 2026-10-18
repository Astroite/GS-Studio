"""Path containment and reparse-point checks for Run-owned files."""
from __future__ import annotations
import os
import stat
from pathlib import Path, PurePosixPath


def safe_path(work: Path, relative: str) -> Path:
    parts = PurePosixPath(relative.replace('\\', '/')).parts
    if not parts or '..' in parts or ':' in relative or PurePosixPath(relative).is_absolute():
        raise ValueError(f'Unsafe retention path: {relative}')
    root = Path(os.path.abspath(work))
    target = root.joinpath(*parts)
    target.relative_to(root)
    for p in (root, *root.parents, *[root.joinpath(*parts[:i]) for i in range(1, len(parts)+1)]):
        if p.is_symlink() or (p.exists() and getattr(p.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError(f'Retention refuses links/junctions: {p}')
    return target

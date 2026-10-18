"""One-time, explicitly scoped retirement audit; never loads historical Runs.

This supplements retention's intermediate allowlist for the user's 2026-10-16
replacement migration. It is not a compatibility reader or a routine CLI.
The coordinator must supply a reviewed plan, acceptance and consumer evidence.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import re
import stat
from gsstudio.infrastructure.path_safety import safe_path
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.pipeline.retention import _disposable


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def disposable(relative):
    # Keep records, configs, QA and source media even when retiring old results.
    # Do not broaden to arbitrary files under a familiar directory name.
    if _disposable(relative):
        return True
    return bool(re.fullmatch(
        r'equirect-(?:fallback|repair|selected)/frame_\d+\.jpg'
        r'|reconstruction-(?:primary|repair)/sparse_pc\.ply'
        r'|reconstruction-(?:primary|repair)/colmap/(?:cameras|images|points3D)\.bin'
        r'|reconstruction-(?:primary|repair)/colmap/project/(?:animation|appConfig|controlpoints|entities|metadata|sfm)\d+\.dat'
        r'|training-data/segment-\d+/(?:points\.npz|colmap/(?:cameras|images|points3D)\.bin|images/image_\d+\.jpg|masks/image_\d+\.jpg\.png)'
        r'|experiments/(?:postshot-longest-photo-on|postshot-photo-on)/postshot-input/(?:colmap/(?:cameras|images|points3D)\.bin|images/image_\d+\.jpg|masks/image_\d+\.png)'
        r'|experiments/gsplat-photo-on/(?:checkpoint\.pt|model\.ply|diagnostic-sh0\.ply|train\.log)'
        r'|experiments/postshot-photo-on/train\.log'
        r'|logs/(?:mask\.log|reconstruct-(?:primary|repair)/realityscan-align\.log)', relative))


def key(path):
    return str(Path(path).absolute()).replace('\\', '/').casefold()


def guard(root, entry, protected):
    relative = entry['relative']
    if not disposable(relative):
        raise ValueError('Not an allowlisted generated file: ' + relative)
    path = safe_path(root, relative)
    if key(path) in protected:
        raise ValueError('Capture/source protection: ' + str(path))
    info = path.stat()
    if not path.is_file() or info.st_size != entry['bytes'] or [info.st_dev, info.st_ino] != entry['inode']:
        raise ValueError('Target identity changed: ' + str(path))
    if digest(path) != entry['sha256']:
        raise ValueError('Target hash changed: ' + str(path))
    return path


def external_consumers(scan_roots, retired_roots):
    """Conservative references: any Run ID mention outside retirement blocks it."""
    retired = [Path(p).absolute() for p in retired_roots]
    scan_roots = [Path(p).absolute() for p in scan_roots]
    found, scanned, aliases = [], [], []
    def link_reference(p):
        info = p.lstat()
        if not (p.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            return False
        target = p.resolve(strict=True)
        affected = [r for r in retired if target == r or target.is_relative_to(r) or r.is_relative_to(target)]
        if affected:
            found.extend({'consumer': str(p), 'retired_root': str(r), 'kind': 'link_target'} for r in affected)
        elif not any(target == s or target.is_relative_to(s) for s in scan_roots):
            raise ValueError('Consumer link leaves audited roots; retain and expand scope: ' + str(p))
        aliases.append({'path': str(p), 'target': str(target), 'mode': 'reference recorded; no traversal; target covered by scan or retirement scope'})
        return True
    for scan in scan_roots:
        scan = Path(scan).absolute()
        for parent, dirs, files in os.walk(scan, followlinks=False):
            dirs[:] = [d for d in dirs if Path(parent, d).absolute() not in retired]
            for d in dirs[:]:
                p = Path(parent) / d
                if link_reference(p):
                    dirs.remove(d)
                else:
                    safe_path(scan, p.relative_to(scan).as_posix())
            for name in files:
                if link_reference(Path(parent) / name):
                    continue
                if Path(name).suffix.lower() not in {'.json', '.jsonl', '.yaml', '.yml'}:
                    continue
                p = safe_path(scan, (Path(parent) / name).relative_to(scan).as_posix())
                h = hashlib.sha256()
                hits = set()
                # Chunk overlap catches IDs split across read boundaries.
                tail = b''
                with p.open('rb') as f:
                    for data in iter(lambda: f.read(1024**2), b''):
                        h.update(data)
                        hay = (tail + data).lower()
                        for r in retired:
                            if r.name.lower().encode() in hay:
                                hits.add(str(r))
                        tail = hay[-128:]
                scanned.append({'path': str(p), 'sha256': h.hexdigest()})
                found.extend({'consumer': str(p), 'retired_root': r} for r in sorted(hits))
    return {'scanned': scanned, 'references': found, 'covered_link_aliases': aliases}


def execute_run(root, root_boundary, manifest_sha256, entries, protected, journal, consumer_check):
    """Journal each verified unlink; no recursive delete, missing files fail closed.

    Every invocation rechecks the Run lock, manifest and external consumers.
    The caller owns full source rehash and completed replacement acceptance.
    """
    root, root_boundary = Path(root).absolute(), Path(root_boundary).absolute()
    if root == root_boundary or not root.is_relative_to(root_boundary):
        raise ValueError('Retirement root escapes the explicitly named Data boundary')
    safe_path(root_boundary, root.relative_to(root_boundary).as_posix())
    with run_session(root):
        if digest(safe_path(root, 'manifest.yaml')) != manifest_sha256:
            raise ValueError('Retired manifest changed')
        consumer_check(root)
        # Verify the complete batch before deleting its first file.
        for entry in entries:
            guard(root, entry, protected)
        with Path(journal).open('x', encoding='utf-8') as log:
            for entry in entries:
                path = guard(root, entry, protected)
                event = {'root': str(root), **entry, 'state': 'verified_unlink_intent'}
                log.write(json.dumps(event, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
                path.unlink()
                event['state'] = 'deleted'
                log.write(json.dumps(event, ensure_ascii=False) + '\n')
                log.flush()
                os.fsync(log.fileno())
    # Metadata and directory structure intentionally remain as retirement audit.

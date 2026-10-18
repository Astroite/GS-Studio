"""Schema 6 retention: verified consumers, allowlisted files, durable deletion journal."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path, PurePosixPath
import stat
from gsstudio.infrastructure.path_safety import safe_path
from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write




def _files(work, relative):
    root = safe_path(work, relative)
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    result = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = safe_path(work, (Path(base)/name).relative_to(work).as_posix())
            if p.is_file():
                result.append(p)
    return result


def _disposable(relative):
    """Unknown files (including raw media placed in scratch) are never owned."""
    if re.fullmatch(r'reconstruction-(?:primary|repair)/colmap/project\.rsproj\.data/.+', relative):
        return Path(relative).suffix.lower() not in {'.insv','.mp4','.mov','.mkv','.avi','.ply','.psht','.pt','.ckpt'}
    return bool(re.fullmatch(
        r'(?:inputs/primary/[0-9a-f]{64}/frames|inputs/repair/frames|frames-primary)/frame_\d+\.jpg'
        # Panoramas stage as view_XX/; a perspective source is one frame per view.
        r'|(?:reconstruction-(?:primary|repair)|repair-new)/(?:images|masks)/(?:view_\d+/)?frame_\d+\.jpg(?:\.png)?'
        r'|reconstruction-(?:primary|repair)/alignment-images/gsdb_\d+\.jpg(?:\.mask\.png)?'
        r'|reconstruction-(?:primary|repair)/colmap/project\.rsproj', relative))


def _validated_masks(dataset, run):
    from gsstudio.pipeline.masks.masking import validate_mask_filter
    from gsstudio.pipeline.masks.finalize import validate_mask_finalization
    filtered = validate_mask_filter(dataset, verify_hashes=True)
    if float(filtered['threshold']) != run.config.masking.mask_discard_threshold:
        raise RuntimeError('Mask threshold changed')
    included = {r['image'] for r in filtered['accepted']}
    if included:
        final = validate_mask_finalization(dataset, included)
        included -= set(final.get('excluded_images', []))
    return included


def eligible_roots(work, run, checkpoint=False):
    """Only registered disposable directories; never glob arbitrary caches/projects."""
    roots, evidence, retained = [], {}, []
    if run.config.retention.mode == 'keep':
        return [], {}, ['retention.mode=keep']
    if run.active_stage and not (checkpoint and run.active_stage == 'reconstruct'):
        return [], {}, ['An active/unfinished stage still owns its inputs']
    if run.stages['mask'].status.value == 'succeeded':
        dataset = safe_path(work, 'reconstruction-primary')
        # After the losing attempt is retired, selected QA is the consumer proof.
        retired = work/'retention-retired.json'
        retired_names = json.loads(retired.read_text(encoding='utf-8'))['attempts'] if retired.exists() else []
        if 'primary' not in retired_names or run.stages['reconstruct'].status.value != 'succeeded':
            if run.config.masking.mask_review_required and not (dataset/'mask-final.json').exists():
                return [], {}, ['Awaiting primary mask finalization']
            _validated_masks(dataset, run)
        evidence['mask'] = 'validated final input inventory'
        roots += [run.config.input_relative_path + '/frames', 'frames-primary']
    repair = work/'reconstruction-repair'
    repair_ready = (repair/'repair-input.json').is_file() and (
        (repair/'mask-final.json').is_file() or not json.loads((repair/'mask-filter.json').read_text(encoding='utf-8'))['accepted'])
    if repair_ready:
        retired = work/'retention-retired.json'
        retired_names = json.loads(retired.read_text(encoding='utf-8'))['attempts'] if retired.exists() else []
        if 'repair' not in retired_names:
            _validated_masks(repair, run)
        roots += ['inputs/repair/frames']
        evidence['repair'] = sha256_file(repair/'repair-input.json')
    if run.stages['reconstruct'].status.value == 'succeeded' and run.selected_dataset:
        from gsstudio.pipeline.quality.segments import validate_segments
        label = run.metrics['selected_attempt']
        if label not in ('primary', 'repair') or run.selected_dataset != f'{run.id}/reconstruction-{label}':
            raise RuntimeError('Invalid selected dataset identity')
        dataset = safe_path(work, f'reconstruction-{label}')
        included = _validated_masks(dataset, run)
        records = [json.loads(s) for s in (work/f'selected-{label}-metrics.jsonl').read_text(encoding='utf-8').splitlines() if s]
        validate_segments(dataset, records, included, run.config.reconstruction.primary, run.config.segment_qa)
        evidence['segments'] = sha256_file(dataset/'segments.json')
        for attempt in ('primary', 'repair'):
            base = f'reconstruction-{attempt}'
            roots += [f'{base}/alignment-images', f'{base}/colmap/project.rsproj',
                      f'{base}/colmap/project.rsproj.data', f'{base}/colmap/text-export']
            if attempt != label:
                roots += [f'{base}/images', f'{base}/masks']
        roots += ['repair-new/images', 'repair-new/masks']
    retained += ['Raw sources, manifests, logs, QA and lineage records',
                 'Selected filtered images/masks, COLMAP, training inputs, checkpoints and results',
                 'Global caches, other Runs and external output directories']
    return roots, evidence, retained


def cleanup_run(scene: Path, run, apply=False, *, checkpoint=False):
    from gsstudio.infrastructure.runtime.run_lock import run_session
    from gsstudio.infrastructure.persistence.run_repository import load_run
    work = scene/run.id
    with run_session(work):
        run = load_run(scene, run.id)
        roots, evidence, retained = eligible_roots(work, run, checkpoint=checkpoint)
        candidates = {p for root in roots for p in _files(work, root)}
        paths = sorted(p for p in candidates if _disposable(p.relative_to(work).as_posix()))
        retained += [f'Unrecognized file: {p.relative_to(work).as_posix()}' for p in sorted(candidates-set(paths))]
        entries = []
        for path in paths:
            info = path.stat()
            entries.append(dict(path=path.relative_to(work).as_posix(), sha256=sha256_file(path),
                bytes=info.st_size, inode=[info.st_dev, info.st_ino], links=info.st_nlink, status='planned'))
        groups = {}
        for e in entries:
            groups.setdefault(tuple(e['inode']), []).append(e)
        status_counts = {}
        for entry in entries:
            status_counts[entry['status']] = status_counts.get(entry['status'], 0) + 1
        logical_bytes = sum(e['bytes'] for e in entries)
        reclaimable_bytes = sum(g[0]['bytes'] for g in groups.values() if len(g) >= g[0]['links'])
        report = dict(schema_version=1, config_hash=run.config_hash, roots=roots,
            trigger='verified downstream commit', consumer_verification=evidence, files=entries,
            planned_files=len(entries),
            status_counts=status_counts,
            logical_bytes=logical_bytes,
            reclaimable_bytes=reclaimable_bytes,
            logical_gib=round(logical_bytes / 1024**3, 3),
            reclaimable_gib=round(reclaimable_bytes / 1024**3, 3),
            hardlink_note=(
                'logical_bytes double-counts hardlinked copies; reclaimable_bytes is '
                'what unlinking each inode group once would free'
            ),
            retained=retained, status='preview')
        if not apply:
            return report
        journal = safe_path(work, 'cleanup')
        journal.mkdir(exist_ok=True)
        # Publish retirement BEFORE the first unlink. A crash halfway through
        # removing the losing inventory must not require those bytes on resume.
        if run.stages['reconstruct'].status.value == 'succeeded' and roots:
            retired = [a for a in ('primary','repair') if a != run.metrics['selected_attempt']
                       and (work/f'reconstruction-{a}').exists()]
            json_write(work/'retention-retired.json', dict(attempts=retired, status='audit_only',
                config_hash=run.config_hash, selected=run.metrics['selected_attempt'], consumer_verification=evidence))
        # Continue interrupted plans before adding a new one. Missing files are
        # trusted only when their deletion was durably planned under this config.
        for old in sorted(journal.glob('*.json')):
            safe_path(work, old.relative_to(work).as_posix())
            previous = json.loads(old.read_text(encoding='utf-8'))
            if previous['status'] == 'complete':
                continue
            if previous['config_hash'] != run.config_hash:
                raise RuntimeError('Cleanup journal config identity changed')
            _execute(work, old, previous, roots)
        if entries:
            identity = canonical_hash(dict(config=run.config_hash, evidence=evidence, files=entries))
            destination = journal/f'{identity}.json'
            if destination.exists():
                saved = json.loads(destination.read_text(encoding='utf-8'))
                if saved['status'] == 'complete':
                    return saved
            report['status'] = 'planned'
            json_write(destination, report)
            _execute(work, destination, report, roots)
        else:
            report['status'] = 'complete'
        return report


def _execute(work, destination, report, allowed):
    for entry in report['files']:
        relative = entry['path']
        if not _disposable(relative) or not any(relative == root or relative.startswith(root + '/') for root in allowed):
            raise RuntimeError(f'Cleanup dependency no longer satisfied: {relative}')
        path = safe_path(work, relative)
        if entry['status'] == 'deleted':
            if path.exists():
                raise RuntimeError(f'Previously deleted file reappeared: {path}')
            continue
        if not path.exists():
            entry['status'] = 'deleted'
        else:
            if sha256_file(path) != entry['sha256']:
                raise RuntimeError(f'Cleanup target changed: {path}')
            try:
                path.unlink()
                entry['status'] = 'deleted'
                entry.pop('error', None)
            except OSError as error:
                entry.update(status='pending', error=str(error))
        json_write(destination, report)
    report['status'] = 'complete' if all(e['status']=='deleted' for e in report['files']) else 'pending'
    counts = {}
    for entry in report.get('files', []):
        counts[entry['status']] = counts.get(entry['status'], 0) + 1
    report['status_counts'] = counts
    report['planned_files'] = len(report.get('files', []))
    json_write(destination, report)


def auto_cleanup(scene, run, *, checkpoint=False):
    if run.config.retention.mode == 'keep':
        return
    try:
        result = cleanup_run(scene, run, apply=True, checkpoint=checkpoint)
        print(
            f"Cleanup: {result['status']}; planned={result.get('planned_files', len(result.get('files', [])))}; "
            f"counts={result.get('status_counts', {})}; "
            f"logical={result.get('logical_gib', result['logical_bytes'] / 1024**3):.3f} GiB, "
            f"reclaimable={result.get('reclaimable_gib', result['reclaimable_bytes'] / 1024**3):.3f} GiB",
            flush=True,
        )
        pending = scene/run.id/'cleanup-pending.json'
        if pending.exists() and result['status'] == 'complete':
            previous = json.loads(pending.read_text(encoding='utf-8'))
            if previous.get('status') != 'complete':
                json_write(pending, dict(status='complete', previous=previous))
    except (OSError, ValueError, RuntimeError) as error:
        # Cleanup is maintenance after a committed computation, never a reason
        # to rewrite a succeeded compute stage to failed.
        json_write(scene/run.id/'cleanup-pending.json', dict(status='pending', error=str(error)))
        print(f'Cleanup pending: {error}', flush=True)

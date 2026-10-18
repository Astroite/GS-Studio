"""Destructive-migration guards run only against temporary data."""
from pathlib import Path
import importlib.util
import json
import multiprocessing
import os
import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/migration_retirement.py'
spec = importlib.util.spec_from_file_location('migration_retirement', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_run(tmp_path):
    root = tmp_path / '20260912T090102Z-0c3b75e2'
    root.mkdir()
    (root / 'manifest.yaml').write_text('id: example\n', encoding='utf-8')
    p = root / 'experiments/gsplat-photo-on/model.ply'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'generated fixture')
    s = p.stat()
    return root, p, {'relative': p.relative_to(root).as_posix(), 'bytes': s.st_size,
                     'inode': [s.st_dev, s.st_ino], 'sha256': m.digest(p)}


@pytest.mark.parametrize('relative', ['../raw.jpg', 'D:/outside/model.ply', 'source-sequences/frames/frame_1.jpg',
    'experiments/gsplat-photo-on/original.insv', 'reconstruction-primary/images/raw.jpg',
    'training-data/segment-001/config.json', 'manifest.yaml'])
def test_unknown_and_source_paths_are_never_disposable(relative):
    assert not m.disposable(relative)


def test_hash_change_refuses_entire_batch(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    p.write_bytes(b'changed content!!')
    with pytest.raises(ValueError):
        m.execute_run(root, tmp_path, m.digest(root/'manifest.yaml'), [entry], set(), tmp_path/'log', lambda _: None)
    assert p.exists() and not (tmp_path/'log').exists()


def test_protected_capture_path_is_retained(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    with pytest.raises(ValueError, match='protection'):
        m.guard(root, entry, {m.key(p)})
    assert p.exists()


def test_consumer_blocks_delete_and_is_scanned(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    consumer = tmp_path/'capture.json'
    consumer.write_text(json.dumps({'source': str(p)}), encoding='utf-8')
    refs = m.external_consumers([tmp_path], [root])
    assert refs['references'] == [{'consumer': str(consumer), 'retired_root': str(root)}]
    def check(_):
        if refs['references']: raise ValueError('external consumer')
    with pytest.raises(ValueError, match='external consumer'):
        m.execute_run(root, tmp_path, m.digest(root/'manifest.yaml'), [entry], set(), tmp_path/'log', check)
    assert p.exists()


def test_manifest_change_and_root_escape(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    for boundary, sha in [(tmp_path, '0'*64), (tmp_path/'elsewhere', m.digest(root/'manifest.yaml'))]:
        with pytest.raises(ValueError):
            m.execute_run(root, boundary, sha, [entry], set(), tmp_path/'log', lambda _: None)
    assert p.exists()


def test_symlink_is_refused(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    outside = tmp_path/'source.ply'
    outside.write_bytes(p.read_bytes())
    p.unlink()
    try:
        p.symlink_to(outside)
    except OSError:
        pytest.skip('Windows symlink privilege unavailable; junction tested separately')
    with pytest.raises(ValueError, match='links/junctions'):
        m.guard(root, entry, set())
    assert outside.exists()


def _hold_lock(root, ready, release):
    from gsstudio.infrastructure.runtime.run_lock import run_session
    with run_session(Path(root)):
        ready.set()
        release.wait(15)


def test_cross_process_busy_run_refuses_delete(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    ctx = multiprocessing.get_context('spawn')
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_hold_lock, args=(str(root), ready, release))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeError, match='busy'):
            m.execute_run(root, tmp_path, m.digest(root/'manifest.yaml'), [entry], set(), tmp_path/'log', lambda _: None)
        assert p.exists()
    finally:
        release.set(); process.join(10)
        if process.is_alive(): process.terminate(); process.join()


def test_exact_unlink_preserves_hardlinked_original_and_records(tmp_path):
    root, p, entry = fixture_run(tmp_path)
    original = tmp_path/'source.ply'
    os.link(p, original)
    unknown = root/'experiments/gsplat-photo-on/private-note.txt'
    unknown.write_text('keep', encoding='utf-8')
    m.execute_run(root, tmp_path, m.digest(root/'manifest.yaml'), [entry], {m.key(original)}, tmp_path/'log', lambda _: None)
    assert not p.exists() and m.digest(original) == entry['sha256']
    assert unknown.exists() and (root/'manifest.yaml').exists()
    assert [json.loads(x)['state'] for x in (tmp_path/'log').read_text(encoding='utf-8').splitlines()] == ['verified_unlink_intent','deleted']


def test_consumer_alias_is_not_followed_and_retired_target_is_blocked(tmp_path):
    root, p, _ = fixture_run(tmp_path)
    live = tmp_path/'live';live.mkdir()
    (live/'project.json').write_text('{}',encoding='utf-8')
    try:
        (tmp_path/'live_current').symlink_to(live,target_is_directory=True)
        (tmp_path/'retired_reference').symlink_to(root,target_is_directory=True)
    except OSError:
        pytest.skip('Windows symlink privilege unavailable')
    result=m.external_consumers([tmp_path],[root])
    assert len(result['covered_link_aliases'])==2
    assert len(result['scanned'])==1
    assert result['references']==[{'consumer':str(tmp_path/'retired_reference'),'retired_root':str(root),'kind':'link_target'}]
    assert p.exists()


def test_uncovered_consumer_link_fails_closed(tmp_path):
    scan=tmp_path/'scan';scan.mkdir()
    outside=tmp_path/'outside';outside.mkdir()
    try:(scan/'unknown').symlink_to(outside,target_is_directory=True)
    except OSError:pytest.skip('Windows symlink privilege unavailable')
    with pytest.raises(ValueError,match='leaves audited roots'):
        m.external_consumers([scan],[])

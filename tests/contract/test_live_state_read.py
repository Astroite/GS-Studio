"""CPU live-state read contracts, including independent native Windows processes."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from gsstudio.application import studio
from gsstudio.infrastructure.persistence import live_json

REPO = Path(__file__).resolve().parents[2]


def win_error(code):
    error = PermissionError(errno.EACCES, 'injected Windows sharing/access denial')
    error.winerror = code
    return error


def fake_clock(monkeypatch):
    clock = [0.0]
    sleeps = []
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0])
    def sleep(delay):
        assert 0 < delay <= .05
        sleeps.append(delay)
        clock[0] += delay
    monkeypatch.setattr(time, 'sleep', sleep)
    return clock, sleeps


@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_windows_read_retries_then_returns_complete_json(tmp_path, monkeypatch, code):
    path = tmp_path/'training.json'
    path.write_text('{"step": 4}', encoding='utf-8')
    original = lambda path: path.read_text(encoding='utf-8')
    calls = []
    clock, sleeps = fake_clock(monkeypatch)
    monkeypatch.setattr(live_json, '_WINDOWS', True)
    def read(path):
        calls.append(path)
        if len(calls) < 3:
            raise win_error(code)
        return original(path)
    monkeypatch.setattr(live_json, '_read_text', read)
    assert studio._optional_json(path) == {'step': 4}
    assert len(calls) == 3 and len(sleeps) == 2 and clock[0] == pytest.approx(.03)


@pytest.mark.parametrize('code', [5, 32, 33])
def test_persistent_windows_read_error_is_bounded_and_never_none(tmp_path, monkeypatch, code):
    error = win_error(code)
    calls = []
    clock, sleeps = fake_clock(monkeypatch)
    monkeypatch.setattr(live_json, '_WINDOWS', True)
    def denied(path):
        calls.append(path)
        raise error
    monkeypatch.setattr(live_json, '_read_text', denied)
    with pytest.raises(PermissionError) as caught:
        studio._optional_json(tmp_path/'training.json')
    assert caught.value is error and len(calls) > 1 and sleeps
    assert clock[0] == pytest.approx(.5)


@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EIO, errno.EACCES])
def test_nontransient_read_failure_is_not_retried_or_optional(tmp_path, monkeypatch, code):
    error = OSError(code, 'injected IO failure without transient WinError')
    calls = []
    def fail(path):
        calls.append(path)
        raise error
    monkeypatch.setattr(live_json, '_read_text', fail)
    monkeypatch.setattr(time, 'sleep', lambda delay: pytest.fail('Nontransient read was retried'))
    with pytest.raises(OSError) as caught:
        studio._optional_json(tmp_path/'training.json')
    assert caught.value is error and len(calls) == 1


def test_only_absent_optional_files_return_none_required_dispatch_still_fails(tmp_path):
    assert studio._optional_json(tmp_path/'training.json') is None
    with pytest.raises(FileNotFoundError):
        studio._experiment({'root': str(tmp_path), 'output': str(tmp_path)})


@pytest.mark.skipif(os.name != 'nt', reason='Windows transient read retries')
def test_training_state_poll_retries_without_relabeling_or_filling_missing_state(tmp_path, monkeypatch):
    dispatch = {'backend': 'gsplat', 'package_sha256': '0'*64, 'status': 'running'}
    (tmp_path/'dispatch.json').write_text(json.dumps(dispatch), encoding='utf-8')
    training = {'status': 'running', 'completed_steps': 4, 'text': '完整'*40000}
    (tmp_path/'training.json').write_text(json.dumps(training, ensure_ascii=False), encoding='utf-8')
    original = live_json._read_text
    calls = []
    def read(path):
        if path.name == 'training.json':
            calls.append(path)
            if len(calls) < 3:
                raise win_error(32)
        return original(path)
    monkeypatch.setattr(live_json, '_read_text', read)
    handled, result = studio.execute_additional({'action': 'training_state',
        'root': str(tmp_path), 'output': str(tmp_path)})
    assert handled and result['training'] == training and result['dispatch'] == dispatch
    assert result['state'] is None and result['config'] is None and result['metrics'] is None
    assert len(calls) == 3


@pytest.mark.parametrize('contents', [b'', b'{"step":', b'{"step":4}garbage', b'\xff'])
def test_truncated_corrupt_or_invalid_utf8_state_is_never_optional(tmp_path, monkeypatch, contents):
    path = tmp_path/'training.json'
    path.write_bytes(contents)
    monkeypatch.setattr(time, 'sleep', lambda delay: pytest.fail('Corruption was retried'))
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        studio._optional_json(path)
    # A required mutable manifest must fail in exactly the same way.
    (tmp_path/'dispatch.json').write_bytes(contents)
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        studio._experiment({'root': str(tmp_path), 'output': str(tmp_path)})


def test_nonwindows_read_never_retries_windows_error(tmp_path, monkeypatch):
    error = win_error(32)
    monkeypatch.setattr(live_json, '_WINDOWS', False)
    def fail(path):
        raise error
    monkeypatch.setattr(live_json, '_read_text', fail)
    monkeypatch.setattr(time, 'sleep', lambda delay: pytest.fail('POSIX read was retried'))
    with pytest.raises(PermissionError) as caught:
        studio._optional_json(tmp_path/'training.json')
    assert caught.value is error


@pytest.mark.skipif(os.name != 'nt', reason='Native Windows error conversion and handle closure')
@pytest.mark.parametrize('stage,code', [('open', 32), ('read', 33)])
def test_native_adapter_preserves_error_code_retries_and_closes_handles(tmp_path, monkeypatch, stage, code):
    import ctypes
    path = tmp_path/'training.json'
    path.write_text('{"step": 4}', encoding='utf-8')
    original_create, original_read = live_json._create_file, live_json._read_file
    original_close = live_json._close_handle
    calls, closed = [], []
    def create(*args):
        if stage == 'open':
            calls.append(1)
            if len(calls) == 1:
                ctypes.set_last_error(code)
                return live_json._INVALID_HANDLE
        return original_create(*args)
    def read(*args):
        if stage == 'read':
            calls.append(1)
            if len(calls) == 1:
                ctypes.set_last_error(code)
                return False
        return original_read(*args)
    def close(handle):
        closed.append(handle)
        return original_close(handle)
    monkeypatch.setattr(live_json, '_create_file', create)
    monkeypatch.setattr(live_json, '_read_file', read)
    monkeypatch.setattr(live_json, '_close_handle', close)
    assert studio._optional_json(path) == {'step': 4}
    assert len(calls) >= 2 and len(closed) == (1 if stage == 'open' else 2)


def test_preprocessing_reader_does_not_import_engine_or_torch(tmp_path):
    script = '''import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.split('.')[0] in ('torch', 'local_postshot'):
   raise AssertionError('cross-runtime import: '+name)
sys.meta_path.insert(0, Block())
from gsstudio.application.studio import _optional_json
from pathlib import Path
assert _optional_json(Path(sys.argv[2])) is None
'''
    result = subprocess.run([sys.executable, '-I', '-c', script, str(REPO/'src'),
                             str(tmp_path/'absent.json')], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name != 'nt', reason='Native Windows delete-sharing handles')
def test_delete_shared_reader_closes_promptly_and_independent_replace_recovers(tmp_path, monkeypatch):
    path = tmp_path/'training.json'
    path.write_text('{"step": 4}', encoding='utf-8')
    original = live_json._read_file
    processes = []
    def read_file(reader_handle, *args):
        if processes:
            return original(reader_handle, *args)
        # Prove FILE_SHARE_DELETE by opening a second native DELETE handle.
        # MoveFileEx may nevertheless refuse to replace an open destination.
        handle = live_json._create_file(str(path), 0x10000, 0x7, None, 3, 0x80, None)
        assert handle != live_json._INVALID_HANDLE
        live_json._close_handle(handle)
        script = '''import sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
from local_postshot.project import atomic_json
original=Path.replace
def replace(temp,destination):
 Path(sys.argv[4]).touch()
 return original(temp,destination)
Path.replace=replace
atomic_json(sys.argv[3],{'step':8})
'''
        process = subprocess.Popen([sys.executable, '-I', '-X', 'utf8', '-c', script,
            str(REPO/'engine'), str(REPO/'src'), str(path), str(tmp_path/'replacing')],
            cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8')
        processes.append(process)
        deadline = time.monotonic()+10
        while not (tmp_path/'replacing').exists():
            assert process.poll() is None, 'Writer failed before replacement'
            assert time.monotonic() < deadline
            time.sleep(.001)
        # Give the independent writer time to encounter this still-open handle.
        time.sleep(.05)
        return original(reader_handle, *args)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(live_json, '_read_file', read_file)
            assert studio._optional_json(path) == {'step': 4}
        stdout, stderr = processes[0].communicate(timeout=10)
        assert processes[0].returncode == 0, stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
    assert studio._optional_json(path) == {'step': 8}


@pytest.mark.skipif(os.name != 'nt', reason='Actual native Windows independent processes')
def test_native_windows_processes_repeatedly_publish_and_poll_complete_json(tmp_path):
    path = tmp_path/'training.json'
    path.write_text(json.dumps({'step': 0, 'payload': [0]*64, 'text': '完整'}), encoding='utf-8')
    script = '''import json,sys,time
from pathlib import Path
repo,root,role=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
sys.path[:0]=[str(repo/'src'),str(repo/'engine')]
if role=='writer':
 from local_postshot.project import atomic_json
else:
 from gsstudio.application.studio import _optional_json
(root/(role+'.ready')).touch()
deadline=time.monotonic()+15
while not (root/'start').exists():
 assert time.monotonic()<deadline, 'start timeout'
 time.sleep(.001)
if role=='writer':
 for step in range(1,251):
  atomic_json(root/'training.json',dict(step=step,payload=[step]*64,text='完整'))
  time.sleep(.001)
 (root/'done').touch()
 print(json.dumps(dict(writes=250)))
else:
 reads=0; versions=set(); last=0
 while not (root/'done').exists() or reads<500 or last!=250:
  assert time.monotonic()<deadline, 'poll timeout'
  state=_optional_json(root/'training.json')
  assert state is not None
  step=state['step']
  assert state==dict(step=step,payload=[step]*64,text='完整'), state
  assert last<=step<=250
  versions.add(step);last=step;reads+=1
  time.sleep(.0001)
 print(json.dumps(dict(reads=reads,versions=len(versions),last=last)))
'''
    processes = []
    try:
        for role in ('writer', 'reader-one', 'reader-two'):
            processes.append(subprocess.Popen([sys.executable, '-I', '-u', '-c', script,
                str(REPO), str(tmp_path), role], cwd=tmp_path, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding='utf-8'))
        deadline = time.monotonic()+15
        while len(list(tmp_path.glob('*.ready'))) != 3:
            assert all(p.poll() is None for p in processes), 'Child failed before ready'
            assert time.monotonic() < deadline, 'Children did not become ready'
            time.sleep(.005)
        (tmp_path/'start').touch()
        reports = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            reports.append(json.loads(stdout))
        assert reports[0] == {'writes': 250}
        for report in reports[1:]:
            assert report['reads'] >= 500 and report['versions'] > 1 and report['last'] == 250
        assert studio._optional_json(path) == {'step': 250, 'payload': [250]*64, 'text': '完整'}
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)

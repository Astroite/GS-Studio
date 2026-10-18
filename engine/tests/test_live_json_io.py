"""Atomic JSON CPU contracts; every path belongs to a disposable fixture."""
import errno
import json
import os
from pathlib import Path
import time

import pytest

from local_postshot import project


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


@pytest.mark.skipif(os.name != 'nt', reason='Windows native retry policy')
@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_replace_retries_only_commit(tmp_path, monkeypatch, code):
    path = tmp_path/'training.json'
    project.atomic_json(path, {'step': 4})
    previous = path.read_bytes()
    real_replace, real_fsync = Path.replace, os.fsync
    calls, syncs = [], []
    clock, sleeps = fake_clock(monkeypatch)
    def fsync(fd):
        syncs.append(fd)
        return real_fsync(fd)
    def replace(temp, destination):
        calls.append(temp.read_bytes())
        assert json.loads(calls[-1]) == {'step': 8}
        assert destination.read_bytes() == previous
        if len(calls) < 3:
            raise win_error(code)
        return real_replace(temp, destination)
    monkeypatch.setattr(Path, 'replace', replace)
    monkeypatch.setattr(os, 'fsync', fsync)
    project.atomic_json(path, {'step': 8})
    assert json.loads(path.read_text()) == {'step': 8}
    assert len(calls) == 3 and len(syncs) == 1
    assert len(sleeps) == 2 and clock[0] == pytest.approx(.03)
    assert not path.with_name(path.name+'.tmp').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows native retry policy')
@pytest.mark.parametrize('code', [5, 32, 33])
def test_persistent_replace_denial_is_bounded_and_keeps_durable_json(tmp_path, monkeypatch, code):
    path = tmp_path/'training.json'
    project.atomic_json(path, {'last_good_step': 4})
    previous = path.read_bytes()
    error = win_error(code)
    calls = []
    clock, sleeps = fake_clock(monkeypatch)
    def denied(temp, destination):
        calls.append(temp)
        raise error
    monkeypatch.setattr(Path, 'replace', denied)
    with pytest.raises(PermissionError) as caught:
        project.atomic_json(path, {'step': 8})
    assert caught.value is error
    assert len(calls) > 1 and sleeps and clock[0] == pytest.approx(.5)
    assert path.read_bytes() == previous
    assert json.loads(path.with_name(path.name+'.tmp').read_text()) == {'step': 8}


@pytest.mark.parametrize('stage', ['serialize', 'flush', 'fsync', 'replace'])
@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EIO, errno.EACCES])
def test_nontransient_write_failure_is_not_retried_and_keeps_prior_destination(
        tmp_path, monkeypatch, stage, code):
    path = tmp_path/'training.json'
    project.atomic_json(path, {'last_good_step': 4})
    previous = path.read_bytes()
    error = OSError(code, 'injected durable IO failure')
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(time, 'sleep', lambda delay: pytest.fail('Nontransient IO was retried'))
    if stage == 'serialize':
        monkeypatch.setattr(json, 'dump', fail)
    elif stage == 'fsync':
        monkeypatch.setattr(os, 'fsync', fail)
    elif stage == 'replace':
        monkeypatch.setattr(Path, 'replace', fail)
    else:
        real_open = Path.open
        class FlushFailure:
            def __enter__(self):
                self.stream = real_open(path.with_name(path.name+'.tmp'), 'w', encoding='utf-8')
                return self
            def __exit__(self, *args):
                self.stream.close()
            def write(self, value):
                return self.stream.write(value)
            flush = fail
        monkeypatch.setattr(Path, 'open', lambda self, *args, **kw: FlushFailure()
                            if self.name == path.name+'.tmp' else real_open(self, *args, **kw))
    with pytest.raises(OSError) as caught:
        project.atomic_json(path, {'step': 8})
    assert caught.value is error and len(calls) == 1
    assert path.read_bytes() == previous
    temp = path.with_name(path.name+'.tmp')
    assert temp.is_file()
    if stage in ('fsync', 'replace'):
        assert json.loads(temp.read_text()) == {'step': 8}


def test_invalid_json_value_never_replaces_last_good_json(tmp_path):
    path = tmp_path/'training.json'
    project.atomic_json(path, {'last_good_step': 4})
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        project.atomic_json(path, {'loss': float('nan')})
    assert path.read_bytes() == previous
    assert path.with_name(path.name+'.tmp').exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows read-only destination behavior')
def test_actual_readonly_destination_is_not_changed_on_permanent_denial(tmp_path):
    import stat
    path = tmp_path/'training.json'
    project.atomic_json(path, {'last_good_step': 4})
    previous = path.read_bytes()
    path.chmod(stat.S_IREAD)
    started = time.monotonic()
    try:
        with pytest.raises(PermissionError):
            project.atomic_json(path, {'step': 8})
        assert .45 <= time.monotonic()-started < 1.5
        assert path.read_bytes() == previous
        assert json.loads(path.with_name(path.name+'.tmp').read_text()) == {'step': 8}
    finally:
        path.chmod(stat.S_IWRITE | stat.S_IREAD)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX IO must not gain Windows retries')
def test_posix_does_not_retry_even_if_error_has_winerror(tmp_path, monkeypatch):
    path = tmp_path/'training.json'
    project.atomic_json(path, {'step': 4})
    error = win_error(32)
    def denied(*args):
        raise error
    monkeypatch.setattr(Path, 'replace', denied)
    monkeypatch.setattr(time, 'sleep', lambda delay: pytest.fail('POSIX IO was retried'))
    with pytest.raises(PermissionError) as caught:
        project.atomic_json(path, {'step': 8})
    assert caught.value is error
    assert json.loads(path.read_text()) == {'step': 4}

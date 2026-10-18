"""Reentrant, process-wide per-Run exclusion for stages and retention."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import threading

_mutex = threading.RLock()
_local = threading.local()


def is_run_active(work: Path) -> bool:
    """Read the existing Run lock without creating files or changing the Run."""
    lock = work / '.run.lock'
    if not lock.is_file():
        return False
    try:
        with lock.open('r+b') as stream:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream, fcntl.LOCK_UN)
    except (OSError, BlockingIOError):
        return True
    return False


@contextmanager
def run_session(work: Path):
    from gsstudio.infrastructure.path_safety import safe_path
    with _mutex:
        key = str(work.absolute())
        active = getattr(_local, 'active', set())
        if key in active:
            yield
            return
        lock = safe_path(work, '.run.lock')
        with lock.open('a+b') as stream:
            if lock.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as error:
                raise RuntimeError('Run is busy; wait for its active stage before cleanup/resume') from error
            _local.active = active | {key}
            try:
                yield
            finally:
                _local.active = active
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def run_locked(function):
    @wraps(function)
    def wrapped(scene_path, run, *args, **kwargs):
        if getattr(run.config, 'schema_version', 0) != 6:
            return function(scene_path, run, *args, **kwargs)
        with run_session(scene_path / run.id):
            # A caller may have loaded before acquiring the lock.
            from gsstudio.infrastructure.persistence.run_repository import load_run
            run = load_run(scene_path, run.id)
            return function(scene_path, run, *args, **kwargs)
    return wrapped




def run_dir_locked(function):
    @wraps(function)
    def wrapped(run_dir, *args, **kwargs):
        load_run = function.__globals__['load_run']
        run_dir = Path(run_dir).absolute()
        run = load_run(run_dir.parent, run_dir.name)
        if run.config.schema_version == 6:
            with run_session(run_dir):
                return function(run_dir, *args, **kwargs)
        return function(run_dir, *args, **kwargs)
    return wrapped

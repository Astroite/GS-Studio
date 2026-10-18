"""Serialize SDK, alignment and training GPU jobs across workspace processes."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import os
import sys
import threading
import time

_mutex=threading.RLock()
_local=threading.local()
from gsstudio.infrastructure.paths import user_settings_dir

GPU_LOCK_ENV = 'GSSTUDIO_GPU_LOCK_PATH'


def gpu_lock_path() -> Path:
    """One writable lock for source CLI, desktop worker and packaged trainer."""
    override = os.environ.get(GPU_LOCK_ENV, '').strip()
    if override:
        selected = Path(override).expanduser()
        if not selected.is_absolute():
            raise ValueError(f'{GPU_LOCK_ENV} must be an absolute path')
        return selected
    from gsstudio.infrastructure.runtime.configuration import runtime_config
    return Path(runtime_config()['gpu_lock_path'])


GPU_LOCK_PATH = gpu_lock_path()


@contextmanager
def gpu_session(path: Path | None = None, *, cancel=None, timeout: float | None = None):
    with _mutex:
        if getattr(_local,'active',False):
            yield
            return
        path=path or gpu_lock_path()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a+b') as stream:
            if path.stat().st_size==0:
                stream.write(b'0')
                stream.flush()
            announced=False
            waiting_since=time.monotonic()
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError('GPU wait cancelled')
                if timeout is not None and time.monotonic()-waiting_since>=timeout:
                    raise TimeoutError('Workspace GPU is busy')
                try:
                    stream.seek(0)
                    if os.name=='nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
                    else:
                        import fcntl
                        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except (BlockingIOError,PermissionError):
                    if not announced:
                        print('Waiting for another workspace GPU stage to finish.',flush=True)
                        announced=True
                    time.sleep(1)
            _local.active=True
            try:
                yield
            finally:
                _local.active=False
                torch=sys.modules.get('torch')
                if torch is not None and torch.cuda.is_initialized():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as error:
                        print(f'GPU cache cleanup failed after stage: {error}',flush=True)
                stream.seek(0)
                if os.name=='nt':
                    msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    fcntl.flock(stream,fcntl.LOCK_UN)


def gpu_locked(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with gpu_session():
            return function(*args,**kwargs)
    return wrapped

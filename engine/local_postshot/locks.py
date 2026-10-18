"""Per-project process ownership, including CLI/desktop concurrency and crashes."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time

_local=threading.local()


class ProjectBusy(RuntimeError):
    pass


def root_path(path):
    path=Path(path).resolve()
    return path.parent if path.name=="project.json" else path


def pid_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        return True
    except (ProcessLookupError,PermissionError):return False


def owner(path):
    try:return json.loads((root_path(path)/".engine-task.json").read_text(encoding="utf-8"))
    except (OSError,ValueError):return None


def _lock(file):
    file.seek(0)
    if os.name=="nt":
        import msvcrt
        msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)


def _unlock(file):
    file.seek(0)
    if os.name=="nt":
        import msvcrt
        msvcrt.locking(file.fileno(),msvcrt.LK_UNLCK,1)
    else:
        import fcntl
        fcntl.flock(file.fileno(),fcntl.LOCK_UN)


def is_active(path):
    root=root_path(path)
    marker=owner(root)
    # Real ownership is the OS byte lock, not a reusable PID. This one temporary
    # migration marker protects a task started before locking was implemented.
    if marker and marker.get("legacy_owner") and pid_alive(marker.get("pid")):return True
    file=root/".engine.lock"
    if not file.exists():return False
    with file.open("r+b") as f:
        try:_lock(f)
        except OSError:return True
        _unlock(f)
    return False


@contextmanager
def operation_lock(path,method):
    root=root_path(path)
    key=str(root).casefold() if os.name=="nt" else str(root)
    held=getattr(_local,"held",set())
    if key in held:
        yield
        return
    marker=owner(root)
    if marker and marker.get("legacy_owner") and marker.get("pid")!=os.getpid() and pid_alive(marker.get("pid")):
        raise ProjectBusy(f"Project is active in another worker (PID {marker['pid']}, {marker.get('method')})")
    if not root.is_dir():raise FileNotFoundError(f"Project directory not found: {root}")
    file=(root/".engine.lock").open("a+b")
    if file.tell()==0:file.write(b"0");file.flush()
    try:
        try:_lock(file)
        except OSError as exc:raise ProjectBusy("Project is active in another worker") from exc
        held.add(key);_local.held=held
        active=root/".engine-task.json"
        temp=root/".engine-task.json.tmp"
        try:
            temp.write_text(json.dumps({"pid":os.getpid(),"method":method,"started_at":time.time()}),encoding="utf-8")
            temp.replace(active)
            yield
        finally:
            current=owner(root)
            if current and current.get("pid")==os.getpid():
                try:active.unlink()
                except OSError:pass
            held.remove(key)
            _unlock(file)
    finally:file.close()


def operation_locked(method):
    def decorator(function):
        @wraps(function)
        def run(path,*args,**kwargs):
            with operation_lock(path,method):return function(path,*args,**kwargs)
        return run
    return decorator

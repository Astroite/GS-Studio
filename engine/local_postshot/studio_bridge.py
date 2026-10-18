"""Isolated pipeline RPC child and shared GPU mutex. No preprocessing imports."""
from __future__ import annotations
import contextlib
import ctypes
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from .runtime import config

MAX_LINE = 4 * 1024 * 1024
PIPELINE_BOOTSTRAP = (
    'import runpy,sys; sys.dont_write_bytecode=True; root=sys.argv[1]; module=sys.argv[2]; '
    '[s.reconfigure(encoding="utf-8") for s in (sys.stdin,sys.stdout,sys.stderr) '
    'if s is not None and hasattr(s,"reconfigure")]; '
    'sys.path.insert(0,root); del sys.argv[1:3]; sys.argv[0]=module; '
    'runpy.run_module(module,run_name="__main__",alter_sys=True)'
)


def preprocess_command(cfg):
    python = Path(cfg.get('preprocess_python') or '')
    root = Path(cfg.get('pipeline_root') or '')
    if not cfg.get('preprocess_python') or not python.is_absolute() or not python.is_file():
        raise BridgeError('RUNTIME_MISSING', 'Configure preprocess_python (private interpreter) in GSSTUDIO_RUNTIME')
    if not cfg.get('pipeline_root') or not root.is_absolute() or not root.is_dir():
        raise BridgeError('RUNTIME_MISSING', 'Configure explicit pipeline_root in GSSTUDIO_RUNTIME')
    return [str(python), '-I', '-B', '-u', '-c', PIPELINE_BOOTSTRAP,
            str(root.resolve()), 'gsstudio.interfaces.worker.main']


class BridgeError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


class ChildJob:
    """KILL_ON_JOB_CLOSE contains every descendant, including native CLI children."""
    def __init__(self, process):
        self.process, self.handle = process, None
        if os.name != 'nt':
            return
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', w.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                        ('active_limit', w.DWORD), ('affinity', ctypes.c_size_t),
                        ('priority', w.DWORD), ('scheduling', w.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ('read_ops','write_ops','other_ops',
                                                     'read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_mem', ctypes.c_size_t),
                        ('job_mem', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.basic.flags = 0x2000
        if (not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits))
                or not kernel.AssignProcessToJobObject(handle, w.HANDLE(int(process._handle)))):
            if handle:
                kernel.CloseHandle(handle)
            process.kill()
            process.wait()
            raise OSError(ctypes.get_last_error(), 'Could not contain preprocessing child')
        self.handle, self.kernel = handle, kernel

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif self.process.poll() is None:
            if os.name != 'nt':
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            else:
                self.process.kill()


def child_environment(cfg=None):
    cfg = cfg or config()
    # The shared environment does not mix site-packages; both module roots are explicit.
    root = cfg.get('pipeline_root')
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from gsstudio.infrastructure.runtime.configuration import child_environment as make_environment
    return make_environment(cfg)


def call(method, params, *, request_id, emit=None, cancel=None):
    cfg = config()
    command = preprocess_command(cfg)
    request = {'id':request_id, 'method':method, 'params':{**params, 'root':params.get('root',cfg['data_root'])}}
    proc = subprocess.Popen(command,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
                            env=child_environment(cfg), text=True, encoding='utf-8', errors='strict',
                            start_new_session=os.name!='nt',
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    job = ChildJob(proc)
    messages = queue.Queue()
    def read():
        try:
            while True:
                line = proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line.encode('utf-8')) > MAX_LINE or not line.endswith('\n'):
                    raise ValueError('Child protocol line exceeds 4 MiB')
                messages.put(json.loads(line))
        except Exception as exc:
            messages.put(exc)
        finally:
            messages.put(None)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    terminal = None
    try:
        proc.stdin.write(json.dumps(request, ensure_ascii=False)+'\n')
        proc.stdin.flush()
        proc.stdin.close()
        while True:
            if cancel is not None and cancel.is_set():
                raise BridgeError('CANCELLED','Pipeline operation cancelled; inspect manifests before resuming')
            try:
                item = messages.get(timeout=.05)
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise BridgeError('CHILD_PROTOCOL',str(item))
            if not isinstance(item,dict) or item.get('id') != request_id:
                raise BridgeError('CHILD_PROTOCOL','Child request identity mismatch')
            kind = item.get('type')
            if kind == 'event' and terminal is None:
                if emit:
                    emit(item.get('event','progress'),item.get('data',{}))
            elif kind in ('result','error') and terminal is None:
                terminal = item
            else:
                raise BridgeError('CHILD_PROTOCOL','Unexpected or duplicate child terminal response')
        code = proc.wait()
        if terminal and terminal['type']=='error':
            error = terminal.get('error',{})
            raise BridgeError(error.get('code','PIPELINE_ERROR'),error.get('message','Pipeline failed'),error.get('details'))
        if code or terminal is None:
            raise BridgeError('CHILD_FAILED',f'Pipeline child exited {code} without a successful structured result')
        return terminal['result']
    finally:
        job.close()
        if proc.poll() is None:
            proc.wait(timeout=10)
        reader.join(timeout=1)
        proc.stdout.close()


_gpu_mutex = threading.RLock()
_gpu_local = threading.local()


@contextlib.contextmanager
def training_control_scope(cancel):
    """Bridge desktop shutdown Events to the core's identity-bound GPU wait."""
    previous=getattr(_gpu_local,'external_cancel',None)
    _gpu_local.external_cancel=cancel
    try:
        yield
    finally:
        _gpu_local.external_cancel=previous


def _cancelled(cancel):
    external=getattr(_gpu_local,'external_cancel',None)
    if external is not None and external.is_set() and cancel is not None:
        # TrainingControl is pure Python. Its own reader publishes stopping, so
        # studio_training records an orderly cancelled wait, never false success.
        if getattr(cancel,'status',None)=='waiting_gpu':
            from .studio_control import submit_control,read_control_state
            state=read_control_state(cancel.output)
            if state and state.get('generation_id')==cancel.generation_id:
                submit_control(cancel.output,cancel.package_sha256,'stop')
        elif not hasattr(cancel,'generation_id'):
            return True
    return cancel is not None and cancel.is_set()


@contextlib.contextmanager
def gpu_session(cancel=None):
    """Same byte lock as gsstudio; caller owns it, never a subprocess parent."""
    with _gpu_mutex:
        if getattr(_gpu_local,'held',False):
            yield
            return
        path = Path(os.environ.get('GSSTUDIO_GPU_LOCK_PATH') or config()['gpu_lock_path'])
        if not path.is_absolute():
            raise ValueError('GSSTUDIO_GPU_LOCK_PATH must be an absolute path')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a+b') as stream:
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            while True:
                if _cancelled(cancel):
                    raise BridgeError('CANCELLED','GPU wait cancelled')
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
                    time.sleep(.1)
            _gpu_local.held = True
            try:
                yield
            finally:
                _gpu_local.held = False
                stream.seek(0)
                if os.name=='nt':
                    msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    fcntl.flock(stream,fcntl.LOCK_UN)

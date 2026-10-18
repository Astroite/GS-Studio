"""Small, stdlib-only reads of JSON published by atomic replacement.

Windows readers share delete access to coexist with rename handles. MoveFileEx
can still reject an open destination, so reads close before JSON parsing and the
writer separately retries replacement.
Only native access/sharing/lock errors are retried, for at most 0.5s of backoff.
WinError 5 can also be a permanent ACL denial: it still raises at the deadline.
JSON/encoding errors and other IO failures are never retried or hidden.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

_WINDOWS = os.name == 'nt'
_RETRY_SECONDS = .5

if _WINDOWS:
    import ctypes
    from ctypes import wintypes
    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _create_file = _kernel.CreateFileW
    _create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _create_file.restype = wintypes.HANDLE
    _read_file = _kernel.ReadFile
    _read_file.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    _read_file.restype = wintypes.BOOL
    _close_handle = _kernel.CloseHandle
    _close_handle.argtypes = [wintypes.HANDLE]
    _close_handle.restype = wintypes.BOOL
    _INVALID_HANDLE = ctypes.c_void_p(-1).value


def _read_text(path: Path) -> str:
    if not _WINDOWS:
        return path.read_text(encoding='utf-8')
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    # OPEN_EXISTING; FILE_ATTRIBUTE_NORMAL. Read one handle, close it promptly,
    # then decode/parse; delete sharing does not guarantee open-target replace.
    handle = _create_file(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == _INVALID_HANDLE:
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = str(path)
        raise error
    try:
        chunks = []
        buffer = ctypes.create_string_buffer(65536)
        size = wintypes.DWORD()
        while True:
            # ReadFile retains native WinError 33 for byte-range locks; CRT
            # reads can reduce it to an ambiguous errno.EACCES instead.
            if not _read_file(handle, buffer, len(buffer), ctypes.byref(size), None):
                error = ctypes.WinError(ctypes.get_last_error())
                error.filename = str(path)
                raise error
            if not size.value:
                break
            chunks.append(buffer.raw[:size.value])
    finally:
        _close_handle(handle)
    return b''.join(chunks).decode('utf-8')


def read_json(path: Path):
    """Read one complete snapshot, propagating absence, corruption and IO errors."""
    path = Path(path)
    deadline = time.monotonic() + _RETRY_SECONDS
    delay = .01
    while True:
        try:
            text = _read_text(path)
            break
        except OSError as error:
            if not _WINDOWS or getattr(error, 'winerror', None) not in (5, 32, 33):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, .05)
    return json.loads(text)

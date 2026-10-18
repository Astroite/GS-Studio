from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable


class CommandError(RuntimeError):
    def __init__(
        self,
        command: list[str],
        returncode: int,
        log_path: Path,
        metrics: dict[str, float] | None = None,
    ):
        super().__init__(
            f"Command failed with exit code {returncode}; see {log_path}: "
            + shlex.join(command)
        )
        self.command = command
        self.returncode = returncode
        self.log_path = log_path
        self.metrics = metrics or {}


def _gpu_memory_used_mib(gpu_index: int | None = None) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
        if gpu_index is not None:
            command.append(f"--id={gpu_index}")
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_logged(
    command: Iterable[str],
    log_path: Path,
    cwd: Path | None = None,
    monitor_gpu: bool = False,
    gpu_index: int | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, float]:
    args = [str(item) for item in command]
    started = time.monotonic()
    peak_gpu_mib: list[float] = []
    stop_monitor = threading.Event()

    def sample_gpu() -> None:
        while not stop_monitor.is_set():
            value = _gpu_memory_used_mib(gpu_index)
            if value is not None:
                peak_gpu_mib.append(value)
            stop_monitor.wait(1.0)

    monitor = (
        threading.Thread(target=sample_gpu, name="gsstudio-gpu-monitor", daemon=True)
        if monitor_gpu
        else None
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("$ " + shlex.join(args) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if monitor:
            monitor.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
            returncode = process.wait()
        finally:
            stop_monitor.set()
            if monitor:
                monitor.join(timeout=6)
    metrics = {"elapsed_seconds": round(time.monotonic() - started, 3)}
    if peak_gpu_mib:
        metrics["gpu_memory_peak_mib"] = max(peak_gpu_mib)
    if returncode != 0:
        raise CommandError(args, returncode, log_path, metrics=metrics)
    return metrics


def command_output(command: Iterable[str]) -> str:
    result = subprocess.run(
        [str(item) for item in command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() or result.stderr.strip()

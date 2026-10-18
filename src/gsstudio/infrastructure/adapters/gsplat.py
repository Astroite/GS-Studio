"""Local gsplat interpreter, CUDA toolchain, and process command adapter."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from gsstudio.infrastructure.paths import find_app_root


def trainer_python() -> Path:
    from gsstudio.infrastructure.runtime.configuration import runtime_config
    cfg = runtime_config()
    python = Path(cfg.get('python') or '')
    if not cfg.get('python') or not python.is_absolute() or not python.is_file():
        raise RuntimeError("Configure private python in GSSTUDIO_RUNTIME; retired native trainer is unavailable")
    return python


def native_training_command(*args: str) -> list[str]:
    """Launch the same trainer contract from source or from an offline release."""
    from gsstudio.infrastructure.runtime.configuration import isolated_module_command, runtime_config
    return isolated_module_command(runtime_config(), 'local_postshot.studio_training', *args)


def trainer_diagnostic_command() -> list[str]:
    from gsstudio.infrastructure.runtime.configuration import isolated_module_command, runtime_config
    return isolated_module_command(runtime_config(), 'gsstudio.interfaces.trainer_diagnostic')


def _locate_cuda_home() -> Path | None:
    """CUDA 13 toolkit root: explicit env, then a system install, then APP\\..\\tools."""
    candidates = [os.environ.get("GSSTUDIO_CUDA_HOME"), os.environ.get("CUDA_HOME")]
    toolkit = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():
        candidates.extend(sorted(
            (d for d in toolkit.iterdir() if d.name.startswith("v13.")),
            key=lambda d: d.name, reverse=True,
        ))
    candidates.append(find_app_root().parent / "tools" / "cuda-13.4")
    for candidate in candidates:
        if candidate and (Path(candidate) / "bin" / "nvcc.exe").is_file():
            return Path(candidate)
    return None


def configure_windows_cuda() -> None:
    from gsstudio.infrastructure.runtime.configuration import runtime_config
    if runtime_config().get('release'):
        from local_postshot.release_runtime import require_binary
        require_binary('gsplat.csrc')
        return
    if os.name != "nt":
        return
    try:
        from gsplat import csrc
        return
    except ImportError:
        pass
    cuda = _locate_cuda_home()
    if cuda is None:
        raise RuntimeError("Pinned native trainer requires a CUDA 13 toolkit; set GSSTUDIO_CUDA_HOME")
    locator = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    installation = subprocess.check_output(
        [str(locator), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        text=True, encoding="utf-8", errors="replace",
    ).strip()
    vcvars = Path(installation) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.is_file():
        raise RuntimeError("MSVC build environment is unavailable")
    # cmd.exe writes OEM code-page bytes on Chinese Windows (often GBK), not UTF-8.
    raw_environment = subprocess.check_output(
        ["cmd.exe", "/d", "/s", "/c", f'""{vcvars}" >nul && set"'],
    )
    for encoding in ("oem", "mbcs", "utf-8"):
        try:
            environment = raw_environment.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        environment = raw_environment.decode("utf-8", errors="replace")
    for line in environment.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            os.environ[key] = value
    os.environ["CUDA_HOME"] = str(cuda)
    os.environ["CUDA_PATH"] = str(cuda)
    os.environ["PATH"] = str(cuda / "bin") + os.pathsep + os.environ["PATH"]
    os.environ["MAX_JOBS"] = "4"
    arch = os.environ.get("GSSTUDIO_CUDA_ARCH_LIST", "").strip()
    if not arch:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, encoding="utf-8", errors="replace",
        )
        arch = output.splitlines()[0].strip()
    if not re.fullmatch(r"\d+\.\d+", arch):
        raise RuntimeError(f"Invalid GPU compute capability: {arch}")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch

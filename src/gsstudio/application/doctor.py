from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from gsstudio.infrastructure.runtime.gpu_lock import gpu_session


def _version(command: list[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, "missing"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        text = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, text[0] if text else executable
    except Exception as error:  # diagnostic boundary
        return False, str(error)


def collect_tool_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name, command in {
        "ffmpeg": ["ffmpeg", "-version"],
        "colmap": ["colmap", "-h"],
    }.items():
        ok, detail = _version(command)
        versions[name] = detail if ok else "unavailable"
    colmap_cuda_ok, _ = _colmap_cuda_build()
    versions["colmap_cuda"] = "enabled" if colmap_cuda_ok else "unavailable"
    for distribution in ("gsplat", "nerfstudio", "torchvision"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    try:
        direct_url = importlib.metadata.distribution("nerfstudio").read_text("direct_url.json")
        if direct_url:
            commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
            if commit:
                versions["nerfstudio_commit"] = commit
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError):
        pass
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        versions["torch_cuda"] = str(torch.version.cuda)
    except Exception:
        versions["torch"] = "unavailable"
    return versions


def _colmap_cuda_build() -> tuple[bool, str]:
    executable = shutil.which("colmap")
    if executable is None:
        return False, "missing"
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        output = "\n".join((result.stdout, result.stderr)).strip()
        first_line = output.splitlines()[0] if output else executable
        cuda_enabled = " with CUDA" in output and " without CUDA" not in output
        return result.returncode == 0 and cuda_enabled, first_line
    except Exception as error:
        return False, str(error)


def _colmap_gpu_sift_smoke(data_root: Path, gpu_index: int = 0) -> tuple[bool, str]:
    try:
        import cv2
        from gsstudio.infrastructure.adapters.image_io import write_cv_image
        import numpy as np

        with tempfile.TemporaryDirectory(dir=data_root, prefix=".gsstudio-colmap-gpu-") as name:
            root = Path(name)
            images = root / "images" / "view_00"
            images.mkdir(parents=True)
            generator = np.random.default_rng(42)
            texture = generator.integers(0, 256, (256, 256), dtype=np.uint8)
            image = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
            cv2.circle(image, (128, 128), 64, (255, 255, 255), 3)
            if not write_cv_image(str(images / "frame_000001.jpg"), image):
                raise RuntimeError("Failed to write COLMAP smoke image")
            database = root / "database.db"
            result = subprocess.run(
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    str(database),
                    "--image_path",
                    str(root / "images"),
                    "--ImageReader.camera_model",
                    "PINHOLE",
                    "--ImageReader.single_camera_per_folder",
                    "1",
                    "--ImageReader.camera_params",
                    "128,128,128,128",
                    "--SiftExtraction.use_gpu",
                    "1",
                    "--SiftExtraction.gpu_index",
                    str(gpu_index),
                    "--SiftExtraction.max_num_features",
                    "512",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip().splitlines()
                return False, detail[-1] if detail else "feature_extractor failed"
            with sqlite3.connect(database) as connection:
                rows = connection.execute("SELECT rows FROM keypoints").fetchall()
            keypoints = sum(int(row[0]) for row in rows)
            return keypoints > 0, f"CUDA SIFT extracted {keypoints} keypoints on GPU {gpu_index}"
    except Exception as error:
        return False, str(error)


def _torch_cuda_check() -> tuple[bool, str]:
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        detail = f"torch={torch.__version__}, cuda={torch.version.cuda}, available={cuda_ok}"
        if cuda_ok:
            value = torch.randn(32, 32, device="cuda", requires_grad=True)
            (value.square().mean()).backward()
            detail += f", gpu={torch.cuda.get_device_name(0)}"
        return cuda_ok, detail
    except Exception as error:
        return False, str(error)


def _person_segmenter_check() -> tuple[bool, str]:
    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        return True, f"Mask R-CNN available; weights={MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.name} (download/cache deferred until mask)"
    except Exception as error:
        return False, str(error)


def _training_python() -> Path:
    from gsstudio.infrastructure.adapters.gsplat import trainer_python
    return trainer_python()


def _gsplat_runtime_check(python: Path) -> tuple[bool, str]:
    """Check the separate environment without starting training or installing packages."""
    if not python.is_file():
        return False, f"Missing native training runtime: {python}"
    try:
        from gsstudio.infrastructure.adapters.gsplat import trainer_diagnostic_command

        from gsstudio.infrastructure.runtime.configuration import child_environment
        result = subprocess.run(trainer_diagnostic_command(), capture_output=True, text=True,
                                env=child_environment(core=True),
                                encoding="utf-8", errors="replace", timeout=120)
        lines = (result.stdout if result.returncode == 0 else result.stderr or result.stdout).strip().splitlines()
        return result.returncode == 0, lines[-1] if lines else f"Native check exited {result.returncode}"
    except Exception as error:
        return False, str(error)


def run_doctor(
    data_root: Path,
    minimum_free_gib: float = 20.0,
    require: set[str] | None = None,
    backend: str = "gsplat",
) -> dict[str, Any]:
    required = set(require or ())
    from gsstudio.pipeline.training.runner import require_backend
    if backend != 'all':
        require_backend(backend)
    unknown = required - {"mediasdk", "colmap"}
    if unknown:
        raise ValueError(f"Unknown optional doctor requirement(s): {sorted(unknown)}")
    checks: dict[str, Any] = {}
    checks["ffmpeg"] = _version(["ffmpeg", "-version"])
    checks["ffprobe"] = _version(["ffprobe", "-version"])
    if "colmap" in required:
        checks["colmap_cuda_build"] = _colmap_cuda_build()
        if checks["colmap_cuda_build"][0]:
            with gpu_session():
                checks["colmap_gpu_sift"] = _colmap_gpu_sift_smoke(data_root)
        else:
            checks["colmap_gpu_sift"] = (False, "CUDA COLMAP unavailable for explicit legacy check")
    for name, resolver in _realityscan_resolvers().items():
        try:
            resolved = resolver()
            if name == "realityscan_export":
                ElementTree.parse(resolved)
            checks[name] = (True, str(resolved))
        except Exception as error:
            checks[name] = (False, str(error))

    deepseek_key_present = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if deepseek_key_present:
        checks["deepseek_environment"] = (
            True,
            "DEEPSEEK_API_KEY is present; value was not inspected or logged",
        )

    with gpu_session():
        checks["torch_cuda"] = _torch_cuda_check()
    checks["person_segmenter"] = _person_segmenter_check()
    try:
        python = _training_python()
    except RuntimeError as error:
        checks["training_python"] = (False, str(error))
        if backend in {"gsplat", "all"}:
            checks["gsplat_cuda"] = (False, f"Configure the private runtime first: {error}")
    else:
        checks["training_python"] = (python.is_file(), f"{python}; also used for shared PLY evaluation")
        if backend in {"gsplat", "all"}:
            checks["gsplat_cuda"] = _gsplat_runtime_check(python)

    try:
        with tempfile.NamedTemporaryFile(dir=data_root, prefix=".gsstudio-doctor-", delete=True) as stream:
            stream.write(b"ok")
            stream.flush()
        checks["data_writable"] = (True, str(data_root))
    except Exception as error:
        checks["data_writable"] = (False, str(error))

    try:
        free_gib = shutil.disk_usage(data_root).free / 1024**3
        checks["disk_free"] = (free_gib >= minimum_free_gib,
                              f"{free_gib:.1f} GiB free; minimum reserve {minimum_free_gib:.1f} GiB")
    except Exception as error:
        checks["disk_free"] = (False, str(error))
    try:
        from gsstudio.infrastructure.adapters.mediasdk import media_capabilities

        with tempfile.TemporaryDirectory(
            dir=data_root, prefix=".gsstudio-mediasdk-doctor-"
        ) as name:
            capabilities = media_capabilities(Path(name))
        checks["mediasdk"] = (
            True,
            f"helper={capabilities.get('helper_version', 'unknown')}, "
            f"sdk={capabilities.get('sdk_version', 'unknown')}",
        )
    except Exception as error:
        checks["mediasdk"] = (False, f"optional unavailable: {error}")
    checks["ok"] = all(
        result[0]
        for key, result in checks.items()
        if key != "ok" and (key not in {"mediasdk"} or key in required)
    )
    return checks


def _realityscan_resolvers():
    from gsstudio.infrastructure.adapters.realityscan import realityscan_executable, export_params_file

    return {"realityscan": realityscan_executable, "realityscan_export": export_params_file}

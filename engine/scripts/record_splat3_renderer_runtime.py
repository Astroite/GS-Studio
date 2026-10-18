"""Incrementally record the private renderer; never rewrite SPZ or dataset files.

The module is import-safe and uses only the Python standard library. Recording
an installed artifact does not import Torch, load a CUDA DLL or initialize CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
DISTRIBUTION = "local-splat3-render"
VERSION = "0.1.0"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_runtime_record(root: Path, renderer_record: dict, wheel: Path) -> dict:
    """Merge only renderer fields, preserving all unrelated provenance and data.

    Writes are atomic per file; a rerun safely completes either file if interrupted
    between manifest and lock updates. No dataset, native wheel or SPZ file is
    modified. The original manifest is backed up once, before the first merge.
    """
    root, wheel = Path(root).resolve(), Path(wheel).resolve()
    relative_wheel = wheel.relative_to(root).as_posix()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    manifest = root / "provisioning-manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8-sig")) if manifest.exists() else {}
    backup = root / "provisioning-manifest.before-splat3-render.json"
    if manifest.exists() and not backup.exists():
        shutil.copyfile(manifest, backup)
    lock = root / "runtime-requirements.lock.txt"
    lock_backup = root / "runtime-requirements.before-splat3-render.lock.txt"
    if lock.exists() and not lock_backup.exists():
        shutil.copyfile(lock, lock_backup)
    record.setdefault("artifacts", {})[relative_wheel] = digest(wheel)
    versions = record.setdefault("versions", {})
    for name in list(versions):
        if canonical(name) == DISTRIBUTION:
            del versions[name]
    versions[DISTRIBUTION] = renderer_record["version"]
    record["local_splat3_render"] = renderer_record
    atomic_text(manifest, json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = lock.read_text(encoding="utf-8-sig").splitlines() if lock.exists() else []
    lines = [line for line in lines if canonical(line.split("==", 1)[0].strip()) != DISTRIBUTION]
    lines.append(f"{DISTRIBUTION}=={renderer_record['version']}")
    atomic_text(lock, "\n".join(sorted(lines, key=str.lower)) + "\n")
    return record


def renderer_runtime_record(root: Path, *, receipt_path=None, strict_receipt=True):
    root = Path(root).resolve()
    dist = importlib.metadata.distribution(DISTRIBUTION)
    if dist.version != VERSION:
        raise RuntimeError(f"Expected {DISTRIBUTION} {VERSION}, found {dist.version}")
    binaries = [Path(dist.locate_file(p)).resolve() for p in (dist.files or ()) if str(p).endswith(".pyd")]
    if len(binaries) != 1:
        raise RuntimeError("Private renderer must contain exactly one native binary")
    binary = binaries[0]
    # Fail closed if invoked with an unrelated/global Python installation.
    binary.relative_to(root / "runtime")
    Path(sys.prefix).resolve().relative_to(root / "runtime")
    wheel = root / "wheels/local_splat3_render-0.1.0-cp310-cp310-win_amd64.whl"
    receipt_path = Path(receipt_path or REPO / "evidence/splat3/round3-renderer/build.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig")) if receipt_path.exists() else None
    wheel_hash, binary_hash = digest(wheel), digest(binary)
    verified = bool(receipt and receipt.get("staging_cpu_verification", {}).get("wheel_sha256") == wheel_hash
                    and receipt.get("identity", {}).get("binary_sha256") == binary_hash)
    if strict_receipt and not verified:
        raise RuntimeError("Installed renderer/wheel does not match validated build receipt; record a fresh build explicitly")
    sources = sorted((REPO / "engine/native/splat3_render").iterdir())
    record = {
        "distribution": DISTRIBUTION, "version": dist.version,
        "implementation": "native_renderer_v1", "binary_sha256": binary_hash,
        "wheel_sha256": wheel_hash, "source": "engine/native/splat3_render",
        "upstream": {"repository": "https://github.com/nerfstudio-project/gsplat",
                     "commit": "937e29912570c372bed6747a5c9bf85fed877bae", "license": "Apache-2.0"},
        "independently_written_stats": True, "original_postshot_code": False,
        "cuda_arch": "8.9", "cuda": "12.9", "compiler": "MSVC14.44.35207",
        "nvcc_flags": ["-O3"], "fast_math_flag": False,
        "reproduction_script": "scripts/setup-build-splat3-render.cmd",
        "source_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in sources
                          if p.is_file() and (p.suffix in (".cpp", ".cu", ".cuh", ".h", ".py", ".json")
                                               or p.name in ("LICENSE", "NOTICE"))},
        "artifact_validation": {"receipt": str(receipt_path), "receipt_sha256": digest(receipt_path) if receipt else None,
                                "matches_validated_build": verified,
                                "tests": receipt.get("validation") if verified else None},
        "known_substitutes": receipt.get("known_substitutes", []) if receipt else [
            "Public gsplat projection, SH, tile metadata and antialiasing compensation remain in use."],
        "packaging_allowlist": [binary.relative_to(root).as_posix(),
            "runtime/python-install/cpython-3.10.22-windows-x86_64-none/Lib/site-packages/local_splat3_render-0.1.0.dist-info/**"],
        "portable_package_allowlist": ["runtime/python/Lib/site-packages/local_splat3_render.cp310-win_amd64.pyd",
            "runtime/python/Lib/site-packages/local_splat3_render-0.1.0.dist-info/**",
            "engine/local_postshot/splat3_renderer.py"],
    }
    return record, wheel


def record_renderer_runtime(root: Path, *, receipt_path=None, strict_receipt=True):
    renderer, wheel = renderer_runtime_record(root, receipt_path=receipt_path, strict_receipt=strict_receipt)
    merge_runtime_record(root, renderer, wheel)
    return renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("LOCAL_POSTSHOT_ROOT", "E:/AI/LocalPostshot")))
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--allow-rebuilt-artifact", action="store_true",
                        help="Record a fresh local build without claiming tests for the bundled receipt's different binary")
    args = parser.parse_args()
    result = record_renderer_runtime(args.root, receipt_path=args.receipt, strict_receipt=not args.allow_rebuilt_artifact)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

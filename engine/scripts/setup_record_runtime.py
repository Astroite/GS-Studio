"""Record runtime provenance, preserving existing SPZ and dataset artifacts."""
import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import sysconfig
import zipfile
from importlib.metadata import distributions
from pathlib import Path
from record_splat3_renderer_runtime import record_renderer_runtime


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


root = Path(os.environ.get("LOCAL_POSTSHOT_ROOT", "E:/AI/LocalPostshot")).resolve()
site = Path(sysconfig.get_paths()["purelib"]).resolve()
# Reject unrelated/global interpreters before touching any site-packages,
# provenance, dataset or wheel file.
Path(sys.prefix).resolve().relative_to(root / "runtime")
site.relative_to(root / "runtime")
info = site / "spz-1.1.0.dist-info"
wheel = root / "wheels/spz-1.1.0-cp310-cp310-win_amd64.whl"
if not wheel.exists():
    # CMake's first install lacks wheel metadata. Existing wheels and metadata
    # are immutable inputs on a later runtime-only provisioning pass.
    if not (site / "spz/__init__.py").is_file() or not any((site / "spz").glob("*.pyd")):
        raise RuntimeError("Official SPZ native binding missing after build")
    info.mkdir(exist_ok=True)
    completed_metadata = False
    if not (info / "METADATA").exists():
        (info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: spz\nVersion: 1.1.0\nSummary: Official Niantic SPZ native Python bindings\nLicense: MIT\nRequires-Dist: numpy\n", encoding="utf-8")
        completed_metadata = True
    if not (info / "WHEEL").exists():
        (info / "WHEEL").write_text("Wheel-Version: 1.0\nGenerator: local-postshot-native-build\nRoot-Is-Purelib: false\nTag: cp310-cp310-win_amd64\n", encoding="utf-8")
        completed_metadata = True
    if not (info / "LICENSE").exists():
        (info / "LICENSE").write_bytes((root / "sources/spz/LICENSE").read_bytes())
        completed_metadata = True
    if not (info / "SOURCE.json").exists():
        (info / "SOURCE.json").write_text(json.dumps({"repository": "https://github.com/nianticlabs/spz", "commit": "affd0ecea7fbb4c265ee119475af7ee5b2997482", "cmake_flags": ["SPZ_BUILD_PYTHON_BINDINGS=ON", "SPZ_BUILD_TOOLS=OFF", "CMAKE_CXX_FLAGS=/utf-8"], "zlib": "1.3.2", "zstd": "1.5.6"}, indent=2), encoding="utf-8")
        completed_metadata = True
    items = [p for p in (site / "spz").glob("*") if p.is_file() and p.suffix != ".pyc"] + [p for p in info.glob("*") if p.is_file() and p.name != "RECORD"]
    if completed_metadata or not (info / "RECORD").exists():
        rows = []
        for path in items:
            raw = bytes.fromhex(digest(path))
            rows.append([path.relative_to(site).as_posix(), "sha256=" + base64.urlsafe_b64encode(raw).decode().rstrip("="), path.stat().st_size])
        rows.append(["spz-1.1.0.dist-info/RECORD", "", ""])
        stream = io.StringIO(newline="")
        csv.writer(stream).writerows(rows)
        (info / "RECORD").write_text(stream.getvalue(), encoding="utf-8")
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*items, info / "RECORD"]:
            archive.write(path, path.relative_to(site).as_posix())

versions = {d.metadata["Name"]: d.version for d in distributions()}
lock = "\n".join(f"{name}=={version}" for name, version in sorted(versions.items(), key=lambda item: item[0].lower())) + "\n"
(root / "runtime-requirements.lock.txt").write_text(lock, encoding="utf-8")

dataset = root / "datasets/south-building"
images = sorted((dataset / "images").glob("*.JPG"))
dataset_manifest = dataset / "manifest.json"
if not dataset_manifest.exists():
    if not images:
        raise RuntimeError("South Building dataset images are missing")
    manifest = {"source": "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip", "archive_sha256": digest(root / "downloads/south-building.zip"), "split_policy": "Sorted filenames; every eighth image (index modulo 8 = 0) is evaluation holdout", "images": [{"name": p.name, "sha256": digest(p), "split": "holdout" if i % 8 == 0 else "train"} for i, p in enumerate(images)]}
    dataset_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
else:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8-sig"))
artifacts = ["downloads/colmap-x64-windows-cuda.zip", "downloads/south-building.zip", "downloads/zlib-1.3.2.tar.gz", "wheels/gsplat-1.5.3-cp310-cp310-win_amd64.whl", "wheels/spz-1.1.0-cp310-cp310-win_amd64.whl", "tools/ffmpeg/ffmpeg.exe"]
record_path = root / "provisioning-manifest.json"
record = json.loads(record_path.read_text(encoding="utf-8-sig")) if record_path.exists() else {}
record.update({"python": sys.executable, "prefix": sys.prefix})
record.setdefault("versions", {}).update(versions)
record.setdefault("artifacts", {}).update({name: digest(root / name) for name in artifacts})
record.setdefault("gsplat", {"repository": "https://github.com/nerfstudio-project/gsplat", "tag": "v1.5.3", "commit": "937e29912570c372bed6747a5c9bf85fed877bae", "cuda_arch": "8.9 (NVIDIA Ada)", "cuda": "12.9", "msvc": "14.44.35207"})
if (info / "SOURCE.json").exists():
    record.setdefault("spz", json.loads((info / "SOURCE.json").read_text(encoding="utf-8-sig")))
record.setdefault("dataset", {"count": len(manifest["images"]), "train": sum(p["split"] == "train" for p in manifest["images"]), "holdout": sum(p["split"] == "holdout" for p in manifest["images"])})
ssim_wheel = root / "wheels/fused_ssim-1.0.0-cp310-cp310-win_amd64.whl"
if ssim_wheel.is_file():
    record["artifacts"][ssim_wheel.relative_to(root).as_posix()] = digest(ssim_wheel)
    record["fused_ssim"] = {**record.get("fused_ssim", {}), "repository":"https://github.com/rahul-goel/fused-ssim",
        "commit":"a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38", "version":"1.0.0",
        "cuda_arch":"8.9", "cuda":"12.9", "license":"MIT", "numerical_policy":"official variance/SSIM clamps and fast math; not bitwise Torch/original equivalence"}
stats_wheel = root / "wheels/local_splat3_stats-0.1.1-cp310-cp310-win_amd64.whl"
if stats_wheel.is_file():
    native_stats = site / "local_splat3_stats.cp310-win_amd64.pyd"
    record["artifacts"][stats_wheel.relative_to(root).as_posix()] = digest(stats_wheel)
    record["local_splat3_stats"] = {**record.get("local_splat3_stats", {}), "version":"0.1.1", "implementation":"native_cuda_v1",
        "source":"engine/native/splat3_stats", "independently_written":True,
        "binary_sha256":digest(native_stats), "cuda_arch":"8.9", "cuda":"12.9",
        "nvcc_flags":["-O3"], "original_postshot_code":False}
(root / "provisioning-manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
# Fresh local rebuilds may differ from the checked-in binary receipt. Record
# actual hashes without claiming that a different binary passed those tests.
renderer = record_renderer_runtime(root, strict_receipt=False)
from record_splat3_round4_runtime import record_round4_native_runtime
round4 = record_round4_native_runtime(root)
print(json.dumps({"spz_wheel": str(wheel), "dataset_images": len(images), "runtime_manifest": str(root / "provisioning-manifest.json"), "renderer": renderer}, indent=2))

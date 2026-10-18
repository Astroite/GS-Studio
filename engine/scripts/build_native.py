"""Explicit, offline native builds; --plan needs only Python and source files.

No installation, source checkout, dependency resolution or runtime JIT occurs.
Every real build copies its input to a fresh, external job directory first.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile


ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parent
PINS = {
    "gsplat": "937e29912570c372bed6747a5c9bf85fed877bae",
    "fused-ssim": "a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38",
    "spz": "affd0ecea7fbb4c265ee119475af7ee5b2997482",
    "zstd": "794ea1b0afca0f020f4e57b6732332231fb23c70",
    "zlib": "1.3.2",
}
GLM_PIN = "33b4a621a697a305bc3a7610d290677b96beb181"
GLM = "gsplat/cuda/csrc/third_party/glm"
MODULES = {
    "gsplat": ("gsplat", "1.5.3", "setup.py"),
    "fused-ssim": ("fused_ssim", "1.0.0", "setup.py"),
    "spz": ("spz", "1.1.0", "pyproject.toml"),
    "splat3-stats": ("local_splat3_stats", "0.1.1", "setup.py"),
    "splat3-render": ("local_splat3_render", "0.1.0", "setup.py"),
    "splat3-tiles": ("local_splat3_tiles", "0.1.0", "setup.py"),
    "splat3-projection": ("local_splat3_projection", "0.1.0", "setup.py"),
    "studio-selection": ("studio_selection_cuda", None, "selection.cpp"),
}


def absolute(value):
    return Path(os.path.abspath(value))


def plain_path(path):
    """Reject symbolic links and Windows junctions before resolving a path."""
    path = absolute(path)
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (
            candidate.exists()
            and getattr(candidate.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"Refuse symlink/reparse point: {candidate}")
    return path


def overlaps(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files(root):
    """Build payloads only, excluding previous build products and Git internals."""
    result = {}
    for parent, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in {
            ".git", "dist", "__pycache__", ".pytest_cache", ".venv"
        } and not d.endswith(".egg-info") and (
            d != "build" or (Path(parent) / d / "cmake/CMakeLists.txt").is_file()))
        for name in directories + files:
            plain_path(Path(parent) / name)
        for name in sorted(files):
            path = Path(parent) / name
            if name == ".git" or path.suffix.lower() in {".pyc", ".pyd", ".obj", ".lib", ".dll", ".whl"}:
                continue
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=MODULES)
    parser.add_argument("--python", default=os.environ.get("GSSTUDIO_BUILD_PYTHON") or os.environ.get("LP_PYTHON") or sys.executable)
    parser.add_argument("--source-root", help="Module source; local modules default relative to this script")
    parser.add_argument("--gsplat-source", default=os.environ.get("GSSTUDIO_GSPLAT_SOURCE") or os.environ.get("LP_GSPLAT_SOURCE"))
    parser.add_argument("--fused-ssim-source", default=os.environ.get("GSSTUDIO_FUSED_SSIM_SOURCE"))
    parser.add_argument("--spz-source", default=os.environ.get("GSSTUDIO_SPZ_SOURCE"))
    parser.add_argument("--zlib-source", default=os.environ.get("GSSTUDIO_ZLIB_SOURCE"))
    parser.add_argument("--zstd-source", default=os.environ.get("GSSTUDIO_ZSTD_SOURCE"))
    parser.add_argument("--source-receipts", help="Verified archive trees: component -> {revision, files:{relative_path:sha256}}")
    parser.add_argument("--output", required=True, help="A new job directory outside the repository and input trees")
    parser.add_argument("--vcvars", default=os.environ.get("GSSTUDIO_VCVARS") or os.environ.get("LP_VCVARS"))
    parser.add_argument("--vcvars-version", default="14.44")
    parser.add_argument("--cuda-home", default=os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"))
    parser.add_argument("--architecture", default=os.environ.get("TORCH_CUDA_ARCH_LIST", "8.9"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cmake", help="Optional explicit cmake executable for SPZ")
    parser.add_argument("--ninja", help="Optional explicit ninja executable")
    parser.add_argument("--plan", "--dry-run", action="store_true", help="Validate paths and print JSON; do not import Torch, compile or create output")
    return parser.parse_args(argv)


def make_plan(args):
    python = plain_path(args.python)
    if not python.is_file():
        raise ValueError(f"Python executable missing: {python}")
    if not re.fullmatch(r"\d+\.\d+(?:\+PTX)?(?:[; ]\d+\.\d+(?:\+PTX)?)*", args.architecture):
        raise ValueError("Architecture must be explicit CUDA capabilities, for example 8.9 or 8.9;9.0")
    if args.jobs < 1 or not re.fullmatch(r"\d+\.\d+", args.vcvars_version):
        raise ValueError("Invalid job count or MSVC toolset version")
    roots = {}
    if args.source_root:
        source = plain_path(args.source_root)
    elif args.module in {"gsplat", "fused-ssim", "spz"}:
        value = getattr(args, args.module.replace("-", "_") + "_source")
        if not value:
            raise ValueError(f"Explicit --{args.module}-source (or --source-root) is required")
        source = plain_path(value)
    elif args.module == "studio-selection":
        source = ENGINE / "editor_native"
    else:
        source = ENGINE / "native" / args.module.replace("-", "_")
    source = plain_path(source)
    if not (source / MODULES[args.module][2]).is_file():
        raise ValueError(f"Module source entry missing: {source / MODULES[args.module][2]}")
    if args.module == "studio-selection" and not (source / "selection.cu").is_file():
        raise ValueError(f"Selection CUDA source missing: {source}")
    roots[args.module] = source
    if args.module in {"gsplat", "splat3-projection"}:
        gsplat = source if args.module == "gsplat" else plain_path(args.gsplat_source) if args.gsplat_source else None
        if gsplat is None or not (gsplat / GLM / "glm/glm.hpp").is_file():
            raise ValueError("Pinned --gsplat-source with its GLM submodule is required")
        roots["gsplat"] = gsplat
    if args.module == "spz":
        for name, entry in (("zlib", "CMakeLists.txt"), ("zstd", "build/cmake/CMakeLists.txt")):
            value = getattr(args, name + "_source")
            if not value or not (plain_path(value) / entry).is_file():
                raise ValueError(f"Explicit --{name}-source with {entry} is required; downloads are disabled")
            roots[name] = plain_path(value)
    output = plain_path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    for root in {REPO, *roots.values(), python.parent}:
        if overlaps(output, root):
            raise ValueError(f"Source/runtime/output overlap: {root}")
    tools = {}
    for key in ("vcvars", "cuda_home", "cmake", "ninja"):
        value = getattr(args, key)
        tools[key] = str(plain_path(value)) if value else None
    for value in (tools["vcvars"],):
        if value and any(c in value for c in '%!"\r\n'):
            raise ValueError("vcvars path contains unsupported cmd expansion characters")
    snapshots = {name: source_files(root) for name, root in roots.items()}
    return {
        "schema": 1, "module": args.module, "distribution": MODULES[args.module][0],
        "version": MODULES[args.module][1], "python": str(python), "output": str(output),
        "architecture": args.architecture, "jobs": args.jobs, "tools": tools,
        "vcvars_version": args.vcvars_version,
        "sources": {name: {"root": str(root), "revision": PINS.get(name), "files": snapshots[name]}
                    for name, root in roots.items()},
        "glm_revision": GLM_PIN if "gsplat" in roots else None,
        "artifacts": "extensions/studio-selection/<ABI-source-key>" if args.module == "studio-selection" else "wheels/*.whl",
        "build_only": True, "downloads": False, "installs": False, "runtime_jit": False,
    }


def git_output(root, *args):
    proc = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return proc.stdout.decode("utf-8").strip()


def cmd_command(command, unicode_output=False):
    # list2cmdline escapes nested quotes with backslashes; cmd.exe does not use
    # those C-runtime rules. Supply its documented /s /c outer quotes directly.
    comspec = os.environ.get("COMSPEC", "cmd.exe")
    return f'"{comspec}" /d {"/u " if unicode_output else ""}/s /c "{command}"'


def verify_external_sources(plan, receipts_path=None):
    receipts = json.loads(Path(receipts_path).read_text(encoding="utf-8-sig")) if receipts_path else {}
    for name, record in plan["sources"].items():
        if name not in PINS:
            continue
        root = Path(record["root"])
        if (root / ".git").exists():
            if git_output(root, "rev-parse", "HEAD") != PINS[name]:
                raise ValueError(f"Unpinned {name} source; expected {PINS[name]}")
            if git_output(root, "status", "--porcelain", "--untracked-files=normal"):
                raise ValueError(f"Modified {name} source checkout")
            if name == "gsplat":
                submodule = root / GLM
                if git_output(submodule, "rev-parse", "HEAD") != GLM_PIN:
                    raise ValueError("Unpinned GLM submodule")
                if git_output(submodule, "status", "--porcelain", "--untracked-files=normal"):
                    raise ValueError("Modified GLM source checkout")
        else:
            receipt = receipts.get(name, {})
            if receipt.get("revision") != PINS[name] or receipt.get("files") != record["files"]:
                raise ValueError(f"{name} archive needs a complete matching --source-receipts tree and pinned revision")
            if name == "gsplat" and receipt.get("glm_revision") != GLM_PIN:
                raise ValueError("gsplat archive receipt must bind the pinned GLM revision")


def compiler_environment(plan):
    if os.name != "nt":
        raise ValueError("Real builds require native Windows")
    vcvars = plan["tools"]["vcvars"]
    if not vcvars or not Path(vcvars).is_file():
        raise ValueError("Existing --vcvars is required; no Visual Studio edition is assumed")
    env = {key.upper(): value for key, value in os.environ.items()}
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX"):
        env.pop(key, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               DISTUTILS_USE_SDK="1", MSSdk="1", MAX_JOBS=str(plan["jobs"]),
               TORCH_CUDA_ARCH_LIST=plan["architecture"],
               CUDA_ARCHITECTURES=";".join(a.replace(".", "").replace("+PTX", "") for a in re.split(r"[; ]", plan["architecture"])),
               PYTORCH_NVML_BASED_CUDA_CHECK="1", CUDA_VISIBLE_DEVICES="",
               PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_BUILD_ISOLATION="1",
               VSCMD_SKIP_SENDTELEMETRY="1")
    command = f'call "{vcvars}" x64 -vcvars_ver={plan["vcvars_version"]} >nul && set'
    proc = subprocess.run(cmd_command(command, unicode_output=True),
                          env=env, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"vcvars failed (exit {proc.returncode}): " + proc.stderr.decode("utf-16-le", errors="replace"))
    for line in proc.stdout.decode("utf-16-le").splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key.upper()] = value
    paths = [str(Path(plan["python"]).parent / "Scripts")]
    if plan["module"] != "spz":
        cuda = plan["tools"]["cuda_home"]
        if not cuda or not (Path(cuda) / "bin/nvcc.exe").is_file():
            raise ValueError("Existing --cuda-home with bin/nvcc.exe is required")
        env["CUDA_HOME"] = env["CUDA_PATH"] = cuda
        paths.insert(0, str(Path(cuda) / "bin"))
    for name in ("cmake", "ninja"):
        explicit = plan["tools"][name]
        if explicit:
            if not Path(explicit).is_file():
                raise ValueError(f"Missing explicit {name}: {explicit}")
            paths.insert(0, str(Path(explicit).parent))
    env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    if not shutil.which("ninja", path=env["PATH"]):
        raise ValueError("Ninja is required in the explicit Python environment or --ninja")
    if plan["module"] == "spz" and not shutil.which("cmake", path=env["PATH"]):
        raise ValueError("CMake is required in the explicit Python environment or --cmake")
    if plan["module"] == "spz":
        configure_spz_environment(env, plan)
    return env


def configure_spz_environment(env, plan):
    """Use the initialized MSVC compiler with Ninja, independent of host defaults."""
    compiler = shutil.which("cl.exe", path=env["PATH"])
    if not compiler:
        raise ValueError("MSVC cl.exe is missing after the explicit vcvars initialization")
    for name in ("CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET", "CMAKE_GENERATOR_INSTANCE",
                 "CMAKE_TOOLCHAIN_FILE", "CMAKE_ARGS"):
        env.pop(name, None)
    env.update(CMAKE_GENERATOR="Ninja", CMAKE_BUILD_PARALLEL_LEVEL=str(plan["jobs"]),
               CC=compiler, CXX=compiler)


def validate_wheel(path, plan):
    """Do not accept a successful build with the wrong frozen name/version/ABI."""
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        wheel_paths = [name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")]
        if len(metadata_paths) != 1 or len(wheel_paths) != 1:
            raise RuntimeError(f"Invalid wheel metadata layout: {path}")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        wheel = BytesParser().parsebytes(archive.read(wheel_paths[0]))
    canonical = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    if canonical(metadata.get("Name", "")) != canonical(plan["distribution"]) or metadata.get("Version") != plan["version"]:
        raise RuntimeError(f"Wheel differs from frozen module/version: {path}")
    if wheel.get_all("Tag", []) != ["cp310-cp310-win_amd64"]:
        raise RuntimeError(f"Wheel must match frozen CPython 3.10 Windows x64 ABI: {path}")


def build(plan, receipts_path=None):
    verify_external_sources(plan, receipts_path)
    env = compiler_environment(plan)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staged = {}
    for name, record in plan["sources"].items():
        target = output / "sources" / name
        target.mkdir(parents=True)
        for relative, expected in record["files"].items():
            original = Path(record["root"]) / relative
            if sha256(original) != expected:
                raise ValueError(f"Source changed since plan: {original}")
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
        staged[name] = target
    env["TORCH_EXTENSIONS_DIR"] = str(output / "cache" / "torch-extensions")
    env["TMP"] = env["TEMP"] = str(output / "temp")
    Path(env["TEMP"]).mkdir()
    if "gsplat" in staged:
        env["LP_GSPLAT_SOURCE"] = str(staged["gsplat"])
    module = plan["module"]
    source = staged[module]
    base = [plan["python"], "-I", "-B", "-X", "utf8"]
    if module == "studio-selection":
        command = base + [str(ENGINE / "scripts/build_studio_selection.py"), "--worker",
                          "--source-root", str(source), "--output", str(output / "extensions"),
                          "--architecture", plan["architecture"]]
    elif module == "spz":
        command = base + ["-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                          "--wheel-dir", str(output / "wheels"), "--config-settings=build-dir=" + str(output / "build"),
                          "--config-settings=wheel.py-api=",
                          "--config-settings=cmake.define.FETCHCONTENT_FULLY_DISCONNECTED=ON",
                          "--config-settings=cmake.define.CMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON",
                          "--config-settings=cmake.define.CMAKE_DISABLE_FIND_PACKAGE_zstd=ON",
                          "--config-settings=cmake.define.CMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
                          "--config-settings=cmake.define.CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
                          "--config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_ZLIB=" + str(staged["zlib"]),
                          "--config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_ZSTD=" + str(staged["zstd"]),
                          "--config-settings=cmake.define.SPZ_BUILD_TOOLS=OFF",
                          "--config-settings=cmake.define.BUILD_SHARED_LIBS=OFF",
                          "--config-settings=cmake.define.CMAKE_CXX_FLAGS=/utf-8",
                          "--config-settings=cmake.define.CMAKE_C_COMPILER=" + env["CC"],
                          "--config-settings=cmake.define.CMAKE_CXX_COMPILER=" + env["CXX"],
                          "--config-settings=cmake.define.Python_EXECUTABLE=" + plan["python"], "."]
    else:
        command = base + [str(ENGINE / "scripts/run_setup_build.py"), "--source-root", str(source),
                          "--", "build", "--build-base", str(output / "build"),
                          "bdist_wheel", "--dist-dir", str(output / "wheels")]
    proof = {"schema": 1, "module": module, "command": command, "success": False}
    if module not in {"spz", "studio-selection"}:
        proof["setup_wrapper"] = {"path": "engine/scripts/run_setup_build.py",
                                  "sha256": sha256(ENGINE / "scripts/run_setup_build.py")}
    try:
        with (output / "build.log").open("wb") as log:
            proc = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        proof["exit_code"] = proc.returncode
        if proc.returncode:
            raise RuntimeError(f"{module} compiler exited {proc.returncode}; see {output / 'build.log'}")
        expected_artifacts = sorted((output / "extensions").rglob("*.pyd")) if module == "studio-selection" else sorted((output / "wheels").glob("*.whl"))
        if not expected_artifacts:
            raise RuntimeError("Compiler returned success without an expected artifact")
        if module != "studio-selection":
            if len(expected_artifacts) != 1:
                raise RuntimeError("Expected exactly one wheel per isolated module job")
            validate_wheel(expected_artifacts[0], plan)
        proof["artifacts"] = {p.relative_to(output).as_posix(): sha256(p) for p in expected_artifacts}
        proof["success"] = True
    finally:
        (output / "build-proof.json").write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return proof


def main(argv=None):
    try:
        args = parse_args(argv)
        plan = make_plan(args)
        result = plan if args.plan else build(plan, args.source_receipts)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Native build refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

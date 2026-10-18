"""Build the frozen Windows runtime from public inputs in a new workspace.

Only prepare downloads. Build and verify are offline. This is a development
bootstrap, never called at product startup; it does not copy an old runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
LOCKS = Path(__file__).resolve().parent
INPUTS = LOCKS / "public-build-inputs.json"
MODULES = ("gsplat", "fused-ssim", "spz", "splat3-stats", "splat3-render",
           "splat3-tiles", "splat3-projection", "studio-selection")
NATIVE_NAMES = {"gsplat", "fused-ssim", "spz", "local-splat3-stats",
                "local-splat3-render", "local-splat3-tiles", "local-splat3-projection"}
HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "github.com",
         "zlib.net", "developer.download.nvidia.com"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def plain_path(value):
    path = Path(os.path.abspath(value))
    for item in (path, *path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), "st_file_attributes", 0)
                                 & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"Refuse symlink or junction: {item}")
    return path


def overlap(left, right):
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def read_pins(path, seen=None):
    path = Path(path)
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Recursive lock include")
    seen.add(path)
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            child = (path.parent / line[3:]).resolve()
            if not child.is_relative_to(path.parent.resolve()):
                raise ValueError("Lock include escapes lock directory")
            values = read_pins(child, seen)
        else:
            match = re.fullmatch(r"([\w.-]+)==([^\s]+)", line)
            if not match:
                raise ValueError(f"Not an exact requirement: {line}")
            values = {canonical(match[1]): match[2]}
        if result.keys() & values.keys():
            raise ValueError("Duplicate requirement")
        result.update(values)
    seen.remove(path)
    return result


def load_inputs():
    from urllib.parse import urlsplit
    inputs = json.loads(INPUTS.read_text(encoding="utf-8-sig"))
    pins = read_pins(LOCKS / "unified-requirements.lock.txt")
    remote = inputs["wheels"] + inputs["sdists"]
    actual = {canonical(item["name"]): item["version"] for item in remote}
    if len(actual) != len(remote) or actual != {k: v for k, v in pins.items() if k not in NATIVE_NAMES}:
        raise ValueError("Public build inputs do not match the frozen dependency lock")
    records = [inputs["python"], inputs["zlib"], *remote, *inputs["cuda"]["components"]]
    filenames = set()
    for item in records:
        if (Path(item["filename"]).name != item["filename"] or item["filename"] in filenames
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Unsafe or duplicate public input")
        filenames.add(item["filename"])
        url = urlsplit(item["url"])
        if url.scheme != "https" or url.hostname not in HOSTS or url.username or url.fragment:
            raise ValueError("Unexpected public download host")
    for item in inputs["sources"]:
        if (not item["url"].startswith("https://github.com/")
                or not item["url"].endswith(".git")
                or not re.fullmatch(r"[a-f0-9]{40}", item["revision"])):
            raise ValueError("Source must have a fixed public GitHub commit")
    receipted = json.loads((LOCKS / "pipeline-wheels.json").read_text())
    by_name = {canonical(w["name"]): w for w in inputs["wheels"]}
    for item in receipted["wheels"]:
        if any(by_name[canonical(item["name"])][key] != item[key]
               for key in ("version", "filename", "sha256")):
            raise ValueError("Public wheel differs from the accepted pipeline wheel")
    return inputs


def numerical_identity():
    paths = []
    for directory in ("engine/local_postshot", "engine/native", "engine/editor_native"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".cu", ".cuh", ".cpp", ".h", ".json", ".txt"}:
                paths.append(path)
    for path in LOCKS.glob("*.txt"):
        paths.append(path)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def make_plan(args):
    inputs = load_inputs()
    workspace = plain_path(args.workspace)
    if overlap(workspace, ROOT):
        raise ValueError("Workspace must be outside the source repository")
    if workspace.exists() and not (workspace / "bootstrap-inputs.json").is_file():
        raise ValueError("Workspace already exists and is not owned by this bootstrap")
    cache = plain_path(args.download_cache) if args.download_cache else None
    if cache and (overlap(cache, ROOT) or overlap(cache, workspace)):
        raise ValueError("Read-only download cache must be outside source and workspace")
    if not re.fullmatch(r"\d+\.\d+(?:\+PTX)?(?:[; ]\d+\.\d+(?:\+PTX)?)*", args.architecture):
        raise ValueError("Specify CUDA architectures, for example 8.9")
    if args.jobs < 1 or not re.fullmatch(r"\d+\.\d+", args.vcvars_version):
        raise ValueError("Specify a positive job count and an explicit MSVC toolset version")
    git = plain_path(args.git) if args.git else None
    return {"schema": 1, "workspace": str(workspace), "download_cache": str(cache) if cache else None,
            "python": str(workspace / "python/python.exe"), "architecture": args.architecture,
            "vcvars": str(plain_path(args.vcvars)) if args.vcvars else None,
            "vcvars_version": args.vcvars_version,
            "cuda_home": str(workspace / "cuda") if args.download_cuda else
                         str(plain_path(args.cuda_home)) if args.cuda_home else None,
            "git": str(git) if git else None,
            "git_checkout_policy": {"core.symlinks": False},
            "download_cuda": args.download_cuda, "jobs": args.jobs,
            "public_inputs_sha256": digest(INPUTS), "numerical_sources": numerical_identity(),
            "inputs": inputs, "claim": "Public-input native Windows build; GPU and release acceptance are separate gates"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def offline_env(workspace):
    env = {key: value for key, value in os.environ.items() if key.upper() in {
        "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE", "APPDATA",
        "LOCALAPPDATA", "PROGRAMDATA", "NUMBER_OF_PROCESSORS"}}
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               CUDA_VISIBLE_DEVICES="", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_BUILD_ISOLATION="1", TORCH_EXTENSIONS_DIR=str(workspace / "forbidden-jit-cache"))
    windows = Path(env.get("SYSTEMROOT", "C:/Windows"))
    env["PATH"] = os.pathsep.join([str(workspace / "python"), str(workspace / "python/Scripts"),
                                      str(windows / "System32"), str(windows)])
    return env


def run(command, *, env=None, cwd=None, log=None):
    print(json.dumps({"command": [str(c) for c in command]}, ensure_ascii=False), flush=True)
    if log:
        with Path(log).open("wb") as stream:
            subprocess.run([str(c) for c in command], cwd=cwd, env=env, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
    else:
        subprocess.run([str(c) for c in command], cwd=cwd, env=env, check=True)


def checked_download(record, destination, cache=None):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != record["sha256"]:
            raise ValueError(f"Existing download has changed: {destination}")
        return
    cached = Path(cache) / record["filename"] if cache else None
    if cached and cached.is_file():
        plain_path(cached)
        if digest(cached) != record["sha256"]:
            raise ValueError(f"Read-only download cache hash mismatch: {cached}")
        shutil.copy2(cached, destination)
    else:
        partial = destination.with_name(destination.name + ".part")
        run(["curl.exe", "--fail", "--location", "--retry", "4", "--retry-all-errors",
             "--connect-timeout", "30", "--output", partial, record["url"]])
        if digest(partial) != record["sha256"]:
            raise ValueError(f"Downloaded file hash mismatch: {partial}")
        partial.replace(destination)


def safe_member(name):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def extract_archive(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                safe_member(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive symlink is not allowed")
            stream.extractall(destination)
    else:
        with tarfile.open(archive) as stream:
            for item in stream.getmembers():
                safe_member(item.name)
                if not item.isfile() and not item.isdir():
                    raise ValueError(f"Archive nonregular member: {item.name}")
            stream.extractall(destination)


def owned_plan(plan):
    workspace = Path(plan["workspace"])
    receipt = workspace / "bootstrap-inputs.json"
    if not receipt.is_file():
        raise ValueError("Run prepare first")
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    for key in ("workspace", "public_inputs_sha256", "numerical_sources", "architecture",
                "vcvars", "vcvars_version", "cuda_home", "download_cuda", "jobs", "git_checkout_policy"):
        if recorded.get(key) != plan[key]:
            raise ValueError(f"Bootstrap identity changed: {key}; use a new workspace")
    return workspace


def checkout_source(git, record, target):
    """Materialize link blobs as files, without changing global Git settings."""
    if not target.exists():
        run([git, "clone", "--config", "core.symlinks=false", "--filter=blob:none",
             "--no-checkout", record["url"], target])
        # --config is persisted by clone before its first working-tree checkout.
        run([git, "-C", target, "checkout", "--detach", record["revision"]])
    policy = subprocess.check_output([git, "-C", str(target), "config", "--local", "--get", "core.symlinks"],
                                     text=True).strip()
    if policy != "false":
        raise ValueError(f"Source checkout lacks the required local symlink policy: {target}")
    if record["name"] == "gsplat":
        run([git, "-c", "core.symlinks=false", "-C", target, "submodule", "update", "--init", "--recursive"])
        glm = target / "gsplat/cuda/csrc/third_party/glm"
        run([git, "-C", glm, "config", "--local", "core.symlinks", "false"])
    observed = subprocess.check_output([git, "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if observed != record["revision"]:
        raise ValueError(f"Public source checkout mismatch: {record['name']}")
    return {"name": record["name"], "revision": observed, "core.symlinks": False}


def prepare(plan):
    workspace = Path(plan["workspace"])
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=False)
        write_json(workspace / "bootstrap-inputs.json", plan)
    else:
        owned_plan(plan)
    for directory in ("downloads", "sources", "logs", "builds", "wheels"):
        (workspace / directory).mkdir(exist_ok=True)
    inputs = plan["inputs"]
    downloads = [inputs["python"], inputs["zlib"], *inputs["wheels"], *inputs["sdists"]]
    if plan["download_cuda"]:
        downloads += inputs["cuda"]["components"]
    for record in downloads:
        checked_download(record, workspace / "downloads" / record["filename"], plan["download_cache"])
    if not (workspace / "python/python.exe").exists():
        extract_archive(workspace / "downloads" / inputs["python"]["filename"], workspace / "python-unpack")
        (workspace / "python-unpack/python").rename(workspace / "python")
    for record in [inputs["zlib"], *inputs["sdists"]]:
        target = workspace / "sources" / f"{record['name']}-{record['version']}"
        if not target.exists():
            temporary = workspace / f"unpack-{record['name']}"
            extract_archive(workspace / "downloads" / record["filename"], temporary)
            (temporary / target.name).rename(target)
    checkouts = []
    for record in inputs["sources"]:
        target = workspace / "sources" / record["name"]
        checkouts.append(checkout_source(plan["git"] or "git", record, target))
    if plan["download_cuda"] and not (workspace / "cuda/bin/nvcc.exe").is_file():
        for record in inputs["cuda"]["components"]:
            temporary = workspace / f"unpack-{record['name']}"
            extract_archive(workspace / "downloads" / record["filename"], temporary)
            roots = list(temporary.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("Unexpected CUDA archive layout")
            for source in roots[0].rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(roots[0])
                # Preserve each component's legal resources even when filenames repeat.
                if len(relative.parts) == 1:
                    relative = Path("component-notices") / record["name"] / relative
                destination = workspace / "cuda" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and digest(source) != digest(destination):
                    raise ValueError(f"Conflicting CUDA archive resource: {relative}")
                shutil.copy2(source, destination)
    helper = native_helper()
    zlib_files = helper.source_files(workspace / "sources/zlib-1.3.2")
    prepared_file = workspace / "prepared.json"
    if prepared_file.is_file():
        previous = json.loads(prepared_file.read_text())
        if previous.get("zlib_source_files", zlib_files) != zlib_files:
            raise ValueError("Prepared zlib source changed; preserve this workspace and use a new one")
    write_json(prepared_file, {"success": True, "zlib_source_files": zlib_files, "git_checkouts": checkouts,
        "download_sha256": {record["filename"]: record["sha256"] for record in downloads}})
    return {"prepared": True, "workspace": str(workspace), "downloads": len(downloads)}


def native_helper():
    spec = importlib.util.spec_from_file_location("studio_offline_native_build", ROOT / "engine/scripts/build_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_recipe_identity():
    return {relative: digest(ROOT / relative) for relative in (
        "scripts/runtime/bootstrap_windows.py", "engine/scripts/build_native.py",
        "engine/scripts/build_studio_selection.py", "engine/scripts/run_setup_build.py")}


def build(plan):
    workspace = owned_plan(plan)
    if not (workspace / "prepared.json").is_file():
        raise ValueError("Prepare has not completed")
    recipe_file = workspace / "build-recipe.json"
    recipe = build_recipe_identity()
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError("Native build recipe changed; retain this workspace and use a new one")
    if not recipe_file.exists():
        write_json(recipe_file, recipe)
    python = Path(plan["python"])
    env = offline_env(workspace)
    if not plan["git"] or not Path(plan["git"]).is_file():
        raise ValueError("Build needs an existing --git executable for pinned source verification")
    env["PATH"] += os.pathsep + str(Path(plan["git"]).parent)
    base = [python, "-I", "-B", "-X", "utf8"]
    wheel_paths = [workspace / "downloads" / w["filename"] for w in plan["inputs"]["wheels"]]
    for item, record in zip(wheel_paths, plan["inputs"]["wheels"]):
        if digest(item) != record["sha256"]:
            raise ValueError(f"Public wheel changed before install: {item.name}")
    run(base + ["-m", "ensurepip", "--upgrade"], env=env, log=workspace / "logs/ensurepip.log")
    run(base + ["-m", "pip", "install", "--no-index", "--no-deps", *wheel_paths],
        env=env, log=workspace / "logs/public-wheel-install.log")
    # The accepted python-devtools==2 is a tiny, public source-only placeholder.
    for record in plan["inputs"]["sdists"]:
        run(base + ["-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                    "--wheel-dir", workspace / "wheels", workspace / "sources" / f"{record['name']}-{record['version']}"],
            env=env, log=workspace / "logs" / f"{record['name']}-build.log")
        matching = list((workspace / "wheels").glob(f"{record['name'].replace('-', '_')}-{record['version']}-*.whl"))
        if len(matching) != 1:
            raise ValueError("Source-only package did not produce one wheel")
        run(base + ["-m", "pip", "install", "--no-index", "--no-deps", matching[0]], env=env)
    helper = native_helper()
    zlib_root = workspace / "sources/zlib-1.3.2"
    zlib_files = helper.source_files(zlib_root)
    if json.loads((workspace / "prepared.json").read_text()).get("zlib_source_files") != zlib_files:
        raise ValueError("zlib source is not bound to the verified public archive; rerun prepare")
    receipt = {"zlib": {"revision": "1.3.2", "files": zlib_files}}
    write_json(workspace / "source-receipts.json", receipt)
    for module in MODULES:
        job = workspace / "builds" / module
        proof_path = job / "build-proof.json"
        if job.exists():
            proof = json.loads(proof_path.read_text()) if proof_path.is_file() else {}
            if not proof.get("success") or not proof.get("artifacts") or any(
                    digest(job / name) != value for name, value in proof.get("artifacts", {}).items()):
                raise ValueError(f"Incomplete or changed build retained at {job}; inspect it and use a new workspace")
        else:
            command = base + [ROOT / "engine/scripts/build_native.py", module, "--python", python,
                "--output", job, "--vcvars", plan["vcvars"] or "", "--vcvars-version", plan["vcvars_version"],
                "--cuda-home", plan["cuda_home"] or "", "--architecture", plan["architecture"],
                "--jobs", str(plan["jobs"]), "--source-receipts", workspace / "source-receipts.json"]
            for name in ("gsplat", "fused-ssim", "spz", "zstd"):
                command += [f"--{name}-source", workspace / "sources" / name]
            command += ["--zlib-source", zlib_root]
            # Toolchain initialization occurs in the native helper, not this process.
            run(command, env=env, log=workspace / "logs" / f"{module}-dispatch.log")
        if module != "studio-selection":
            wheels = list((job / "wheels").glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError(f"Expected exactly one {module} wheel")
            run(base + ["-m", "pip", "install", "--no-index", "--no-deps", wheels[0]], env=env)
    write_json(workspace / "built.json", {"success": True, "modules": list(MODULES),
        "numerical_sources": plan["numerical_sources"], "gpu_tested": False})
    return {"built": True, "gpu_tested": False}


def verify(plan):
    workspace = owned_plan(plan)
    if not (workspace / "built.json").is_file():
        raise ValueError("Native build has not completed")
    if json.loads((workspace / "build-recipe.json").read_text()) != build_recipe_identity():
        raise ValueError("Verification recipe differs from the recorded native build")
    base = [Path(plan["python"]), "-I", "-B", "-X", "utf8"]
    env = offline_env(workspace)
    run(base + [ROOT / "scripts/provision_runtime.py", "verify", "--python", plan["python"]], env=env,
        log=workspace / "logs/frozen-versions.log")
    run(base + ["-m", "pip", "check"], env=env, log=workspace / "logs/pip-check.log")
    selection_job = workspace / "builds/studio-selection"
    selection_proof = json.loads((selection_job / "build-proof.json").read_text())
    selection_artifacts = [selection_job / name for name, expected in selection_proof["artifacts"].items()
                           if name.endswith(".pyd") and digest(selection_job / name) == expected]
    if not selection_proof.get("success") or len(selection_artifacts) != 1:
        raise ValueError("Editor precompiled artifact is missing or changed")
    code = ("import importlib,importlib.util,json,torch,torchvision; names=['gsplat.csrc','fused_ssim_cuda','spz',"
            "'local_splat3_stats','local_splat3_render','local_splat3_tiles','local_splat3_projection',"
            "'cv2','numpy','scipy','PIL','pydantic','typer','yaml']; "
            "[importlib.import_module(n) for n in names]; "
            f"spec=importlib.util.spec_from_file_location('studio_selection_cuda',{str(selection_artifacts[0])!r}); "
            "extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension); "
            "assert not torch.cuda.is_initialized(); print(json.dumps(dict(imports=names,gpu_tested=False)))")
    run(base + ["-c", code], env=env, log=workspace / "logs/precompiled-imports.log")
    if (workspace / "forbidden-jit-cache").exists():
        raise ValueError("Verification created a JIT cache")
    result = {"success": True, "packages": 52, "offline_imports": True, "gpu_tested": False,
              "scope": "Fresh public-input Windows compiler/runtime import gate; no GPU or final package claim"}
    write_json(workspace / "verified.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "prepare", "build", "verify", "full"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--download-cache", type=Path, help="Optional read-only, hash-checked public download cache")
    parser.add_argument("--git", type=Path, default=shutil.which("git"), help="Explicit Git executable; defaults to the current build process tool")
    parser.add_argument("--vcvars", type=Path)
    parser.add_argument("--vcvars-version", default="14.44")
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--download-cuda", action="store_true", help="Extract pinned NVIDIA redists into workspace; no installer or driver")
    parser.add_argument("--architecture", default="8.9")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.cuda_home and args.download_cuda:
            raise ValueError("Choose existing --cuda-home or isolated --download-cuda")
        plan = make_plan(args)
        if args.phase != "plan" and os.name != "nt":
            raise ValueError("Execution requires native Windows")
        if args.phase in {"build", "full"} and (not plan["vcvars"] or not plan["cuda_home"]):
            raise ValueError("Build requires explicit --vcvars and --cuda-home or --download-cuda")
        result = plan
        for phase, action in (("prepare", prepare), ("build", build), ("verify", verify)):
            if args.phase in {phase, "full"}:
                result = action(plan)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Windows bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

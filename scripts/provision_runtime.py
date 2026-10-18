"""Offline reproduction: clone frozen core, add eight hash-locked pipeline wheels.

Never installs into an existing environment. No toolkit, native build, model
download, GPU import or test execution is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

LOCKS = Path(__file__).resolve().parent / "runtime"


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Recursive lock include: {path}")
    seen.add(path)
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            child = (path.parent / line[3:].strip()).resolve()
            if not child.is_relative_to(path.parent):
                raise ValueError("Lock includes must remain in the lock directory")
            values = read_pins(child, seen)
        else:
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)(?: --hash=sha256:[a-f0-9]{64})?", line)
            if not match:
                raise ValueError(f"Expected exact frozen requirement: {line}")
            values = {canonical(match[1]): match[2]}
        for name, version in values.items():
            if name in result:
                raise ValueError(f"Duplicate requirement: {name}")
            result[name] = version
    seen.remove(path)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_environment():
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA",
        "APPDATA", "PROGRAMDATA", "COMSPEC", "NUMBER_OF_PROCESSORS", "PATH"}}
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    return env


def environment_identity(python):
    code = ("import importlib.metadata as m,json,sys;"
            "print(json.dumps(dict(python=sys.version.split()[0],"
            "packages={d.metadata['Name']:d.version for d in m.distributions()})))")
    proc = subprocess.run([str(python), "-I", "-B", "-c", code],
                          env=clean_environment(), check=True, capture_output=True,
                          text=True, encoding="utf-8", timeout=30)
    identity = json.loads(proc.stdout)
    identity["packages"] = {canonical(k): v for k, v in identity["packages"].items()}
    return identity


def verify(python, kind="unified"):
    lock = LOCKS / ("core-constraints.txt" if kind == "core" else "unified-requirements.lock.txt")
    expected = read_pins(lock)
    identity = environment_identity(python)
    if identity["python"] != "3.10.22":
        raise ValueError(f"Frozen interpreter requires Python 3.10.22: {identity['python']}")
    actual = identity["packages"]
    differences = {name: {"expected": expected.get(name), "actual": actual.get(name)}
                   for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name)}
    if differences:
        raise ValueError(f"Frozen {kind} environment mismatch: {json.dumps(differences, sort_keys=True)}")
    return dict(python=identity["python"], packages=len(actual), kind=kind, ready=True)


def assert_plain_path(path):
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        if candidate.exists() and candidate.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"Refuse reparse point: {candidate}")


def plan(core_root, output, wheelhouse):
    core_root, output, wheelhouse = (Path(os.path.abspath(p)) for p in (core_root, output, wheelhouse))
    for path in (core_root, output, wheelhouse):
        assert_plain_path(path)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    for suffix in ('.dependency-install.json', '.provisioning.json'):
        sidecar = output.parent / (output.name + suffix)
        if sidecar.exists():
            raise ValueError(f"Provisioning evidence already exists: {sidecar}")
    for source in (core_root, wheelhouse, LOCKS):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"Source/output overlap: {source}")
    for parent, directories, files in os.walk(core_root, followlinks=False):
        for name in directories + files:
            assert_plain_path(Path(parent) / name)
    identity = verify(core_root / "python.exe", "core")
    wheel_records = json.loads((LOCKS / "pipeline-wheels.json").read_text())["wheels"]
    for record in wheel_records:
        wheel = wheelhouse / record["filename"]
        assert_plain_path(wheel)
        if not wheel.is_file() or sha256(wheel) != record["sha256"]:
            raise ValueError(f"Missing/changed frozen wheel: {wheel}")
    return dict(core_root=str(core_root), output=str(output), wheelhouse=str(wheelhouse),
                core=identity, pipeline_wheels=wheel_records,
                locks={p.name: sha256(p) for p in sorted(LOCKS.iterdir()) if p.is_file()})


def provision(core_root, output, wheelhouse):
    planned = plan(core_root, output, wheelhouse)
    core_root, output = Path(planned["core_root"]), Path(planned["output"])
    # copytree refuses existing output. Partial output is retained on failure.
    shutil.copytree(core_root, output, copy_function=shutil.copy2)
    copied_binaries = []
    for source in sorted(core_root.rglob("*")):
        if source.is_file() and source.suffix.lower() in {".pyd", ".dll"}:
            relative = source.relative_to(core_root)
            digest = sha256(source)
            if sha256(output / relative) != digest:
                raise RuntimeError(f"Frozen binary copy mismatch: {relative}")
            copied_binaries.append(dict(path=relative.as_posix(), sha256=digest))
    python = output / "python.exe"
    subprocess.run([str(python), "-I", "-B", "-m", "pip", "install",
                    "--no-index", "--find-links", planned["wheelhouse"], "--no-deps",
                    "--require-hashes", "--only-binary=:all:", "--constraint", str(LOCKS / "core-constraints.txt"),
                    "--requirement", str(LOCKS / "pipeline-additions.lock.txt"),
                    "--report", str(output.parent / (output.name + ".dependency-install.json"))],
                   env=clean_environment(), check=True)
    result = verify(python, "unified")
    subprocess.run([str(python), "-I", "-B", "-m", "pip", "check"],
                   env=clean_environment(), check=True)
    evidence = dict(plan=planned, result=result, frozen_binaries=copied_binaries)
    (output.parent / (output.name + ".provisioning.json")).write_text(
        json.dumps(evidence, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Read-only exact version verification; no imports of runtime modules")
    check.add_argument("--python", required=True, type=Path)
    check.add_argument("--kind", choices=("core", "unified"), default="unified")
    for name in ("plan", "provision"):
        command = commands.add_parser(name)
        command.add_argument("--core-root", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--wheelhouse", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify(args.python, args.kind)
    else:
        result = (plan if args.command == "plan" else provision)(
            args.core_root, args.output, args.wheelhouse)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

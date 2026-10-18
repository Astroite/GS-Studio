"""Frozen provisioning and runtime-import boundaries; no installation or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import tomli

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runtime_provisioner", REPO / "scripts/provision_runtime.py")
provisioner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provisioner)


def test_frozen_lock_partition_and_pipeline_wheel_hashes():
    locks = provisioner.LOCKS
    core = provisioner.read_pins(locks / "core-constraints.txt")
    runtime = provisioner.read_pins(locks / "runtime-requirements.lock.txt")
    build = provisioner.read_pins(locks / "build-test-requirements.lock.txt")
    unified = provisioner.read_pins(locks / "unified-requirements.lock.txt")
    additions = provisioner.read_pins(locks / "pipeline-additions.lock.txt")
    assert (len(core), len(runtime), len(build), len(unified), len(additions)) == (44, 41, 11, 52, 8)
    assert not runtime.keys() & build.keys()
    assert unified == core | additions == runtime | build
    assert not core.keys() & additions.keys()
    assert runtime["torch"] == core["torch"] == "2.8.0+cu129"
    assert runtime["pillow"] == "12.3.0"
    assert runtime["opencv-python-headless"] == "4.11.0.86"
    assert all(name in runtime for name in ("setuptools", "tomli", "ninja"))
    assert all(name in build for name in ("cmake", "pytest", "scikit-build-core", "nanobind", "python-devtools"))
    wheels = json.loads((locks / "pipeline-wheels.json").read_text())["wheels"]
    hash_lock = (locks / "pipeline-additions.lock.txt").read_text()
    assert len(wheels) == len({w["filename"] for w in wheels}) == 8
    for wheel in wheels:
        assert additions[provisioner.canonical(wheel["name"])] == wheel["version"]
        assert "--hash=sha256:" + wheel["sha256"] in hash_lock


def test_pins_reject_resolution_and_escaping_includes(tmp_path):
    lock = tmp_path / "requirements.txt"
    for text in ("torch>=2.8", "-e ../source", "-r ../outside.txt", "numpy==1.26.4\nnumpy==2.0"):
        lock.write_text(text)
        with pytest.raises(ValueError):
            provisioner.read_pins(lock)


def test_verify_rejects_changed_or_additional_core_packages(monkeypatch):
    core = provisioner.read_pins(provisioner.LOCKS / "core-constraints.txt")
    monkeypatch.setattr(provisioner, "environment_identity",
                        lambda _: dict(python="3.10.22", packages={**core, "pillow": "11.0.0"}))
    with pytest.raises(ValueError, match="pillow"):
        provisioner.verify("unused", "core")
    monkeypatch.setattr(provisioner, "environment_identity",
                        lambda _: dict(python="3.10.22", packages={**core, "unexpected": "1"}))
    with pytest.raises(ValueError, match="unexpected"):
        provisioner.verify("unused", "core")


def test_provisioning_refuses_existing_output_and_evidence(tmp_path):
    core, wheels = tmp_path / "core", tmp_path / "wheels"
    core.mkdir(); wheels.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="Output already exists"):
        provisioner.plan(core, output, wheels)
    fresh = tmp_path / "fresh"
    (tmp_path / "fresh.provisioning.json").touch()
    with pytest.raises(ValueError, match="evidence already exists"):
        provisioner.plan(core, fresh, wheels)
    with pytest.raises(ValueError, match="overlap"):
        provisioner.plan(core, core / "nested", wheels)


def test_provisioning_install_is_offline_additions_only(tmp_path, monkeypatch):
    core, output, wheels = (tmp_path / p for p in ("core", "new-python", "wheels"))
    core.mkdir(); wheels.mkdir()
    (core / "python.exe").write_bytes(b"fixture interpreter")
    (core / "frozen.pyd").write_bytes(b"frozen native bytes")
    monkeypatch.setattr(provisioner, "plan", lambda *a: dict(core_root=str(core),
                        output=str(output), wheelhouse=str(wheels)))
    monkeypatch.setattr(provisioner, "verify", lambda *a: dict(ready=True, packages=52))
    commands = []
    monkeypatch.setattr(provisioner.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    assert provisioner.provision(core, output, wheels)["ready"]
    assert (core / "frozen.pyd").read_bytes() == (output / "frozen.pyd").read_bytes() == b"frozen native bytes"
    install = commands[0][0]
    for flag in ("--no-index", "--no-deps", "--require-hashes", "--constraint"):
        assert flag in install
    assert install[0] == str(output / "python.exe")
    assert str(provisioner.LOCKS / "pipeline-additions.lock.txt") in install
    assert commands[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert commands[-1][0][-2:] == ["pip", "check"]


def test_declared_dependencies_match_frozen_environment():
    from importlib.metadata import version
    from packaging.requirements import Requirement
    document = tomli.loads((REPO / "pyproject.toml").read_text())
    for value in document["project"]["dependencies"]:
        requirement = Requirement(value)
        assert version(requirement.name) in requirement.specifier, value
    data = document["tool"]["setuptools"]["package-data"]["gsstudio"]
    assert all(pattern in data for pattern in ("_vendor/*.md", "_vendor/*.json", "_vendor/*.txt"))


def test_runtime_imports_without_development_payloads(tmp_path):
    script = """
import importlib,importlib.abc,sys
sys.path[:0]=sys.argv[1:3]
from local_postshot import runtime
original=runtime.config
runtime.config=lambda:dict(original(),release=True)
from local_postshot.release_runtime import install_if_packaged
install_if_packaged()
blocked={'cmake','pytest','_pytest','scikit_build_core','nanobind','devtools','python_devtools',
         'iniconfig','pluggy','pathspec','exceptiongroup','pip','wheel'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in blocked:
            raise ModuleNotFoundError('Pruned development payload: '+name,name=name)
sys.meta_path.insert(0,Block())
for name in ('torch.utils.cpp_extension','gsplat.cuda._backend','setuptools','tomli','ninja',
             'pydantic','typer','yaml','cv2','PIL.Image',
             'gsstudio._vendor.nerfstudio.colmap_parsing_utils',
             'gsstudio._vendor.nerfstudio.equirect_utils'):
    importlib.import_module(name)
import torch
assert not torch.cuda.is_initialized()
from torch.utils.cpp_extension import load
try: load('must-never-compile',[])
except RuntimeError as exc: assert 'disabled' in str(exc)
else: raise AssertionError('JIT allowed')
print('PASS: runtime imports; no development payloads or CUDA initialization')
"""
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run([sys.executable, "-I", "-B", "-c", script,
                          str(REPO / "engine"), str(REPO / "src")],
                          env=env, cwd=tmp_path, capture_output=True, text=True,
                          encoding="utf-8", timeout=45)
    assert proc.returncode == 0, proc.stderr
    assert "PASS: runtime imports" in proc.stdout

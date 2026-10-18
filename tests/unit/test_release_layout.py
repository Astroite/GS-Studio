"""Portable release path, Data preference and command-contract checks."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from gsstudio.infrastructure import paths
from gsstudio.infrastructure.adapters import gsplat
from gsstudio.infrastructure.runtime import gpu_lock, layout


def test_saved_data_root_is_shared_with_cli_and_override(tmp_path: Path, monkeypatch):
    data = tmp_path / "Data 中文"
    other = tmp_path / "Other"
    data.mkdir()
    other.mkdir()
    monkeypatch.setenv("GSSTUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("GSSTUDIO_DATA_ROOT", raising=False)
    assert paths.save_data_root(data) == data
    assert paths.find_data_root() == data
    assert json.loads((tmp_path / "settings" / "settings.json").read_text(encoding="utf-8"))["data_root"] == str(data)
    monkeypatch.setenv("GSSTUDIO_DATA_ROOT", str(other))
    assert paths.find_data_root() == other


def test_frozen_nested_executable_resolves_release_root(tmp_path: Path, monkeypatch):
    release = tmp_path / "GSStudio"
    worker = release / "worker"
    worker.mkdir(parents=True)
    (release / "release-manifest.json").write_text("{}", encoding="utf-8")
    executable = worker / "GSStudioWorker.exe"
    executable.touch()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    assert paths.find_app_root() == release


def test_frozen_commands_use_isolated_core_module(tmp_path: Path, monkeypatch):
    worker = tmp_path / "worker" / "GSStudioWorker.exe"
    trainer = tmp_path / "gpu-runtime" / "GSStudioTrainer.exe"
    worker.parent.mkdir()
    trainer.parent.mkdir()
    worker.touch()
    trainer.touch()
    monkeypatch.setattr(layout, "is_frozen", lambda: True)
    monkeypatch.setattr(layout, "find_app_root", lambda: tmp_path)
    runtime=tmp_path/'runtime.json'
    runtime.write_text(json.dumps({'python':str(trainer),'engine_root':str(tmp_path),
                                  'pipeline_root':str(tmp_path)}),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',str(runtime))
    monkeypatch.delenv('GSSTUDIO_GSPLAT_PYTHON',raising=False)
    assert layout.worker_command() == [str(worker)]
    command = gsplat.native_training_command("--dataset", "D:/Data")
    assert command[:5] == [str(trainer), '-I', '-B', '-u', '-c']
    assert command[7:] == ['local_postshot.studio_training', '--dataset', 'D:/Data']


def test_gpu_lock_defaults_to_shared_runtime_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GSSTUDIO_SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.delenv("GSSTUDIO_GPU_LOCK_PATH", raising=False)
    config=tmp_path/'runtime.json'
    config.write_text(json.dumps({'gpu_lock_path':'shared/gpu.lock'}),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',str(config))
    assert gpu_lock.gpu_lock_path() == tmp_path / 'shared' / 'gpu.lock'
    override = tmp_path / "shared.lock"
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", str(override))
    assert gpu_lock.gpu_lock_path() == override
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", "relative.lock")
    with pytest.raises(ValueError, match="absolute path"):
        gpu_lock.gpu_lock_path()


def test_release_command_missing_component_fails_explicitly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(layout, "is_frozen", lambda: True)
    monkeypatch.setattr(layout, "find_app_root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="worker is missing"):
        layout.worker_command()


def test_frozen_release_uses_bundled_models_and_tools(tmp_path: Path, monkeypatch):
    from gsstudio.infrastructure.runtime.model_cache import MODEL_SHA256

    models = tmp_path / "model-cache"
    checkpoints = models / "hub" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for filename in MODEL_SHA256:
        (checkpoints / filename).touch()
    tools = tmp_path / "tools" / "ffmpeg"
    tools.mkdir(parents=True)
    monkeypatch.setattr(layout, "is_frozen", lambda: True)
    monkeypatch.setattr(layout, "find_app_root", lambda: tmp_path)
    monkeypatch.setenv("TORCH_HOME", "C:/external-cache")
    monkeypatch.setenv("PATH", "C:/system")
    layout.configure_release_path()
    assert os.environ["TORCH_HOME"] == str(models)
    assert os.environ["PATH"] == str(tools) + os.pathsep + "C:/system"


def test_source_environment_keeps_its_model_cache(monkeypatch):
    monkeypatch.setattr(layout, "is_frozen", lambda: False)
    monkeypatch.setenv("TORCH_HOME", "C:/source-cache")
    monkeypatch.setenv("PATH", "C:/system")
    layout.configure_release_path()
    assert os.environ["TORCH_HOME"] == "C:/source-cache"
    assert os.environ["PATH"] == "C:/system"


def test_offline_model_cache_rejects_missing_and_changed_files(tmp_path: Path, monkeypatch):
    import hashlib
    from gsstudio.infrastructure.runtime import model_cache

    monkeypatch.setattr(model_cache, "MODEL_SHA256", {"model.pth": hashlib.sha256(b"original").hexdigest()})
    with pytest.raises(FileNotFoundError, match="offline model is missing"):
        model_cache.require_model_cache(tmp_path, verify_hashes=True)
    path = tmp_path / "hub" / "checkpoints" / "model.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original")
    assert model_cache.require_model_cache(tmp_path, verify_hashes=True) == [path]
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 does not match"):
        model_cache.require_model_cache(tmp_path, verify_hashes=True)

"""Direct trainer and source dispatch use one process-level GPU lock owner."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsstudio.interfaces import trainer_diagnostic
from gsstudio.pipeline.training import native, runner
from gsstudio.pipeline.training.data import json_write
from gsstudio.pipeline.input import sources


def test_direct_native_evaluation_holds_gpu_lock(monkeypatch, tmp_path: Path):
    active = False
    calls = []

    @contextmanager
    def lock():
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    monkeypatch.setattr(native, "gpu_session", lock)
    monkeypatch.setattr(native, "configure_windows_cuda", lambda: calls.append(("cuda", active)))
    monkeypatch.setattr(native, "validate_package", lambda _: {})
    monkeypatch.setattr(native, "load_ply", lambda _: "model")
    monkeypatch.setattr(native, "evaluate", lambda *_: calls.append(("evaluate", active)))
    monkeypatch.setattr(sys, "argv", ["trainer", "--dataset", str(tmp_path),
                                      "--output", str(tmp_path / "evaluation"),
                                      "--evaluate-ply", str(tmp_path / "model.ply")])
    native.main()
    assert calls == [("cuda", True), ("evaluate", True)]
    assert not active


def test_cpu_media_probe_and_preparation_do_not_claim_gpu_lock(monkeypatch, tmp_path: Path):
    active = False
    calls = []

    @contextmanager
    def lock():
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    monkeypatch.setattr(sources, "gpu_session", lock)
    monkeypatch.setattr(sources, "_prepare_capture_input", lambda _scene, capture, *_args:
                        calls.append((capture.source.kind, active)))
    for kind in ("perspective_video", "equirect_video", "equirect_sequence", "insta360_insv"):
        capture = SimpleNamespace(source=SimpleNamespace(kind=kind))
        sources.prepare_capture_input(tmp_path, capture, tmp_path, 1.0)
    assert calls == [("perspective_video", False), ("equirect_video", False),
                     ("equirect_sequence", False), ("insta360_insv", True)]
    calls.clear()
    monkeypatch.setattr(sources, "_probe_capture_source", lambda capture, _protocol:
                        calls.append((capture.source.kind, active)))
    for kind in ("perspective_video", "equirect_video", "equirect_sequence", "insta360_insv"):
        capture = SimpleNamespace(source=SimpleNamespace(kind=kind))
        sources.probe_capture_source(capture, tmp_path)
    assert calls == [("perspective_video", False), ("equirect_video", False),
                     ("equirect_sequence", False), ("insta360_insv", True)]


def test_gsplat_dispatch_leaves_lock_to_trainer(monkeypatch, tmp_path: Path, independent_evidence, mock_dispatch_package):
    package = tmp_path / "package"
    package.mkdir()
    (package / "dataset.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "experiment"
    mock_dispatch_package({"images": [
        {"split": "train", "image": "train.png"},
        {"split": "validation", "image": "validation.png"}]})
    monkeypatch.setattr(runner, "native_training_command", lambda *args: ["trainer", *args])

    def unexpected_parent_lock():
        raise AssertionError("Native child must own the GPU lock")

    @contextmanager
    def fail_lock():
        unexpected_parent_lock()
        yield

    def fake_logged(_command, _log, **_kwargs):
        independent_evidence(output,package)
        return {"elapsed_seconds": 1}

    monkeypatch.setattr(runner, "gpu_session", fail_lock)
    monkeypatch.setattr(runner, "run_logged", fake_logged)
    record = runner.train_package(package, output, backend="gsplat", steps=1)
    assert record["status"] == "succeeded"
    assert record["evaluation_sha256"] == runner.sha256_file(output / "evaluation" / "metrics.json")


def test_dispatch_does_not_publish_ply_without_evaluation(monkeypatch, tmp_path: Path, independent_evidence, mock_dispatch_package):
    package = tmp_path / "package"
    package.mkdir()
    (package / "dataset.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "experiment"
    mock_dispatch_package({"images": [
        {"split": "train", "image": "train.png"},
        {"split": "validation", "image": "validation.png"}]})
    monkeypatch.setattr(runner, "native_training_command", lambda *args: ["trainer", *args])

    def fake_logged(_command, _log, **_kwargs):
        independent_evidence(output,package)
        (output/'evaluation'/'metrics.json').unlink()
        return {"elapsed_seconds": 1}

    monkeypatch.setattr(runner, "run_logged", fake_logged)
    with pytest.raises(FileNotFoundError):
        runner.train_package(package, output, backend="gsplat", steps=1)
    assert json.loads((output / "dispatch.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_direct_trainer_diagnostic_holds_gpu_lock(monkeypatch):
    active = False

    @contextmanager
    def lock():
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    monkeypatch.setattr(trainer_diagnostic, "gpu_session", lock)
    monkeypatch.setattr(trainer_diagnostic, "_diagnose", lambda: 0 if active else 1)
    assert trainer_diagnostic.main() == 0

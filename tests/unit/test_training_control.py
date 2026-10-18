from __future__ import annotations

import json
import hashlib
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsstudio.application.experiments import control_training
from gsstudio.pipeline.training.control import TrainingControl, read_control_state, submit_control
from gsstudio.pipeline.training.data import json_write


DIGEST = "a" * 64


def _state(output):
    return read_control_state(output)


def _wait_for(output, status, request_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = _state(output)
        if state["status"] == status and state["request_id"] == request_id:
            return state
        time.sleep(0.02)
    pytest.fail(f"Training control did not reach {status}")


def test_pause_checkpoint_resume_and_stop_at_boundaries(tmp_path):
    control = TrainingControl(tmp_path, DIGEST)
    saved = []
    pause = submit_control(tmp_path, DIGEST, "pause")
    result = []
    thread = threading.Thread(target=lambda: result.append(control.observe(12, lambda: saved.append(12))), daemon=True)
    thread.start()
    _wait_for(tmp_path, "paused", pause["request_id"])
    checkpoint = submit_control(tmp_path, DIGEST, "checkpoint")
    state = _wait_for(tmp_path, "paused", checkpoint["request_id"])
    assert state["message"] == "checkpoint_saved"
    assert thread.is_alive()
    resume = submit_control(tmp_path, DIGEST, "resume")
    _wait_for(tmp_path, "running", resume["request_id"])
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert result == [None]
    assert saved == [12, 12]

    stop = submit_control(tmp_path, DIGEST, "stop")
    assert control.observe(13, lambda: saved.append(13)) == "stop"
    _wait_for(tmp_path, "stopping", stop["request_id"])
    control.complete("stopped", 13)
    assert _state(tmp_path)["status"] == "stopped"
    resumed = TrainingControl(tmp_path, DIGEST)
    assert resumed.observe(13, lambda: saved.append(13)) is None


def test_control_state_retries_transient_windows_access_denial(tmp_path, monkeypatch):
    from gsstudio.pipeline.training import control as control_module

    TrainingControl(tmp_path, DIGEST)
    original = Path.read_text
    attempts = []

    def interrupted_read(path, **kwargs):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("record is being replaced")
        return original(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", interrupted_read)
    monkeypatch.setattr(control_module.time, "sleep", lambda _: None)
    assert read_control_state(tmp_path)["status"] == "running"
    assert len(attempts) == 2


def test_control_state_does_not_hide_persistent_access_denial(tmp_path, monkeypatch):
    from gsstudio.pipeline.training import control as control_module

    TrainingControl(tmp_path, DIGEST)

    def denied_read(path, **kwargs):
        raise PermissionError("permanent denial")

    monkeypatch.setattr(Path, "read_text", denied_read)
    monkeypatch.setattr(control_module.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="permanent denial"):
        read_control_state(tmp_path)


def test_control_service_rejects_wrong_backend_and_escaped_output(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    json_write(output / "dispatch.json", {"backend": "postshot", "package_sha256": DIGEST})
    with pytest.raises(ValueError, match="only available"):
        control_training(tmp_path, output, "stop")
    outside = tmp_path.parent / "outside-experiment"
    with pytest.raises(ValueError, match="within"):
        control_training(tmp_path, outside, "stop")


def test_stop_at_completed_target_reports_completion(tmp_path):
    control = TrainingControl(tmp_path, DIGEST)
    request = submit_control(tmp_path, DIGEST, "stop")
    assert control.observe(20, lambda: None, target=20) is None
    state = _wait_for(tmp_path, "running", request["request_id"])
    assert "reached its target" in state["message"]
    control.complete("succeeded", 20)
    assert _state(tmp_path)["status"] == "succeeded"


def test_gpu_wait_reports_waiting_and_stop_cancels_before_step_zero(tmp_path):
    control = TrainingControl(tmp_path, DIGEST, waiting_for_gpu=True)
    assert _state(tmp_path)["status"] == "waiting_gpu"
    with pytest.raises(RuntimeError, match="not running"):
        submit_control(tmp_path, DIGEST, "pause")
    with pytest.raises(RuntimeError, match="checkpoint boundary"):
        submit_control(tmp_path, DIGEST, "checkpoint")
    request = submit_control(tmp_path, DIGEST, "stop")
    assert control.is_set()
    assert _state(tmp_path)["status"] == "stopping"
    assert _state(tmp_path)["request_id"] == request["request_id"]
    control.complete("stopped", 0)

    # A restart must not replay the already acknowledged stop request.
    restarted = TrainingControl(tmp_path, DIGEST, waiting_for_gpu=True)
    assert not restarted.is_set()
    restarted.start_running()
    assert _state(tmp_path)["status"] == "running"


def test_stale_control_request_cannot_cancel_a_new_process_generation(tmp_path):
    with pytest.raises(RuntimeError, match="not accepting"):
        submit_control(tmp_path, DIGEST, "stop")
    original = TrainingControl(tmp_path, DIGEST, waiting_for_gpu=True)
    first_generation = _state(tmp_path)["generation_id"]
    stale = submit_control(tmp_path, DIGEST, "stop")
    assert stale["generation_id"] == first_generation

    restarted = TrainingControl(tmp_path, DIGEST, waiting_for_gpu=True)
    assert _state(tmp_path)["generation_id"] != first_generation
    assert not restarted.is_set()
    fresh = submit_control(tmp_path, DIGEST, "stop")
    assert fresh["generation_id"] == restarted.generation_id
    assert restarted.is_set()
    assert _state(tmp_path)["request_id"] == fresh["request_id"]


def test_native_gpu_wait_stop_records_no_checkpoint(tmp_path, monkeypatch):
    from gsstudio.pipeline.training import native

    package = tmp_path / "package"
    package.mkdir()
    output = tmp_path / "experiment"
    monkeypatch.setattr(native, "validate_package", lambda _: {"images": []})
    monkeypatch.setattr(native, "sha256_file", lambda _: DIGEST)
    monkeypatch.setattr(sys, "argv", ["trainer", "--dataset", str(package),
                                    "--output", str(output), "--steps", "20"])

    class WaitingGpu:
        def __init__(self, cancel):
            self.cancel = cancel

        def __enter__(self):
            assert _state(output)["step"] == 0
            submit_control(output, DIGEST, "stop")
            assert self.cancel.is_set()
            raise InterruptedError("GPU wait cancelled")

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(native, "gpu_session", lambda *, cancel: WaitingGpu(cancel))
    native.main()
    result = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert result == {"status": "stopped", "steps": 0, "target": 20,
                      "reason": "gpu_wait_cancelled", "checkpoint": None}
    assert _state(output)["status"] == "stopped"
    assert not (output / "checkpoint.pt").exists()


def test_native_gpu_wait_stop_keeps_prior_checkpoint_and_summary(tmp_path, monkeypatch):
    from gsstudio.pipeline.training import native

    package = tmp_path / "package"
    package.mkdir()
    output = tmp_path / "experiment"
    output.mkdir()
    checkpoint = output / "checkpoint.pt"
    checkpoint.write_bytes(b"prior checkpoint")
    (output / "model.ply").write_bytes(b"prior preview model")
    prior_summary = {"status": "stopped", "steps": 5, "target": 20,
                     "model_sha256": DIGEST}
    json_write(output / "training.json", prior_summary)
    monkeypatch.setattr(native, "validate_package", lambda _: {"images": []})
    monkeypatch.setattr(native, "sha256_file", lambda _: DIGEST)
    monkeypatch.setattr(sys, "argv", ["trainer", "--dataset", str(package),
                                    "--output", str(output), "--steps", "20", "--resume"])

    class WaitingGpu:
        def __init__(self, cancel):
            self.cancel = cancel

        def __enter__(self):
            assert _state(output)["step"] == 5
            submit_control(output, DIGEST, "stop")
            assert self.cancel.is_set()
            raise InterruptedError("GPU wait cancelled")

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(native, "gpu_session", lambda *, cancel: WaitingGpu(cancel))
    native.main()
    summary = json.loads((output / "training.json").read_text(encoding="utf-8"))
    assert summary["status"] == "stopped"
    assert summary["steps"] == 5
    assert summary["reason"] == "gpu_wait_cancelled"
    assert summary["checkpoint"] == str(checkpoint)
    assert summary["model_sha256"] == DIGEST
    archive = output / summary["previous_training_record"]["path"]
    assert json.loads(archive.read_text(encoding="utf-8")) == prior_summary
    assert checkpoint.read_bytes() == b"prior checkpoint"
    assert _state(output)["step"] == 5


def test_wait_cancelled_experiment_restarts_from_zero_only_for_matching_identity(tmp_path, monkeypatch, mock_dispatch_package):
    from gsstudio.pipeline.training import runner

    package = tmp_path / "package"
    package.mkdir()
    (package / "dataset.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "experiment"
    output.mkdir()
    previous = {"backend": "gsplat", "package_sha256": DIGEST,
                "requested_steps": 20, "photo_comp": True,
                "use_bilateral_grid": True, "use_sparse_depth": True,
                'engine':runner.ENGINE_ID,'checkpoint_format':runner.CHECKPOINT,
                'training_recipe':'quality_v2','sh_degree':3,
                "status": "stopped", "completed_steps": 0,
                "stop_reason": "gpu_wait_cancelled"}
    json_write(output / "dispatch.json", previous)
    mock_dispatch_package({"images": [{"split": "train"}]})
    monkeypatch.setattr(runner, "sha256_file", lambda _: DIGEST)
    monkeypatch.setattr(runner, "native_training_command", lambda *args: ["trainer", *args])
    _, _, current = runner.prepare_training(package, steps=20)
    previous.update({k:v for k,v in current.items() if k!='steps'})
    json_write(output/'dispatch.json',previous)
    with pytest.raises(RuntimeError,match="explicitly set restart_from_step_zero"):
        runner.train_package(package,output,backend="gsplat",steps=20,resume=True,dry_run=True)
    record = runner.train_package(package, output, backend="gsplat", steps=20,
                                  resume=True, dry_run=True,restart_from_step_zero=True)
    assert record["restart_from_step_zero"] is True
    assert "--resume" not in record["command"]
    assert "--restart-from-step-zero" in record["command"]

    previous["completed_steps"] = 5
    json_write(output / "dispatch.json", previous)
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        runner.train_package(package, output, backend="gsplat", steps=20,
                             resume=True, dry_run=True)
    (output / "checkpoint.pt").write_bytes(b"existing checkpoint")
    with pytest.raises(RuntimeError,match='old checkpoints'):
        runner.train_package(package,output,backend='gsplat',steps=20,resume=True,dry_run=True)
    (output/'checkpoints').mkdir()
    (output/'checkpoints'/'latest.pt').write_bytes(b'current checkpoint')
    record = runner.train_package(package, output, backend="gsplat", steps=20,
                                  resume=True, dry_run=True)
    assert "--resume" in record["command"]


def test_failed_before_first_checkpoint_retries_with_archived_evidence(tmp_path, monkeypatch, mock_dispatch_package):
    from gsstudio.pipeline.training import runner

    package = tmp_path / "package"
    package.mkdir()
    (package / "dataset.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "experiment"
    output.mkdir()
    previous = {"backend": "gsplat", "package_sha256": DIGEST,
                "requested_steps": 20, "photo_comp": True,
                "use_bilateral_grid": True, "use_sparse_depth": True,
                'engine':runner.ENGINE_ID,'checkpoint_format':runner.CHECKPOINT,
                'training_recipe':'quality_v2','sh_degree':3,
                "status": "failed", "error": "out of memory"}
    json_write(output / "dispatch.json", previous)
    json_write(output / "config.json", {"package_sha256": DIGEST})
    (output / "train.log").write_text("first attempt failed\n", encoding="utf-8")
    json_write(output / "failure.json", {"error": "out of memory"})
    real_sha256 = runner.sha256_file
    monkeypatch.setattr(runner, "sha256_file", lambda path: (
        DIGEST if Path(path) == package / "dataset.json" else real_sha256(path)))
    mock_dispatch_package({"images": [{"split": "train"}]})
    monkeypatch.setattr(runner, "native_training_command", lambda *args: ["trainer", *args])
    _, _, current = runner.prepare_training(package, steps=20)
    previous.update({k:v for k,v in current.items() if k!='steps'})
    json_write(output/'dispatch.json',previous)

    def fake_run(command, log_path, **_kwargs):
        assert "--restart-from-step-zero" in command
        assert "--resume" not in command
        log_path.write_text("second attempt stopped\n", encoding="utf-8")
        config={'package_sha256':DIGEST,'steps':20,'photo_comp':True,'use_bilateral_grid':True,
                'use_sparse_depth':True,'sparse_depth_requested':True,'sparse_depth_available':True,
                'sparse_depth_observations':8,'sparse_depth_status':'enabled','training_recipe':'quality_v2',
                'engine':{'name':'local-postshot-independent-gsstudio','schema_version':1,
                          'sources':{'training.py':DIGEST},'versions':{'torch':'2.8.0+cu129'}},
                'checkpoint':str(output/'checkpoints'/'latest.pt')}
        json_write(output/'config.json',config)
        json_write(output / "training.json", {**config,"status": "stopped", "steps": 20,'completed_steps':0,
                                                'checkpoint':None,
                                                "reason": "gpu_wait_cancelled"})
        return {"elapsed_seconds": 0.01}

    monkeypatch.setattr(runner, "run_logged", fake_run)
    result = runner.train_package(package, output, backend="gsplat", steps=20, resume=True,restart_from_step_zero=True)
    assert result["status"] == "stopped"
    assert result["restart_from_step_zero"] is True
    attempt = result["attempt_history"][0]
    assert attempt["status"] == "failed"
    old_log = output / attempt["log"]["path"]
    assert old_log.read_text(encoding="utf-8") == "first attempt failed\n"
    assert attempt["log"]["sha256"] == hashlib.sha256(old_log.read_bytes()).hexdigest()
    assert json.loads((output / attempt["dispatch"]["path"]).read_text(encoding="utf-8")) == previous
    assert json.loads((output / attempt["failure"]["path"]).read_text(encoding="utf-8")) == {"error": "out of memory"}
    assert (output / "train.log").read_text(encoding="utf-8") == "second attempt stopped\n"
    old_log.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence changed"):
        runner.train_package(package, output, backend="gsplat", steps=20, resume=True,restart_from_step_zero=True)


def test_native_same_identity_restart_accepts_config_without_checkpoint(tmp_path, monkeypatch):
    from gsstudio.pipeline.training import native

    package = tmp_path / "package"
    package.mkdir()
    output = tmp_path / "experiment"
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(DefaultStrategy=object))
    monkeypatch.setattr(native.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(native, "validate_package", lambda _: {"images": [{"split": "train"}]})
    monkeypatch.setattr(native, "load_sparse_depth", lambda _: None)
    monkeypatch.setattr(native, "sha256_file", lambda _: DIGEST)

    def fail_initialization(_package):
        raise RuntimeError("simulated initialization failure")

    monkeypatch.setattr(native, "initialize", fail_initialization)
    with pytest.raises(RuntimeError, match="simulated initialization failure"):
        native.train(package, output, steps=20)
    assert (output / "config.json").is_file()
    assert not (output / "checkpoint.pt").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        native.train(package, output, steps=20)
    with pytest.raises(RuntimeError, match="simulated initialization failure"):
        native.train(package, output, steps=20, restart_from_step_zero=True)
    with pytest.raises(RuntimeError, match="configuration changed"):
        native.train(package, output, steps=20, photo_comp=False, restart_from_step_zero=True)

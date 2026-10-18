"""CPU-only probe/command contracts. These tests never launch CUDA training."""
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from local_postshot import project, training, studio_training
from local_postshot.studio_checkpoint import assert_restored
from local_postshot.studio_dataset import prepare


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_script("validate_studio_failures")
synthetic = load_script("studio_synthetic")


@pytest.fixture(autouse=True)
def cpu_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", str(tmp_path / "CPU-ONLY.lock"))
    runtime = tmp_path / "cpu-runtime.json"
    runtime.write_text(json.dumps({"gpu_lock_path":str(tmp_path / "CPU-ONLY.lock")}), encoding="utf-8")
    monkeypatch.setattr(probe, "RUNTIME", runtime)
    def forbidden(*args, **kwargs):
        pytest.fail("CPU failure-probe test attempted CUDA initialization")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert not torch.cuda.is_initialized()


def cpu_checkpoint(p, config, step=4):
    parameters = {name:torch.nn.Parameter(torch.ones(4, 3))
                  for name in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    optimizers = {}
    for name, parameter in parameters.items():
        optimizer = torch.optim.Adam([parameter])
        parameter.sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizers[name] = optimizer.state_dict()
    aux = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.Adam([aux])
    aux.sum().backward()
    optimizer.step()
    sampler = training.Sampler(p["studio_dataset"]["train_indices"])
    sampler.next()
    return {"schema":1, "step":step, "parameters":{k:v.detach().clone() for k,v in parameters.items()},
            "optimizers":optimizers, "schedulers":{"means":{"last_epoch":step}},
            "strategy_state":{"count":torch.ones(4)}, "strategy_config":{"a":1},
            "sampler":sampler.state_dict(), "rng_python":random.getstate(), "rng_numpy":np.random.get_state(),
            "rng_torch":torch.get_rng_state(), "rng_cuda":[torch.zeros(16, dtype=torch.uint8)],
            "config":p["settings"], "dataset_key":p["studio_dataset"]["identity"], "scene_scale":1.,
            "training_recipe":{"name":"quality_v2"}, "studio_experiment":config,
            "studio_auxiliary":{"schema_version":1, "identity":config,
                                "photometric":{"values":aux.detach()}, "optimizer":optimizer.state_dict()}}


def failed_experiment(tmp_path, monkeypatch):
    package = synthetic.make_package(tmp_path / "package", size=48)
    experiment = tmp_path / "experiment"
    # Exercise real studio status/config preparation, but replace the GPU scope
    # and training body. No production/shared lock can be acquired by this mock.
    monkeypatch.setattr("local_postshot.studio_bridge.gpu_session", lambda **kw:contextlib.nullcontext())
    def fake_train(path, *args, checkpoint_observer, **kwargs):
        p = project.load(path)
        config = json.loads((Path(path) / "config.json").read_text())
        training.atomic_checkpoint(Path(path) / "checkpoints/latest.pt", cpu_checkpoint(p, config))
        checkpoint_observer(4)
        raise RuntimeError("CPU mocked training failure")
    monkeypatch.setattr(training, "train", fake_train)
    with pytest.raises(RuntimeError, match="CPU mocked"):
        studio_training.run(package, experiment, **probe.run_options(12))
    return package, experiment


@pytest.mark.parametrize("arguments,code", [(["--help"], 0), ([], 2), (["--output", "unused"], 2)])
def test_cli_gate_imports_no_torch_or_core(tmp_path, arguments, code):
    # A fresh interpreter blocks imports, instead of relying on whether CUDA is
    # initialized already in pytest. subprocess here executes ONLY CLI parsing.
    source = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in ('torch', 'numpy', 'local_postshot', 'studio_synthetic'):
            raise AssertionError('gate imported '+name)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1]] + sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", source, str(SCRIPTS / "validate_studio_failures.py"), *arguments],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == code, result.stderr
    assert "gate imported" not in result.stderr
    assert not (tmp_path / "unused").exists()


def test_existing_output_is_refused_without_probe_execution(tmp_path, monkeypatch):
    sentinel = tmp_path / "user.txt"
    sentinel.write_text("keep")
    monkeypatch.setattr(probe, "probe", lambda *args:pytest.fail("Existing output reached probe"))
    assert probe.main(["--allow-cuda", "--output", str(tmp_path)]) == 1
    assert sentinel.read_text() == "keep"


def test_original_data_output_is_rejected_before_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ORIGINAL_DATA", tmp_path / "protected-data")
    output = tmp_path / "protected-data/new-probe"
    assert probe.main(["--allow-cuda", "--output", str(output)]) == 1
    assert not output.exists()


def test_child_environment_binds_runtime_lock_instead_of_stale_shell_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GSSTUDIO_GPU_LOCK_PATH", str(tmp_path / "wrong.lock"))
    environment = probe.child_environment()
    assert environment["GSSTUDIO_GPU_LOCK_PATH"] == str(tmp_path / "CPU-ONLY.lock")
    assert os.environ["GSSTUDIO_GPU_LOCK_PATH"] == str(tmp_path / "wrong.lock")


@pytest.mark.parametrize("option,value", [("--steps", "11"), ("--steps", "41"), ("--timeout-seconds", "301")])
def test_budget_gate_precedes_output_creation(tmp_path, option, value):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as result:
        probe.main(["--allow-cuda", "--output", str(output), option, value])
    assert result.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("kind", ["checkpoint_fsync", "checkpoint_replace"])
def test_scoped_syscall_failure_preserves_actual_checkpoint_bytes(tmp_path, kind):
    file = tmp_path / "checkpoints/latest.pt"
    original = {"step":4, "moments":torch.tensor([1., 2.])}
    training.atomic_checkpoint(file, original)
    before = file.read_bytes()
    with probe.checkpoint_fault(kind, file) as receipt:
        # Unrelated JSON fsync/replacement must still work in the injection scope.
        project.atomic_json(tmp_path / "unrelated.json", {"ok":True})
        with pytest.raises(OSError, match=probe.INJECTED):
            training.atomic_checkpoint(file, {"step":8, "moments":torch.tensor([3., 4.])})
    assert receipt["calls"] == 1
    assert file.read_bytes() == before
    assert_restored(original, training.load_checkpoint(file, map_location="cpu"))
    assert (file.with_name("latest.pt.tmp")).is_file()
    project.atomic_json(tmp_path / "status.json", {"status":"failed"})
    training.atomic_checkpoint(file, {"step":9})
    assert training.load_checkpoint(file, map_location="cpu")["step"] == 9


def test_optimizer_audit_requires_real_trained_moments(tmp_path):
    package = synthetic.make_package(tmp_path / "package")
    p = prepare(package, tmp_path / "out")
    state = cpu_checkpoint(p, {"test":"CPU only"})
    audit = probe.audit_optimizer_state(state, require_cuda=False)
    assert set(audit["optimizers"]) == set(state["parameters"]) | {"auxiliary"}
    assert audit["optimizers"]["means"]["nonzero_moment_elements"] > 0
    with pytest.raises(AssertionError, match="not on CUDA"):
        probe.audit_optimizer_state(state, require_cuda=True)
    for field in ("exp_avg", "exp_avg_sq", "step"):
        bad = copy.deepcopy(state)
        del next(iter(bad["optimizers"]["means"]["state"].values()))[field]
        with pytest.raises(AssertionError, match="missing Adam state"):
            probe.audit_optimizer_state(bad, require_cuda=False)
    bad = copy.deepcopy(state)
    next(iter(bad["studio_auxiliary"]["optimizer"]["state"].values()))["exp_avg"][0,0] = float("nan")
    with pytest.raises(AssertionError, match="Nonfinite"):
        probe.audit_optimizer_state(bad, require_cuda=False)


def test_restore_audit_uses_exact_recursive_state_and_negative_controls(tmp_path, monkeypatch):
    package = synthetic.make_package(tmp_path / "package")
    p = prepare(package, tmp_path / "out")
    saved = cpu_checkpoint(p, {"test":"CPU only"})
    real_audit = probe.audit_optimizer_state
    monkeypatch.setattr(probe, "audit_optimizer_state", lambda state, **kw:real_audit(state, require_cuda=False))
    proof = probe.restore_audit(saved, copy.deepcopy(saved))
    assert proof["all_state_bit_exact"] and proof["before_first_update"]
    assert len(proof["negative_controls"]) == 3
    current = copy.deepcopy(saved)
    current["rng_torch"][0] ^= 1
    with pytest.raises(AssertionError, match="rng_torch"):
        probe.restore_audit(saved, current)
    current = copy.deepcopy(saved)
    current["sampler"]["position"] += 1
    with pytest.raises(AssertionError, match="sampler"):
        probe.restore_audit(saved, current)


def test_actual_resume_identity_rejections_preserve_inputs_and_checkpoint(tmp_path, monkeypatch):
    package, experiment = failed_experiment(tmp_path, monkeypatch)
    other = synthetic.make_package(tmp_path / "other", size=50)
    inputs = probe.inventory(package), probe.inventory(other)
    before = probe.digest(experiment / "checkpoints/latest.pt")
    receipts = probe.reject_identities(package, other, experiment, tmp_path / "negative", 12)
    assert len(receipts) == 4 and all(row["rejected"] and not row["gpu_entered"] for row in receipts)
    assert inputs == (probe.inventory(package), probe.inventory(other))
    assert before == probe.digest(experiment / "checkpoints/latest.pt")
    assert probe.read_json(experiment / "training.json")["status"] == "failed"


def test_failed_status_and_last_good_bytes_are_required(tmp_path, monkeypatch):
    package, experiment = failed_experiment(tmp_path, monkeypatch)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    checkpoint = experiment / "checkpoints/latest.pt"
    (attempt / "last-good-before.pt").write_bytes(checkpoint.read_bytes())
    probe.write_json(attempt / "fault.json", {"last_good_sha256":probe.digest(checkpoint)})
    evidence = probe.interrupted_evidence("oom", experiment, attempt, 12)
    assert evidence["observed_status"] == "failed"
    assert (attempt / "interrupted-failure.json").is_file()
    assert (attempt / "restore-input.pt").read_bytes() == checkpoint.read_bytes()
    summary = probe.read_json(experiment / "training.json")
    summary["status"] = "succeeded"
    probe.write_json(experiment / "training.json", summary)
    with pytest.raises(AssertionError, match="Failure recorded as succeeded"):
        probe.interrupted_evidence("oom", experiment, attempt, 12)


def test_abrupt_status_is_preserved_unfinalized_without_manual_relabel(tmp_path, monkeypatch):
    package, experiment = failed_experiment(tmp_path, monkeypatch)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    checkpoint = experiment / "checkpoints/latest.pt"
    (attempt / "last-good-before.pt").write_bytes(checkpoint.read_bytes())
    probe.write_json(attempt / "fault.json", {"last_good_sha256":probe.digest(checkpoint)})
    for name in ("training.json", "control-state.json"):
        record = probe.read_json(experiment / name)
        record["status"] = "running"
        probe.write_json(experiment / name, record)
    (experiment / "failure.json").unlink()  # This synthetic CPU fixture only.
    before = {name:(experiment / name).read_bytes() for name in ("training.json", "control-state.json")}
    evidence = probe.interrupted_evidence("abrupt_exit", experiment, attempt, 12)
    assert evidence["abrupt_status_is_unfinalized"] and evidence["observed_status"] == "running"
    assert before == {name:(experiment / name).read_bytes() for name in before}
    assert not (experiment / "failure.json").exists()


@pytest.mark.parametrize("case", ["external_cancel", "external_save_pause", "control_stop"])
def test_parent_owned_command_and_external_signal(tmp_path, monkeypatch, case):
    attempt = tmp_path / "attempt"
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    observed = {}
    if case == "control_stop":
        from local_postshot.studio_control import TrainingControl
        controller = TrainingControl(experiment, "a" * 64)
        controller.observe(5, lambda:None)

    class FakeProcess:
        pid = 123456789
        returncode = None
        polls = 0
        def __init__(self, command, **kwargs):
            observed.update(command=command, options=kwargs)
            probe.write_json(attempt / "ready.json", {"case":case, "step":5})
        def poll(self):
            self.polls += 1
            if self.polls > 1:
                self.returncode = 0
            return self.returncode
        def kill(self):
            pytest.fail("Completed owned child should not be killed")

    monkeypatch.setattr(probe.subprocess, "Popen", FakeProcess)
    job = {"case":case, "phase":"interrupt", "attempt":str(attempt), "experiment":str(experiment)}
    result = probe.run_child(job, time.monotonic() + 10)
    assert result["returncode"] == 0 and result["external_signal_sent"]
    assert observed["command"][:3] == [sys.executable, str(SCRIPTS / "validate_studio_failures.py"), "--allow-cuda"]
    assert observed["options"]["env"]["PYTHONPATH"] == str(SCRIPTS.parent)
    assert observed["options"]["env"]["GSSTUDIO_RUNTIME"] == str(probe.RUNTIME)
    assert observed["options"]["env"]["GSSTUDIO_GPU_LOCK_PATH"] == str(tmp_path / "CPU-ONLY.lock")
    assert "shell" not in observed["options"]
    if case == "control_stop":
        request = probe.read_json(experiment / "control-request.json")
        assert request["action"] == "stop" and request["generation_id"] == controller.generation_id
    else:
        assert probe.read_json(attempt / "external-request.json") == {"action":case, "pid":FakeProcess.pid}


def test_timeout_kills_only_owned_popen_handle_and_preserves_logs(tmp_path, monkeypatch):
    calls = []
    clock = [0.]
    monkeypatch.setattr(probe.time, "monotonic", lambda:clock[0])
    class FakeProcess:
        pid = 123456789
        returncode = None
        def __init__(self, *args, **kwargs):
            clock[0] = 2.
        def poll(self):
            return self.returncode
        def kill(self):
            calls.append(("kill", self.pid))
            self.returncode = -9
        def wait(self, timeout):
            calls.append(("wait", timeout))
            return self.returncode
    monkeypatch.setattr(probe.subprocess, "Popen", FakeProcess)
    attempt = tmp_path / "attempt"
    with pytest.raises(TimeoutError, match="runtime budget"):
        probe.run_child({"attempt":str(attempt)}, 1.)
    assert calls == [("kill", FakeProcess.pid), ("wait", 10)]
    assert (attempt / "child.log").exists() and (attempt / "job.json").exists()
    assert probe.read_json(attempt / "timeout.json")["owned_pid"] == FakeProcess.pid


def test_expired_budget_cannot_launch_another_process(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *args, **kwargs:pytest.fail("Expired budget launched child"))
    with pytest.raises(TimeoutError, match="before starting another child"):
        probe.run_child({"attempt":str(tmp_path / "never-started")}, time.monotonic() - 1)
    assert not (tmp_path / "never-started").exists()


def test_inventory_detects_any_input_mutation(tmp_path):
    package = synthetic.make_package(tmp_path / "package")
    initial = probe.inventory(package)
    # Mutate only this CPU-test fixture to prove detection, never a probe input.
    (package / "extra.txt").write_text("mutation")
    assert probe.inventory(package) != initial


def test_orchestration_schedules_fresh_interrupt_and_resume_for_all_cases(tmp_path, monkeypatch):
    jobs = []
    monkeypatch.setattr(probe, "run_child", lambda job, deadline:launch(job))
    monkeypatch.setattr(probe, "interrupted_evidence", lambda *args:{"restore_input":"CPU-only.pt", "generation_id":"interrupted"})
    monkeypatch.setattr(probe, "reject_identities", lambda *args:[{"CPU_mock":True}])
    # Imports inside probe() must find the existing fixture without path changes.
    monkeypatch.setitem(sys.modules, "studio_synthetic", synthetic)
    def launch(job):
        jobs.append(job.copy())
        if job["phase"] == "resume":
            attempt = Path(job["attempt"])
            probe.write_json(attempt / "result.json", {"status":"recovered", "restore":{"all_state_bit_exact":True},
                                                      "result":{"summary":{"generation_id":"new"}}})
            return {"pid":len(jobs), "returncode":0}
        code = probe.ABRUPT_EXIT if job["case"] == "abrupt_exit" else probe.EXPECTED_FAULT if job["case"] in ("oom", "checkpoint_fsync", "checkpoint_replace") else 0
        return {"pid":len(jobs), "returncode":code}
    report = probe.probe(tmp_path, 12, 300)
    assert report["status"] == "passed" and report["input_unchanged"]
    assert len(jobs) == 2 * len(probe.CASES)
    for interrupt, resume in zip(jobs[::2], jobs[1::2]):
        assert interrupt["case"] == resume["case"] and interrupt["phase"] == "interrupt" and resume["phase"] == "resume"
        assert interrupt["experiment"] == resume["experiment"] and interrupt["steps"] == resume["steps"] == 12
        assert interrupt["owned_abrupt_exit_fixture"] == (interrupt["case"] == "abrupt_exit")
    assert all(row["status"] == "passed" for row in report["cases"])


def test_orchestration_retains_failure_instead_of_weakening_exit_assertion(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "studio_synthetic", synthetic)
    monkeypatch.setattr(probe, "run_child", lambda *args:{"pid":123456789, "returncode":1})
    with pytest.raises(AssertionError, match="oom: child exited 1"):
        probe.probe(tmp_path, 12, 300)
    report = probe.read_json(tmp_path / "validation.json")
    assert report["status"] == "failed" and report["input_unchanged"]
    assert "failure-requests.md" in report["action"]


@pytest.mark.parametrize("case", ["oom", "checkpoint_fsync", "checkpoint_replace"])
def test_child_faults_hit_real_cpu_checkpoint_writer_and_expected_exception(tmp_path, monkeypatch, case):
    package = synthetic.make_package(tmp_path / "package")
    experiment, attempt = tmp_path / "out", tmp_path / "attempt"
    attempt.mkdir()
    p = prepare(package, experiment)
    state = cpu_checkpoint(p, {"test":"CPU mock"})
    real_audit = probe.audit_optimizer_state
    monkeypatch.setattr(probe, "audit_optimizer_state", lambda value, **kw:real_audit(value, require_cuda=False))
    def fake_run(package, output, *, boundary_observer, **kwargs):
        training.atomic_checkpoint(experiment / "checkpoints/latest.pt", state)
        if case.startswith("checkpoint_"):
            training.atomic_checkpoint(experiment / "checkpoints/latest.pt", {**state, "step":8})
        else:
            boundary_observer({}, {"step":4, "snapshot":lambda:state})
        pytest.fail("Injected exception did not interrupt mocked training")
    monkeypatch.setattr(studio_training, "run", fake_run)
    job = {"case":case, "phase":"interrupt", "attempt":str(attempt), "experiment":str(experiment), "package":str(package), "steps":12}
    assert probe.child_main(job) == probe.EXPECTED_FAULT
    fault = probe.read_json(attempt / "fault.json")
    assert fault["last_good_step"] == 4 and fault["safe_boundary"]
    assert probe.digest(experiment / "checkpoints/latest.pt") == fault["last_good_sha256"]
    assert probe.read_json(attempt / "result.json")["status"] == "expected_fault"
    if case.startswith("checkpoint_"):
        assert probe.read_json(attempt / "syscall.json")["calls"] == 1


@pytest.mark.parametrize("allowed", [False, True])
def test_abrupt_exit_fixture_requires_explicit_ownership_and_preserves_evidence(tmp_path, monkeypatch, allowed):
    package = synthetic.make_package(tmp_path / "package")
    experiment, attempt = tmp_path / "out", tmp_path / "attempt"
    attempt.mkdir()
    p = prepare(package, experiment)
    state = cpu_checkpoint(p, {"test":"CPU mock"})
    real_audit = probe.audit_optimizer_state
    monkeypatch.setattr(probe, "audit_optimizer_state", lambda value, **kw:real_audit(value, require_cuda=False))
    events = []
    def fake_exit(code):
        # Never abruptly terminate pytest. Verify the durable marker exists at
        # the instant the child would die in the parent-authorized CUDA run.
        assert (attempt / "fault.json").is_file()
        assert (attempt / "last-good-before.pt").is_file()
        events.append(("owned_exit", code))
        raise RuntimeError("CPU intercepted owned exit")
    monkeypatch.setattr(probe.os, "_exit", fake_exit)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda:events.append("drained"))
    def fake_run(package, output, *, boundary_observer, **kwargs):
        training.atomic_checkpoint(experiment / "checkpoints/latest.pt", state)
        boundary_observer({}, {"step":4, "snapshot":lambda:state})
    monkeypatch.setattr(studio_training, "run", fake_run)
    job = {"case":"abrupt_exit", "phase":"interrupt", "attempt":str(attempt), "experiment":str(experiment),
           "package":str(package), "steps":12, "owned_abrupt_exit_fixture":allowed}
    assert probe.child_main(job) == 1  # The intercepted exit is never GPU evidence.
    assert events == (["drained", ("owned_exit", probe.ABRUPT_EXIT)] if allowed else [])
    if not allowed:
        assert "Explicit owned-child" in probe.read_json(attempt / "result.json")["traceback"]


@pytest.mark.parametrize("case", ["external_cancel", "external_save_pause", "control_stop"])
def test_child_external_requests_set_real_events_and_save_boundary(tmp_path, monkeypatch, case):
    from local_postshot.studio_control import TrainingControl, submit_control
    package = synthetic.make_package(tmp_path / "package")
    experiment, attempt = tmp_path / "out", tmp_path / "attempt"
    attempt.mkdir()
    p = prepare(package, experiment)
    state = cpu_checkpoint(p, {"test":"CPU mock"})
    real_audit = probe.audit_optimizer_state
    monkeypatch.setattr(probe, "audit_optimizer_state", lambda value, **kw:real_audit(value, require_cuda=False))
    errors = []
    def sender():
        try:
            probe.wait_for(lambda:(attempt / "ready.json").exists(), "CPU control gate")
            if case == "control_stop":
                submit_control(experiment, "a" * 64, "stop")
                probe.write_json(attempt / "release.json", {})
            else:
                probe.write_json(attempt / "external-request.json", {"action":case, "pid":os.getpid()})
        except BaseException as error:
            errors.append(error)
    def fake_run(package, output, *, boundary_observer, external_control, external_pause, **kwargs):
        controller = TrainingControl(experiment, "a" * 64)
        training.atomic_checkpoint(experiment / "checkpoints/latest.pt", state)
        boundary_observer({}, {"step":5, "snapshot":lambda:{**state, "step":5}})
        def save():
            training.atomic_checkpoint(experiment / "checkpoints/latest.pt", {**state, "step":5})
        if case == "control_stop":
            assert controller.observe(5, save) == "stop"
        else:
            assert isinstance(external_control, threading.Event) and external_control.is_set()
            assert external_pause.is_set() == (case == "external_save_pause")
            save()
        return {"status":"paused" if case == "external_save_pause" else "stopped", "completed_steps":5}
    monkeypatch.setattr(studio_training, "run", fake_run)
    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()
    try:
        assert probe.child_main({"case":case, "phase":"interrupt", "attempt":str(attempt), "experiment":str(experiment),
                                 "package":str(package), "steps":12}) == 0
    finally:
        sender_thread.join(15)
    assert not sender_thread.is_alive() and not errors
    assert training.load_checkpoint(experiment / "checkpoints/latest.pt", map_location="cpu")["step"] == 5
    assert probe.read_json(attempt / "result.json")["status"] == "interrupted"


def test_resuming_child_audits_first_boundary_once_without_reinjecting(tmp_path, monkeypatch):
    package = synthetic.make_package(tmp_path / "package")
    experiment, attempt = tmp_path / "out", tmp_path / "attempt"
    attempt.mkdir()
    p = prepare(package, experiment)
    state = cpu_checkpoint(p, {"test":"CPU mock"})
    restore_input = tmp_path / "restore-input.pt"
    training.atomic_checkpoint(restore_input, state)
    real_audit = probe.audit_optimizer_state
    monkeypatch.setattr(probe, "audit_optimizer_state", lambda value, **kw:real_audit(value, require_cuda=False))
    monkeypatch.setattr(probe, "completed_evidence", lambda *args:{"summary":{"status":"succeeded"}, "CPU_mock":True})
    snapshots = []
    def snapshot():
        snapshots.append(4)
        return copy.deepcopy(state)
    def fake_run(package, output, *, boundary_observer, resume, external_control, **kwargs):
        assert resume and not external_control.is_set()
        boundary_observer({}, {"step":4, "snapshot":snapshot})
        boundary_observer({}, {"step":5, "snapshot":lambda:pytest.fail("Second restore audit after update")})
        # The disk-fault case name must not re-arm injection during recovery.
        training.atomic_checkpoint(experiment / "checkpoints/latest.pt", {**state, "step":8})
        return {"status":"succeeded"}
    monkeypatch.setattr(studio_training, "run", fake_run)
    assert probe.child_main({"case":"checkpoint_fsync", "phase":"resume", "attempt":str(attempt),
                             "experiment":str(experiment), "package":str(package), "steps":12,
                             "restore_input":str(restore_input)}) == 0
    assert snapshots == [4]
    assert (attempt / "restored-pre-update.pt").exists()
    result = probe.read_json(attempt / "result.json")
    assert result["status"] == "recovered" and result["restore"]["all_state_bit_exact"]
    assert not result["fault_injection_requested"] and not (attempt / "fault.json").exists()


def completed_cpu_experiment(tmp_path, monkeypatch):
    from local_postshot.formats import write_ply
    package, experiment = failed_experiment(tmp_path, monkeypatch)
    p = project.load(experiment)
    config = probe.read_json(experiment / "config.json")
    training.atomic_checkpoint(experiment / "checkpoints/latest.pt", cpu_checkpoint(p, config, step=12))
    cloud = {"means":np.ones((4,3), np.float32), "scales":np.zeros((4,3), np.float32),
             "quats":np.tile(np.array([1.,0.,0.,0.], np.float32), (4,1)), "opacities":np.zeros(4, np.float32),
             "sh":np.zeros((4,16,3), np.float32)}
    write_ply(experiment / "model.ply", cloud)
    evaluation = experiment / "evaluation"
    evaluation.mkdir()
    images = []
    for index in p["studio_dataset"]["test_indices"]:
        render = evaluation / f"{index}.png"
        render.write_bytes(b"CPU mocked evaluation artifact")
        images.append({"index":index, "image":p["reconstruction"]["cameras"][index]["image"],
                       "psnr":20., "ssim":.9, "render_file":render.name, "render_sha256":probe.digest(render)})
    metrics = {"schema_version":1, "package_sha256":config["package_sha256"], "model_sha256":probe.digest(experiment / "model.ply"),
               "images":images, "averages":{"psnr":20., "ssim":.9}}
    probe.write_json(evaluation / "metrics.json", metrics)
    summary = {**config, "status":"succeeded", "completed_steps":12, "target":12, "evaluation_status":"completed",
               "checkpoint_sha256":probe.digest(experiment / "checkpoints/latest.pt"), "model_sha256":metrics["model_sha256"],
               "evaluation_sha256":probe.digest(evaluation / "metrics.json")}
    probe.write_json(experiment / "training.json", summary)
    return experiment


def test_completion_requires_and_reads_finite_ply_spz_and_hashed_metrics(tmp_path, monkeypatch):
    experiment = completed_cpu_experiment(tmp_path, monkeypatch)
    result = probe.completed_evidence(experiment, 12)
    assert result["finite_ply_spz"] and result["heldout_images"] == 2
    assert result["spz_sha256"] == probe.digest(experiment / "recovered.spz")


@pytest.mark.parametrize("defect", ["failed_status", "short_budget", "missing_metric_row", "nonfinite_metrics", "wrong_checkpoint_hash"])
def test_completion_rejects_incomplete_or_nonfinite_evidence(tmp_path, monkeypatch, defect):
    experiment = completed_cpu_experiment(tmp_path, monkeypatch)
    summary = probe.read_json(experiment / "training.json")
    metrics_file = experiment / "evaluation/metrics.json"
    metrics = probe.read_json(metrics_file)
    if defect == "failed_status":
        summary["status"] = "failed"
    elif defect == "short_budget":
        summary["completed_steps"] = 11
    elif defect == "wrong_checkpoint_hash":
        summary["checkpoint_sha256"] = "0" * 64
    else:
        if defect == "missing_metric_row":
            metrics["images"].pop()
        else:
            metrics["images"][0]["psnr"] = float("nan")
        metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
        summary["evaluation_sha256"] = probe.digest(metrics_file)
    probe.write_json(experiment / "training.json", summary)
    with pytest.raises(AssertionError):
        probe.completed_evidence(experiment, 12)

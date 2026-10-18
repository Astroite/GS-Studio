"""Parent-run real CUDA failure/recovery probe; explicit --allow-cuda required.

Only synthetic inputs in a NEW --output directory are used. Faults are injected,
NOT real GPU/disk exhaustion. No production code or numerical policy is changed.
Every recovery uses a fresh process and audits the entire saved state BEFORE its
first optimizer update. CPU tests/mocks are not GPU acceptance evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch


ENGINE = Path(__file__).resolve().parents[1]
RUNTIME = Path("E:/AI/GSStudio/coordination/validation-runtime.json")
ORIGINAL_DATA = ENGINE.parent.parent / "Data"
CASES = ("oom", "checkpoint_fsync", "checkpoint_replace", "abrupt_exit",
         "external_cancel", "external_save_pause", "control_stop")
EXPECTED_FAULT = 17
ABRUPT_EXIT = 73
INJECTED = "GSSTUDIO_FAILURE_PROBE_INJECTED"
SETTINGS = {"max_splats":64, "checkpoint_every":4, "preview_every":1000,
            "sh_degree_interval":2, "refine_start":100000, "refine_stop":0,
            "reset_every":3000}


def read_json(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write_json(file, value):
    """Durable probe evidence, separate from the production status writer."""
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(file)


def digest(file):
    value = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root):
    root = Path(root)
    return {file.relative_to(root).as_posix():digest(file)
            for file in sorted(root.rglob("*")) if file.is_file()}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_for(predicate, description, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise TimeoutError(description)


@contextlib.contextmanager
def checkpoint_fault(kind, checkpoint):
    """Replace one syscall ONLY inside the real checkpoint writer's call scope.

    fsync additionally matches the checkpoint temporary file's inode/device;
    replacement additionally matches both exact resolved paths. Other IO keeps
    working, including the core's failure/status evidence writes.
    """
    checkpoint = Path(checkpoint).resolve()
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    receipt = {"kind":kind, "calls":0}
    fsync, replace = os.fsync, Path.replace

    def fail_fsync(fd):
        try:
            expected = temporary.stat()
        except FileNotFoundError:
            return fsync(fd)
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            receipt["calls"] += 1
            raise OSError(errno.ENOSPC, INJECTED + ": checkpoint fsync; no disk exhaustion")
        return fsync(fd)

    def fail_replace(source, destination):
        if Path(source).resolve() == temporary and Path(destination).resolve() == checkpoint:
            receipt["calls"] += 1
            raise OSError(errno.EIO, INJECTED + ": checkpoint replace; no disk exhaustion")
        return replace(source, destination)

    if kind == "checkpoint_fsync":
        manager = patch.object(os, "fsync", fail_fsync)
    elif kind == "checkpoint_replace":
        manager = patch.object(Path, "replace", fail_replace)
    else:
        raise ValueError("Unknown checkpoint fault")
    with manager:
        yield receipt


def finite_tree(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint tensor")
    elif isinstance(value, np.ndarray):
        require(bool(np.isfinite(value).all()), "Nonfinite checkpoint array")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite_tree(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite checkpoint scalar")


def audit_optimizer_state(state, *, require_cuda):
    """Validate actual Gaussian/aux Adam moments; never infer from rendered RGB."""
    import torch
    finite_tree(state)
    parameters = state["parameters"]
    require(set(parameters) == set(state["optimizers"]), "Optimizer/parameter keys differ")
    require(bool(state["schedulers"]), "Missing scheduler state")
    require(bool(state["rng_cuda"]), "Missing saved CUDA RNG")
    rows = {}

    def audit(name, optimizer, expected_shape=None):
        ids = [p for group in optimizer["param_groups"] for p in group["params"]]
        require(bool(ids) and set(ids) == set(optimizer["state"]), name + ": incomplete moments")
        nonzero = 0
        devices = set()
        for key in ids:  # Optimizer dict keys are integers, not path strings.
            item = optimizer["state"][key]
            require({"step", "exp_avg", "exp_avg_sq"} <= item.keys(), name + ": missing Adam state")
            require(float(item["step"]) > 0, name + ": uninitialized optimizer")
            for field in ("exp_avg", "exp_avg_sq"):
                tensor = item[field]
                require(isinstance(tensor, torch.Tensor), name + ": moment is not a tensor")
                if expected_shape is not None:
                    require(tensor.shape == expected_shape, name + ": moment shape differs")
                if require_cuda:
                    require(tensor.is_cuda, name + ": moment not on CUDA")
                devices.add(str(tensor.device))
                nonzero += int(torch.count_nonzero(tensor))
        rows[name] = {"parameters":len(ids), "nonzero_moment_elements":nonzero,
                      "moment_devices":sorted(devices)}

    for name, parameter in parameters.items():
        if require_cuda:
            require(parameter.is_cuda, name + ": parameter not on CUDA")
        audit(name, state["optimizers"][name], parameter.shape)
    auxiliary = state["studio_auxiliary"]
    require(bool(auxiliary["photometric"]) and bool(auxiliary["optimizer"]), "Missing auxiliary state")
    if require_cuda:
        require(all(tensor.is_cuda for tensor in auxiliary["photometric"].values()),
                "Auxiliary parameters/buffers not on CUDA")
    audit("auxiliary", auxiliary["optimizer"])
    require(rows["means"]["nonzero_moment_elements"] > 0, "No trained Gaussian position moments")
    require(rows["auxiliary"]["nonzero_moment_elements"] > 0, "No trained auxiliary moments")
    return {"step":state["step"], "cuda_required":require_cuda, "optimizers":rows,
            "schedulers":list(state["schedulers"]), "cuda_rng_devices":len(state["rng_cuda"])}


def restore_audit(saved, current):
    import copy
    import torch
    from local_postshot.studio_checkpoint import assert_restored
    assert_restored(saved, current)
    proof = audit_optimizer_state(current, require_cuda=True)
    # Negative controls catch optimizers being omitted, even if images look alike.
    for field in ("optimizers", "studio_auxiliary"):
        broken = copy.deepcopy(current)
        broken[field] = {}
        try:
            assert_restored(saved, broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("Exact restore audit missed absent " + field)
    broken = copy.deepcopy(current)
    moment = next(iter(broken["optimizers"]["means"]["state"].values()))
    require(bool(torch.count_nonzero(moment["exp_avg_sq"])), "Zero position second moment fixture")
    moment["exp_avg_sq"].zero_()
    try:
        assert_restored(saved, broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Exact restore audit missed zeroed optimizer moment")
    proof.update(all_state_bit_exact=True, before_first_update=True,
                 negative_controls=["absent Gaussian optimizers", "absent auxiliary state", "zeroed position second moment"])
    return proof


def completed_evidence(experiment, steps):
    import numpy as np
    from local_postshot import project
    from local_postshot.formats import read_ply, read_spz, write_spz
    from local_postshot.training import load_checkpoint
    from local_postshot.studio_training import validate_checkpoint
    experiment = Path(experiment)
    summary = read_json(experiment / "training.json")
    config = read_json(experiment / "config.json")
    p = project.load(experiment)
    require(summary["status"] == "succeeded", "Recovery did not succeed")
    require(summary["completed_steps"] == summary["target"] == summary["steps"] == steps,
            "Recovered run did not complete the unchanged target")
    require(summary["evaluation_status"] == "completed", "Evaluation not complete")
    require(summary["checkpoint_sha256"] == digest(experiment / "checkpoints/latest.pt"), "Final checkpoint hash mismatch")
    require(summary["model_sha256"] == digest(experiment / "model.ply"), "Final export hash mismatch")
    state = load_checkpoint(experiment / "checkpoints/latest.pt", map_location="cpu")
    validate_checkpoint(state, config, p["studio_dataset"]["identity"], p["settings"], p["studio_dataset"]["train_indices"])
    require(state["step"] == steps, "Final checkpoint short of target")
    audit = audit_optimizer_state(state, require_cuda=False)
    metrics_file = experiment / "evaluation/metrics.json"
    require(summary["evaluation_sha256"] == digest(metrics_file), "Metrics hash mismatch")
    metrics = read_json(metrics_file)
    require(metrics["schema_version"] == 1 and metrics["package_sha256"] == config["package_sha256"], "Metrics identity mismatch")
    require(metrics["model_sha256"] == summary["model_sha256"], "Metrics model mismatch")
    expected = p["studio_dataset"]["test_indices"]
    require([row["index"] for row in metrics["images"]] == expected and bool(expected), "Missing heldout metrics")
    for row in metrics["images"]:
        require(row["image"] == p["reconstruction"]["cameras"][row["index"]]["image"], "Metric source mismatch")
        require(all(math.isfinite(row[k]) for k in ("psnr", "ssim")), "Nonfinite metric")
        require(digest(experiment / "evaluation" / row["render_file"]) == row["render_sha256"], "Evaluation render hash mismatch")
    require(all(math.isfinite(metrics["averages"][k]) for k in ("psnr", "ssim")), "Nonfinite average metrics")
    cloud = read_ply(experiment / "model.ply")
    require(len(cloud["means"]) > 0, "Empty recovered PLY")
    for value in cloud.values():
        require(bool(np.isfinite(value).all()), "Nonfinite PLY")
    spz = experiment / "recovered.spz"
    require(not spz.exists(), "SPZ evidence must be new")
    write_spz(spz, cloud)
    decoded = read_spz(spz)
    require(len(decoded["means"]) == len(cloud["means"]), "SPZ count mismatch")
    for value in decoded.values():
        require(bool(np.isfinite(value).all()), "Nonfinite SPZ")
    return {"summary":summary, "final_optimizer_audit":audit, "heldout_images":len(expected),
            "finite_ply_spz":True, "spz_sha256":digest(spz)}


def run_options(steps):
    return {"steps":steps, "recipe":"quality_v2", "photo_comp":True,
            "use_bilateral_grid":True, "use_sparse_depth":True, "preview_every":0,
            "settings":SETTINGS.copy(), "emit":lambda *args:None}


def reject_identities(package, other_package, experiment, destination, steps):
    """Actual run(resume=True) rejection; neither path can enter a GPU session."""
    from local_postshot import project, studio_training
    from local_postshot.studio_dataset import prepare
    from local_postshot.training import load_checkpoint
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    original = Path(experiment)
    protected = ("config.json", "project.json", "training.json", "checkpoints/latest.pt")
    before = {key:digest(original / key) for key in protected}
    receipts = []

    def forbidden_gpu(*args, **kwargs):
        raise AssertionError("Identity mismatch reached GPU session")

    def rejection(name, dataset, output, expected):
        try:
            studio_training.run(dataset, output, resume=True, **run_options(steps))
        except ValueError as error:
            require(expected in str(error), name + ": unexpected rejection: " + str(error))
            receipts.append({"case":name, "rejected":True, "reason":str(error), "gpu_entered":False})
        else:
            raise AssertionError(name + ": mismatched identity accepted")

    with patch("local_postshot.studio_bridge.gpu_session", forbidden_gpu):
        rejection("different_immutable_input", other_package, original, "identity")
        require(before == {key:digest(original / key) for key in protected}, "Input rejection changed existing experiment")
        # The foreign checkpoint stays byte-identical. The new experiment has a
        # correctly bound config, so rejection must come from checkpoint identity.
        foreign = destination / "foreign-checkpoint"
        p = prepare(package, foreign, {**SETTINGS, "steps":steps, "training_recipe":"quality_v2"})
        config = read_json(original / "config.json")
        config["checkpoint"] = str(foreign.resolve() / "checkpoints/latest.pt")
        project.atomic_json(foreign / "config.json", config)
        shutil.copyfile(original / "checkpoints/latest.pt", foreign / "checkpoints/latest.pt")
        foreign_sha = digest(foreign / "checkpoints/latest.pt")
        rejection("foreign_checkpoint", package, foreign, "Checkpoint experiment/config/dataset identity mismatch")
        require(digest(foreign / "checkpoints/latest.pt") == foreign_sha, "Rejected foreign checkpoint changed")
        # Also exercise dataset-key and auxiliary-identity guards directly with
        # a real durable payload; never edit the durable checkpoint to do so.
        state = load_checkpoint(original / "checkpoints/latest.pt", map_location="cpu")
        own = project.load(original)
        own_config = read_json(original / "config.json")
        for field in ("dataset_key", "studio_auxiliary"):
            broken = dict(state)
            if field == "dataset_key":
                broken[field] = "foreign-dataset"
            else:
                broken[field] = {**state[field], "identity":{}}
            try:
                studio_training.validate_checkpoint(broken, own_config, own["studio_dataset"]["identity"], own["settings"], own["studio_dataset"]["train_indices"])
            except ValueError as error:
                receipts.append({"case":field, "rejected":True, "reason":str(error), "gpu_entered":False})
            else:
                raise AssertionError("Accepted mismatched " + field)
    require(before == {key:digest(original / key) for key in protected}, "Identity probes changed source experiment")
    write_json(destination / "rejections.json", receipts)
    return receipts


def exception_chain(error):
    values = []
    while error is not None and len(values) < 10:
        values.append({"type":type(error).__name__, "message":str(error)})
        error = error.__cause__ or error.__context__
    return values


def child_main(job):
    """Internal owned-child fixture; called only after the CLI authorization gate."""
    import torch
    from local_postshot import training, studio_training
    case, phase = job["case"], job["phase"]
    attempt, experiment = Path(job["attempt"]), Path(job["experiment"])
    checkpoint = experiment / "checkpoints/latest.pt"
    package = Path(job["package"])
    receipt = {"case":case, "phase":phase, "pid":os.getpid(), "synthetic_fixture":True,
               "fault_injection_requested":phase == "interrupt" and case in CASES[:4],
               "deliberate_resource_exhaustion":False, "torch":torch.__version__}
    write_json(attempt / "started.json", receipt)
    cancel, pause, done = threading.Event(), threading.Event(), threading.Event()
    monitor = None
    fired = []
    restored = []

    def retain_fault(step, live):
        saved = training.load_checkpoint(checkpoint, map_location="cpu")
        require(saved["step"] == 4, "Expected trained last-good checkpoint at step 4")
        disk_audit = audit_optimizer_state(saved, require_cuda=False)
        live_audit = audit_optimizer_state(live, require_cuda=True)
        if step == 4:
            from local_postshot.studio_checkpoint import assert_restored
            assert_restored(saved, live)
        value = {**receipt, "boundary_step":step, "last_good_step":saved["step"],
                 "last_good_sha256":digest(checkpoint), "last_good_optimizer_audit":disk_audit,
                 "live_gpu_optimizer_audit":live_audit, "safe_boundary":True,
                 "post_checkpoint_live_exact":step == 4}
        shutil.copyfile(checkpoint, attempt / "last-good-before.pt")
        require(digest(attempt / "last-good-before.pt") == value["last_good_sha256"], "Evidence copy differs")
        write_json(attempt / "fault.json", value)
        fired.append(value)

    saved = training.load_checkpoint(job["restore_input"], map_location="cpu") if phase == "resume" else None

    def boundary(params, context):
        step = context["step"]
        if phase == "resume":
            if not restored:
                require(step == saved["step"], "Resumed from wrong checkpoint step")
                current = context["snapshot"]()
                proof = restore_audit(saved, current)
                torch.save(current, attempt / "restored-pre-update.pt")
                write_json(attempt / "restore-boundary.json", proof)
                restored.append(proof)
            return
        if case in ("checkpoint_fsync", "checkpoint_replace") or fired:
            return
        trigger = 5 if case in ("external_cancel", "external_save_pause", "control_stop") else 4
        if step != trigger:
            return
        retain_fault(step, context["snapshot"]())
        if case == "oom":
            raise torch.cuda.OutOfMemoryError(INJECTED + ": OOM at post-checkpoint safe boundary; no GPU exhaustion")
        if case == "abrupt_exit":
            require(job.get("owned_abrupt_exit_fixture") is True, "Explicit owned-child abrupt-exit fixture required")
            # Drain completed CUDA work and durably write evidence, then emulate
            # process death without Python cleanup/production status rewriting.
            torch.cuda.synchronize()
            os._exit(ABRUPT_EXIT)
        write_json(attempt / "ready.json", {"case":case, "step":step})
        if case == "control_stop":
            wait_for(lambda:(attempt / "release.json").is_file(), "Parent stop request", 15)
            time.sleep(.16)  # Explicit fixture wait for the documented control IO poll.
        else:
            wait_for(lambda:cancel.is_set(), "External cancellation event", 15)

    original_writer = training.atomic_checkpoint

    def writer(file, state):
        if phase == "interrupt" and case in ("checkpoint_fsync", "checkpoint_replace") and state["step"] == 8 and not fired:
            require(Path(file).resolve() == checkpoint.resolve(), "Unexpected checkpoint destination")
            retain_fault(state["step"], state)
            try:
                with checkpoint_fault(case, checkpoint) as call:
                    return original_writer(file, state)
            finally:
                write_json(attempt / "syscall.json", call)
        return original_writer(file, state)

    def external_monitor():
        signal = attempt / "external-request.json"
        while not done.wait(.01):
            if signal.is_file():
                value = read_json(signal)
                require(value == {"action":case, "pid":os.getpid()}, "External request belongs to another process")
                if case == "external_save_pause":
                    pause.set()
                cancel.set()
                return

    if phase == "interrupt" and case in ("external_cancel", "external_save_pause"):
        monitor = threading.Thread(target=external_monitor, daemon=True)
        monitor.start()
    try:
        with patch.object(training, "atomic_checkpoint", writer):
            result = studio_training.run(package, experiment, resume=phase == "resume",
                                         boundary_observer=boundary, external_control=cancel,
                                         external_pause=pause, **run_options(job["steps"]))
        if phase == "resume":
            require(bool(restored), "Missing pre-update restoration audit")
            receipt.update(status="recovered", result=completed_evidence(experiment, job["steps"]), restore=restored[0])
        else:
            require(case in ("external_cancel", "external_save_pause", "control_stop") and bool(fired), "Fault did not interrupt training")
            expected = "paused" if case == "external_save_pause" else "stopped"
            require(result["status"] == expected and 0 < result["completed_steps"] < job["steps"], "Controlled failure became success")
            receipt.update(status="interrupted", result=result)
        write_json(attempt / "result.json", receipt)
        return 0
    except BaseException as error:
        chain = exception_chain(error)
        expected = phase == "interrupt" and bool(fired) and (
            (case == "oom" and any(row["type"] == "OutOfMemoryError" and INJECTED in row["message"] for row in chain)) or
            (case in ("checkpoint_fsync", "checkpoint_replace") and isinstance(error, OSError) and INJECTED in str(error)))
        receipt.update(status="expected_fault" if expected else "probe_failed", exception_chain=chain, traceback=traceback.format_exc())
        write_json(attempt / "result.json", receipt)
        return EXPECTED_FAULT if expected else 1
    finally:
        done.set()
        if monitor is not None:
            monitor.join(1)


def child_environment():
    environment = os.environ.copy()
    configured_lock = Path(read_json(RUNTIME)["gpu_lock_path"])
    if not configured_lock.is_absolute():
        configured_lock = RUNTIME.parent / configured_lock
    environment.update(PYTHONPATH=str(ENGINE), PYTHONNOUSERSITE="1", PYTHONUTF8="1",
                       PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", GSSTUDIO_RUNTIME=str(RUNTIME),
                       GSSTUDIO_GPU_LOCK_PATH=str(configured_lock.resolve()))
    environment.pop("PYTHONHOME", None)
    # Bind REAL CUDA work to this runtime's shared lock, even if the parent's
    # shell inherited an old CPU-test override. CPU command tests substitute
    # both this runtime config and Popen; they never touch the production lock.
    return environment


def run_child(job, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Probe runtime budget exceeded before starting another child")
    attempt = Path(job["attempt"])
    attempt.mkdir(exist_ok=False)
    file = attempt / "job.json"
    write_json(file, job)
    command = [sys.executable, str(Path(__file__).resolve()), "--allow-cuda", "--_job", str(file)]
    write_json(attempt / "command.json", {"argv":command, "runtime":str(RUNTIME), "pythonpath":str(ENGINE)})
    signalled = False
    with (attempt / "child.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(ENGINE.parent), env=child_environment(),
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Probe runtime budget exceeded; terminating only its owned child")
                if not signalled and (attempt / "ready.json").is_file():
                    ready = read_json(attempt / "ready.json")
                    require(job["phase"] == "interrupt" and ready["case"] == job["case"] and ready["step"] == 5, "Unexpected child gate")
                    if job["case"] == "control_stop":
                        from local_postshot.studio_control import submit_control, read_control_state
                        state = read_control_state(job["experiment"])
                        request = submit_control(job["experiment"], state["package_sha256"], "stop")
                        require(request["generation_id"] == state["generation_id"], "Stop generation mismatch")
                        write_json(attempt / "external-control.json", request)
                        write_json(attempt / "release.json", {"pid":process.pid})
                    else:
                        write_json(attempt / "external-request.json", {"action":job["case"], "pid":process.pid})
                    signalled = True
                time.sleep(.02)
            result = {"pid":process.pid, "returncode":process.returncode, "external_signal_sent":signalled}
            write_json(attempt / "process.json", result)
            return result
        finally:
            if process.poll() is None:
                process.kill()  # Popen handle owned by this probe, never a PID search.
                process.wait(timeout=10)
                write_json(attempt / "timeout.json", {"owned_pid":process.pid, "returncode":process.returncode})


def interrupted_evidence(case, experiment, attempt, steps):
    from local_postshot.training import load_checkpoint
    experiment, attempt = Path(experiment), Path(attempt)
    fault = read_json(attempt / "fault.json")
    checkpoint = experiment / "checkpoints/latest.pt"
    summary = read_json(experiment / "training.json")
    control = read_json(experiment / "control-state.json")
    require(summary["status"] != "succeeded" and control["status"] != "succeeded", "Failure recorded as succeeded")
    require(summary["completed_steps"] < steps, "Interrupted run claims target complete")
    for name in ("training.json", "project.json", "control-state.json", "failure.json"):
        if (experiment / name).exists():
            shutil.copyfile(experiment / name, attempt / ("interrupted-" + name))
    saved = load_checkpoint(checkpoint, map_location="cpu")
    audit = audit_optimizer_state(saved, require_cuda=False)
    if case in ("oom", "checkpoint_fsync", "checkpoint_replace", "abrupt_exit"):
        require(saved["step"] == 4 and digest(checkpoint) == fault["last_good_sha256"], "Last-good checkpoint overwritten by fault")
        require(checkpoint.read_bytes() == (attempt / "last-good-before.pt").read_bytes(), "Last-good bytes changed")
        if case != "abrupt_exit":
            require(summary["status"] == control["status"] == "failed", "Injected failure not durably failed")
            failure = read_json(experiment / "failure.json")
            require(failure["status"] == "failed" and failure["recoverable"] is True, "Missing recoverable failure record")
        if case.startswith("checkpoint_"):
            require(read_json(attempt / "syscall.json")["calls"] == 1, "Checkpoint syscall injection not exercised exactly once")
            require((checkpoint.with_name(checkpoint.name + ".tmp")).is_file(), "Failed temporary checkpoint evidence missing")
            shutil.copyfile(checkpoint.with_name(checkpoint.name + ".tmp"), attempt / "failed-checkpoint.tmp")
    else:
        expected = "paused" if case == "external_save_pause" else "stopped"
        require(summary["status"] == control["status"] == expected, "External control status differs")
        require(saved["step"] == summary["completed_steps"] == 5, "External stop/pause did not save at its boundary")
        require(summary["evaluation_status"] == "not_run", "Interrupted run evaluated as completed")
    require(summary["checkpoint_sha256"] == digest(checkpoint), "Interrupted summary not bound to last durable checkpoint")
    require(summary["package_sha256"] == saved["studio_experiment"]["package_sha256"], "Interrupted package identity differs")
    restore_input = attempt / "restore-input.pt"
    shutil.copyfile(checkpoint, restore_input)
    require(digest(restore_input) == digest(checkpoint), "Restore evidence copy differs")
    evidence = {"observed_status":summary["status"], "control_status":control["status"],
                "abrupt_status_is_unfinalized":case == "abrupt_exit", "checkpoint_sha256":digest(checkpoint),
                "optimizer_audit":audit, "generation_id":summary["generation_id"],
                "fault":fault, "restore_input":str(restore_input)}
    write_json(attempt / "interruption.json", evidence)
    return evidence


def probe(root, steps, timeout):
    from studio_synthetic import make_package
    package = make_package(root / "package", size=48)
    other_package = make_package(root / "other-package", size=50)
    inputs = {"package":inventory(package), "other-package":inventory(other_package)}
    write_json(root / "input-inventory.json", inputs)
    report = {"schema_version":1, "status":"running", "scope":"real CUDA synthetic injected failures and fresh-process recovery",
              "deliberate_resource_exhaustion":False, "recipe":"quality_v2", "steps":steps, "timeout_seconds":timeout,
              "runtime":str(RUNTIME), "cases":[], "limits":[
                  "Injected OOM/IO exceptions, not actual allocator or disk exhaustion",
                  "Owned-child os._exit after a completed checkpoint, not a mid-kernel crash or machine power loss",
                  "Synthetic 48px images / at most 64 Gaussians; no real-scene quality or speed acceptance",
                  "External training cancel/save-pause and identity-bound stop; desktop/pipeline routing is out of scope",
                  "Abrupt exit preserves unfinalized production status verbatim; never relabelled succeeded/failed by this probe"]}
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    try:
        for case in CASES:
            base = root / case
            base.mkdir()
            experiment, interruption = base / "experiment", base / "interrupt"
            job = {"case":case, "phase":"interrupt", "attempt":str(interruption),
                   "experiment":str(experiment), "package":str(package), "steps":steps,
                   "owned_abrupt_exit_fixture":case == "abrupt_exit"}
            process = run_child(job, deadline)
            expected_exit = ABRUPT_EXIT if case == "abrupt_exit" else EXPECTED_FAULT if case in ("oom", "checkpoint_fsync", "checkpoint_replace") else 0
            require(process["returncode"] == expected_exit, f"{case}: child exited {process['returncode']}; inspect {interruption / 'child.log'}")
            partial = interrupted_evidence(case, experiment, interruption, steps)
            case_report = {"case":case, "status":"interrupted_verified", "interrupt_process":process, "interruption":partial}
            report["cases"].append(case_report)
            write_json(root / "validation.json", report)
            if case == "oom":
                report["identity_rejections"] = reject_identities(package, other_package, experiment, root / "identity-rejections", steps)
            resume_job = {**job, "phase":"resume", "attempt":str(base / "resume"), "restore_input":partial["restore_input"]}
            resumed = run_child(resume_job, deadline)
            require(resumed["returncode"] == 0, f"{case}: recovery exited {resumed['returncode']}; inspect {base / 'resume/child.log'}")
            require(resumed["pid"] != process["pid"], "Recovery must run in a fresh process")
            result = read_json(base / "resume/result.json")
            require(result["status"] == "recovered" and result["restore"]["all_state_bit_exact"], "Recovery proof missing")
            require(result["result"]["summary"]["generation_id"] != partial["generation_id"], "Recovery reused failed control generation")
            if (interruption / "interrupted-failure.json").is_file():
                require(digest(experiment / "failure.json") == digest(interruption / "interrupted-failure.json"),
                        "Recovery rewrote the prior durable failure record")
            case_report.update(status="passed", resume_process=resumed, recovered=result)
            require(inputs == {"package":inventory(package), "other-package":inventory(other_package)}, "Immutable inputs changed")
            write_json(root / "validation.json", report)
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc(),
                      action="Retain this entire directory; report production defects to coordination/failure-requests.md; do not weaken assertions")
        raise
    finally:
        report["input_unchanged"] = inputs == {"package":inventory(package), "other-package":inventory(other_package)}
        if not report["input_unchanged"]:
            report["status"] = "failed"
        report["wall_seconds"] = time.monotonic() - started
        write_json(root / "validation.json", report)
    require(report["input_unchanged"], "Immutable inputs changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda", action="store_true", help="Parent explicitly grants a GPU slot")
    parser.add_argument("--output", type=Path, help="New evidence directory; existing paths are rejected")
    parser.add_argument("--steps", type=int, default=12, help="Unchanged target for each case (12..40; default 12)")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="Total child runtime budget (30..300)")
    parser.add_argument("--_job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # This gate precedes all torch/numpy/core imports and all filesystem writes.
    if not args.allow_cuda:
        parser.error("Parent GPU authorization required: --allow-cuda")
    if not 12 <= args.steps <= 40 or not 30 <= args.timeout_seconds <= 300:
        parser.error("Require 12..40 steps and 30..300 timeout seconds")
    if args._job is None and args.output is None:
        parser.error("--output NEWDIR is required")
    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        if args._job is not None:
            return child_main(read_json(args._job))
        root = args.output.resolve()
        if root == ORIGINAL_DATA.resolve() or ORIGINAL_DATA.resolve() in root.parents:
            raise ValueError("Original Data is protected; choose a new evidence directory elsewhere")
        root.mkdir(parents=True, exist_ok=False)
        report = probe(root, args.steps, args.timeout_seconds)
        print(json.dumps({"status":report["status"], "report":str(root / "validation.json")}))
        return 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    code = main()
    print("EXIT=" + str(code), flush=True)
    sys.exit(code)

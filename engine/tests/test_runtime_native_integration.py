"""Runtime integration contracts without importing Torch or CUDA extensions."""
import hashlib
import importlib.metadata
import importlib.util
import json
import runpy
import sys
import sysconfig
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_postshot import runtime, worker

PRIVATE_BACKENDS = {
    "local_splat3_render": "native_renderer_v1",
    "local_splat3_tiles": "tile_support_v1",
    "local_splat3_projection": "native_floor_v1",
}


@pytest.fixture
def isolated_health(monkeypatch, tmp_path):
    """All dependency probes are fakes, including the direct ``import torch``."""
    imports = []
    binaries = {}
    for module in PRIVATE_BACKENDS:
        binary = tmp_path / (module + ".cp310-win_amd64.pyd")
        binary.write_bytes(("cpu-only-test-" + module).encode())
        binaries[module] = binary
    fake_torch = SimpleNamespace(
        version=SimpleNamespace(cuda="12.9"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda: "Mock GPU",
            get_device_properties=lambda device: SimpleNamespace(total_memory=16 * 1048576),
            init=lambda: pytest.fail("health must not initialize a CUDA context"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime, "config", lambda: {"data_root": str(tmp_path)})
    monkeypatch.setattr(runtime, "executable", lambda name: str(tmp_path / (name + ".exe")))

    def import_module(name):
        imports.append(name)
        return fake_torch if name == "torch" else SimpleNamespace(implementation_version=PRIVATE_BACKENDS.get(name))

    def distribution(name):
        module = name.replace("-", "_")
        assert module in PRIVATE_BACKENDS
        binary = binaries[module]
        return SimpleNamespace(version="0.1.0", files=[Path(binary.name)],
                               locate_file=lambda file: tmp_path / file)

    monkeypatch.setattr(runtime.importlib, "import_module", import_module)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "0.1.0")
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", distribution)
    return SimpleNamespace(imports=imports, binaries=binaries,
                           binary=binaries["local_splat3_render"], importer=import_module)


def test_optional_old_backends_absent_do_not_break_base_runtime(monkeypatch, isolated_health):
    def importer(name):
        if name in {"fused_ssim", "local_splat3_stats"}:
            raise ImportError("optional dependency absent")
        return isolated_health.importer(name)

    monkeypatch.setattr(runtime.importlib, "import_module", importer)
    result = runtime.health()
    assert result["ready"] is True
    assert result["missing"] == []
    for name in ("fused_ssim", "local_splat3_stats"):
        assert result["optional_backends"][name]["ready"] is False
        assert "optional dependency absent" in result["optional_backends"][name]["error"]


@pytest.mark.parametrize("module", PRIVATE_BACKENDS)
def test_renderer_health_reports_distribution_version_and_binary_digest(isolated_health, module):
    result = runtime.health()
    renderer = result["optional_backends"][module]
    assert result["ready"] is True
    assert renderer["ready"] is True
    assert renderer["distribution"] == module.replace("_", "-")
    assert renderer["version"] == "0.1.0"
    assert renderer["binary_sha256"] == hashlib.sha256(isolated_health.binaries[module].read_bytes()).hexdigest()
    assert renderer["implementation"] == PRIVATE_BACKENDS[module]
    assert module in isolated_health.imports


@pytest.mark.parametrize("error", [ImportError("renderer absent"), OSError("renderer DLL unavailable")])
@pytest.mark.parametrize("module", PRIVATE_BACKENDS)
def test_renderer_import_failure_is_optional(monkeypatch, isolated_health, error, module):
    def importer(name):
        if name == module:
            raise error
        return isolated_health.importer(name)

    monkeypatch.setattr(runtime.importlib, "import_module", importer)
    result = runtime.health()
    assert result["ready"] is True
    assert result["missing"] == []
    renderer = result["optional_backends"][module]
    assert renderer["ready"] is False
    assert str(error) in renderer["error"]


@pytest.mark.parametrize("error", [None, ImportError("test import failure"), OSError("test DLL failure")])
@pytest.mark.parametrize("module", PRIVATE_BACKENDS)
def test_optional_renderer_preloads_on_main_thread_without_protocol_output(monkeypatch, capsys, error, module):
    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: False,
        init=lambda: pytest.fail("CPU integration test attempted CUDA init"),
    ))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def importer(name):
        calls.append((name, threading.current_thread()))
        if name == module and error is not None:
            raise error
        return fake_torch if name == "torch" else SimpleNamespace()

    monkeypatch.setattr(worker.importlib, "import_module", importer)
    monkeypatch.setattr(worker.importlib.util, "find_spec", lambda name: object())
    worker.initialize_native_runtime()
    assert (module, threading.main_thread()) in calls
    assert all((name, threading.main_thread()) in calls for name in PRIVATE_BACKENDS)
    assert next(i for i, (name, _) in enumerate(calls) if name == "torch") < next(i for i, (name, _) in enumerate(calls) if name == module)
    assert all(thread is threading.main_thread() for _, thread in calls)
    captured = capsys.readouterr()
    assert captured.out == ""
    if error is not None:
        assert f"Native preflight {module}" in captured.err
        assert str(error) in captured.err


def test_renderer_record_merge_preserves_dataset_spz_unknown_fields_and_original_backups(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/record_splat3_renderer_runtime.py"
    spec = importlib.util.spec_from_file_location("record_splat3_renderer_runtime_cpu_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = tmp_path / "provisioning-manifest.json"
    lock_path = tmp_path / "runtime-requirements.lock.txt"
    initial = {
        "versions": {"spz": "1.1.0", "gsplat": "1.5.3", "local-splat3-stats": "0.1.1"},
        "artifacts": {"wheels/spz.whl": "original-spz-hash"},
        "dataset": {"train": 112, "holdout": 16, "custom_split": [1, 9, 17]},
        "spz": {"commit": "unchanged", "custom": {"build": 7}},
        "local_splat3_stats": {"version": "0.1.1", "binary_sha256": "old-stats"},
        "unknown_future_field": {"keep": ["exact", "values"]},
    }
    manifest_path.write_text(json.dumps(initial), encoding="utf-8-sig")
    lock_path.write_text("spz==1.1.0\ngsplat==1.5.3\nlocal_splat3_render==0.0.8\n"
                         "local-splat3-render==0.0.9\nlocal-splat3-stats==0.1.1\n", encoding="utf-8-sig")
    initial_manifest_bytes = manifest_path.read_bytes()
    initial_lock_bytes = lock_path.read_bytes()
    dataset = tmp_path / "datasets/south-building/manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"dataset bytes must not change")
    spz_wheel = tmp_path / "wheels/spz.whl"
    spz_wheel.parent.mkdir()
    spz_wheel.write_bytes(b"official SPZ unchanged")
    wheel = tmp_path / "wheels/local_splat3_render-0.1.0-test.whl"
    wheel.write_bytes(b"test renderer wheel")
    identity = {"distribution": "local-splat3-render", "version": "0.1.0",
                "binary_sha256": "renderer-binary", "source": "engine/native/splat3_render"}

    result = module.merge_runtime_record(tmp_path, identity, wheel)
    saved = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert result == saved
    for name in ("dataset", "spz", "local_splat3_stats", "unknown_future_field"):
        assert saved[name] == initial[name]
    assert saved["local_splat3_render"] == identity
    assert saved["versions"]["local-splat3-render"] == "0.1.0"
    assert saved["versions"]["spz"] == "1.1.0"
    assert saved["versions"]["local-splat3-stats"] == "0.1.1"
    assert saved["artifacts"]["wheels/spz.whl"] == "original-spz-hash"
    assert saved["artifacts"][wheel.relative_to(tmp_path).as_posix()] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock = lock_path.read_text(encoding="utf-8-sig").splitlines()
    assert [line for line in lock if line.lower().startswith(("local-splat3-render==", "local_splat3_render=="))] == ["local-splat3-render==0.1.0"]
    assert "spz==1.1.0" in lock and "local-splat3-stats==0.1.1" in lock
    assert dataset.read_bytes() == b"dataset bytes must not change"
    assert spz_wheel.read_bytes() == b"official SPZ unchanged"

    # Backup names are an implementation detail; their original bytes are not.
    backups = {path: path.read_bytes() for path in tmp_path.iterdir()
               if path.is_file() and path not in (manifest_path, lock_path)}
    assert initial_manifest_bytes in backups.values()
    assert initial_lock_bytes in backups.values()
    module.merge_runtime_record(tmp_path, {**identity, "version": "0.1.1"}, wheel)
    assert all(path.read_bytes() == content for path, content in backups.items())
    saved_again = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assert saved_again["local_splat3_render"]["version"] == "0.1.1"
    assert saved_again["dataset"] == initial["dataset"]


@pytest.mark.parametrize("mutation", ["wheel", "binary", "missing_receipt", None])
def test_renderer_receipt_only_attaches_tests_to_matching_artifact(monkeypatch, tmp_path, mutation):
    script = Path(__file__).resolve().parents[1] / "scripts/record_splat3_renderer_runtime.py"
    spec = importlib.util.spec_from_file_location("renderer_receipt_cpu_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private_prefix = tmp_path / "runtime/python"
    binary = private_prefix / "Lib/site-packages/local_splat3_render.pyd"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"validated binary")
    wheel = tmp_path / "wheels/local_splat3_render-0.1.0-cp310-cp310-win_amd64.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"validated wheel")
    receipt = tmp_path / "build.json"
    receipt.write_text(json.dumps({
        "identity": {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "staging_cpu_verification": {"wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()},
        "validation": {"gpu_expanded": "13passed"},
        "known_substitutes": ["Public projection retained"],
    }), encoding="utf-8")
    source = tmp_path / "repo/engine/native/splat3_render"
    source.mkdir(parents=True)
    (source / "renderer.cu").write_bytes(b"independent source")
    monkeypatch.setattr(module, "REPO", tmp_path / "repo")
    monkeypatch.setattr(sys, "prefix", str(private_prefix))
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.1.0", files=[Path(binary.name)], locate_file=lambda file: binary))
    if mutation == "wheel":
        wheel.write_bytes(b"rebuilt wheel")
    elif mutation == "binary":
        binary.write_bytes(b"rebuilt binary")
    elif mutation == "missing_receipt":
        receipt.unlink()

    if mutation is not None:
        with pytest.raises(RuntimeError, match="does not match validated build receipt"):
            module.renderer_runtime_record(tmp_path, receipt_path=receipt)
    record, actual_wheel = module.renderer_runtime_record(tmp_path, receipt_path=receipt, strict_receipt=False)
    assert actual_wheel == wheel
    assert record["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert record["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    validation = record["artifact_validation"]
    assert validation["matches_validated_build"] is (mutation is None)
    assert validation["tests"] == ({"gpu_expanded": "13passed"} if mutation is None else None)
    assert not (tmp_path / "provisioning-manifest.json").exists()


@pytest.mark.parametrize("outside", ["prefix", "binary"])
def test_renderer_recorder_rejects_nonprivate_locations_before_writing(monkeypatch, tmp_path, outside):
    script = Path(__file__).resolve().parents[1] / "scripts/record_splat3_renderer_runtime.py"
    spec = importlib.util.spec_from_file_location("renderer_private_prefix_cpu_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private_prefix = tmp_path / "runtime/python"
    binary = (tmp_path / "global-python" if outside == "binary" else private_prefix) / "renderer.pyd"
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "global-python" if outside == "prefix" else private_prefix))
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.1.0", files=[Path(binary.name)], locate_file=lambda file: binary))
    with pytest.raises(ValueError):
        module.record_renderer_runtime(tmp_path, strict_receipt=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("partial_metadata", [False, True])
def test_base_runtime_record_builds_spz_once_and_preserves_existing_dataset_and_wheel(monkeypatch, tmp_path, partial_metadata):
    """Execute only the metadata recorder against temporary files and fake deps."""
    script = Path(__file__).resolve().parents[1] / "scripts/setup_record_runtime.py"
    site = tmp_path / "runtime/python/Lib/site-packages"
    spz_package = site / "spz"
    spz_package.mkdir(parents=True)
    (spz_package / "__init__.py").write_bytes(b"# official binding stub\n")
    (spz_package / "_spz.pyd").write_bytes(b"not a real DLL")
    info = site / "spz-1.1.0.dist-info"
    metadata_bytes = b"Metadata-Version: 2.1\nName: spz\nVersion: 1.1.0\nSummary: existing metadata\n"
    if partial_metadata:
        info.mkdir()
        (info / "METADATA").write_bytes(metadata_bytes)
        (info / "RECORD").write_bytes(b"stale partial record")
    required = ["sources/spz/LICENSE", "downloads/colmap-x64-windows-cuda.zip", "downloads/south-building.zip",
                "downloads/zlib-1.3.2.tar.gz", "wheels/gsplat-1.5.3-cp310-cp310-win_amd64.whl", "tools/ffmpeg/ffmpeg.exe"]
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    dataset = tmp_path / "datasets/south-building/manifest.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(json.dumps({"images": [{"name": "custom.JPG", "split": "holdout"}],
                                   "split_policy": "existing custom split"}), encoding="utf-8-sig")
    dataset_bytes = dataset.read_bytes()
    manifest = tmp_path / "provisioning-manifest.json"
    manifest.write_text(json.dumps({"unknown": {"keep": True}, "spz": {"custom": "retained"}}), encoding="utf-8-sig")
    monkeypatch.setenv("LOCAL_POSTSHOT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "runtime/python"))
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site)})
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name": "spz"}, version="1.1.0")])
    monkeypatch.setitem(sys.modules, "record_splat3_renderer_runtime", SimpleNamespace(record_renderer_runtime=lambda *a, **k: {}))
    # This test owns only SPZ/base provenance behavior. Round4 native recording
    # is exercised with the real helper and fake distributions in its own file.
    round4_calls = []
    monkeypatch.setitem(sys.modules, "record_splat3_round4_runtime", SimpleNamespace(
        record_round4_native_runtime=lambda root: round4_calls.append(Path(root)) or {}))
    runpy.run_path(str(script))
    wheel = tmp_path / "wheels/spz-1.1.0-cp310-cp310-win_amd64.whl"
    assert wheel.is_file()
    assert all((info / name).is_file() for name in ("METADATA", "WHEEL", "LICENSE", "SOURCE.json", "RECORD"))
    if partial_metadata:
        assert (info / "METADATA").read_bytes() == metadata_bytes
        record_text = (info / "RECORD").read_text(encoding="utf-8")
        assert "spz-1.1.0.dist-info/LICENSE" in record_text
        assert "spz-1.1.0.dist-info/SOURCE.json" in record_text
        assert "stale partial record" not in record_text
    preserved = {p: p.read_bytes() for p in info.iterdir()}
    preserved[wheel] = wheel.read_bytes()
    runpy.run_path(str(script))
    assert all(p.read_bytes() == content for p, content in preserved.items())
    assert dataset.read_bytes() == dataset_bytes
    saved = json.loads(manifest.read_text(encoding="utf-8-sig"))
    assert saved["unknown"] == {"keep": True}
    assert saved["spz"] == {"custom": "retained"}
    assert saved["dataset"] == {"count": 1, "train": 0, "holdout": 1}
    assert round4_calls == [tmp_path.resolve(), tmp_path.resolve()]


@pytest.mark.parametrize("outside", ["prefix", "site"])
def test_base_runtime_recorder_rejects_global_python_before_any_write(monkeypatch, tmp_path, outside):
    script = Path(__file__).resolve().parents[1] / "scripts/setup_record_runtime.py"
    root = tmp_path / "private"
    global_prefix = tmp_path / "global-python"
    private_prefix = root / "runtime/python"
    prefix = global_prefix if outside == "prefix" else private_prefix
    site = (global_prefix if outside == "site" else private_prefix) / "Lib/site-packages"
    monkeypatch.setenv("LOCAL_POSTSHOT_ROOT", str(root))
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sysconfig, "get_paths", lambda: {"purelib": str(site)})
    monkeypatch.setitem(sys.modules, "record_splat3_renderer_runtime", SimpleNamespace(
        record_renderer_runtime=lambda *a, **k: pytest.fail("late renderer guard was reached")))
    monkeypatch.setitem(sys.modules, "record_splat3_round4_runtime", SimpleNamespace(
        record_round4_native_runtime=lambda *a, **k: pytest.fail("late round4 guard was reached")))
    with pytest.raises(ValueError):
        runpy.run_path(str(script))
    assert list(tmp_path.iterdir()) == []

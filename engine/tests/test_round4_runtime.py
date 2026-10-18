"""CPU-only optional backends and real incremental recorder contracts.

Native DLL imports are never executed. The recorder is real production code;
only installed-distribution metadata and the private filesystem root are faked.
"""
import builtins
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import pytest

from local_postshot import runtime, worker
from test_runtime_native_integration import isolated_health


@pytest.mark.parametrize("absent", ["local_splat3_tiles", "local_splat3_projection"])
def test_new_optional_preflight_absence_is_silent_and_does_not_import(monkeypatch, capsys, absent):
    calls=[]
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False,
        init=lambda:pytest.fail("CPU fixture attempted CUDA initialization")))
    monkeypatch.setitem(sys.modules,"torch",fake_torch)
    def importer(name):
        calls.append((name,threading.current_thread()))
        assert name!=absent
        return fake_torch if name=="torch" else SimpleNamespace()
    monkeypatch.setattr(worker.importlib,"import_module",importer)
    monkeypatch.setattr(worker.importlib.util,"find_spec",lambda name:None if name==absent else object())
    worker.initialize_native_runtime()
    other="local_splat3_projection" if absent=="local_splat3_tiles" else "local_splat3_tiles"
    assert (other,threading.main_thread()) in calls
    assert all(thread is threading.main_thread() for _,thread in calls)
    captured=capsys.readouterr()
    assert captured.out=="" and captured.err==""


@pytest.mark.parametrize("module", ["local_splat3_tiles", "local_splat3_projection"])
@pytest.mark.parametrize("metadata", ["missing_distribution", "zero_binaries", "two_binaries"])
def test_new_backend_bad_metadata_is_optional_and_never_reports_ready(monkeypatch, isolated_health, module, metadata):
    original=runtime.importlib.metadata.distribution
    def distribution(name):
        if name!=module.replace("_","-"):return original(name)
        if metadata=="missing_distribution":raise importlib.metadata.PackageNotFoundError(name)
        files=[] if metadata=="zero_binaries" else [Path("a.pyd"),Path("b.pyd")]
        return SimpleNamespace(version="0.1.0",files=files,locate_file=lambda path:isolated_health.binary.parent/path)
    monkeypatch.setattr(runtime.importlib.metadata,"distribution",distribution)
    result=runtime.health()
    assert result["ready"] is True and result["missing"]==[]
    assert result["optional_backends"][module]["ready"] is False
    assert result["optional_backends"][module]["error"]


@pytest.fixture
def round4_record_fixture(monkeypatch,tmp_path):
    script=Path(__file__).resolve().parents[1]/"scripts/record_splat3_round4_runtime.py"
    spec=importlib.util.spec_from_file_location("record_splat3_round4_runtime_cpu_test",script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/"private";prefix=root/"runtime/python"
    repo=tmp_path/"repo";(repo/"evidence/splat3").mkdir(parents=True)
    initial={"versions":{"spz":"1.1.0","gsplat":"1.5.3","local-splat3-stats":"0.1.1","local-splat3-render":"0.1.0",
                         "local-splat3-tiles":"0.0.8"},
        "artifacts":{"wheels/spz.whl":"existing-spz", "wheels/old-render.whl":"existing-render"},
        "dataset":{"images":[{"name":"fixed.png","split":"holdout"}],"train":112,"holdout":16},
        "spz":{"custom":["preserve","exact"]},"local_splat3_stats":{"binary_sha256":"old-stats"},
        "local_splat3_render":{"binary_sha256":"old-render"},"unknown":{"nested":[1,2,3]}}
    root.mkdir();manifest=root/"provisioning-manifest.json";lock=root/"runtime-requirements.lock.txt"
    manifest.write_text(json.dumps(initial),encoding="utf-8-sig")
    unrelated_lines=["# Preserve human pin comment","spz==1.1.0","gsplat==1.5.3","local-splat3-stats==0.1.1",
                     "local-splat3-render==0.1.0","some-extra-package==7.2"]
    lock.write_text("\n".join(unrelated_lines+["local_splat3_tiles==0.0.7","local-splat3-tiles==0.0.8",
                                            "local_splat3_projection==0.0.4"])+"\n",encoding="utf-8-sig")
    binaries={};wheels={}
    for suffix in ("tiles","projection"):
        name="local_splat3_"+suffix
        binary=prefix/"Lib/site-packages"/(name+".pyd");binary.parent.mkdir(parents=True,exist_ok=True)
        binary.write_bytes(("fake binary "+suffix).encode());binaries[suffix]=binary
        wheel=root/"wheels"/(name+"-0.1.0-cp310-cp310-win_amd64.whl");wheel.parent.mkdir(exist_ok=True)
        wheel.write_bytes(("fake wheel "+suffix).encode());wheels[suffix]=wheel
        src=repo/"engine/native"/("splat3_"+suffix);src.mkdir(parents=True)
        (src/"kernel.cu").write_bytes(b"independently written test source")
        (src/"LICENSE").write_bytes(b"test license")
    protected={}
    for relative,content in [("datasets/south-building/manifest.json",b"exact dataset bytes"),
        ("wheels/spz.whl",b"exact old SPZ wheel"),("wheels/old-render.whl",b"exact renderer wheel"),
        ("runtime/python/Lib/site-packages/local_splat3_stats.pyd",b"exact old stats binary")]:
        path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content);protected[path]=content
    for path in (*binaries.values(),*wheels.values()):protected[path]=path.read_bytes()
    lookups=[]
    def distribution(name):
        lookups.append(name)
        assert name in ("local-splat3-tiles","local-splat3-projection")
        binary=binaries[name.rsplit("-",1)[-1]]
        return SimpleNamespace(version="0.1.0",files=[Path(binary.name)],locate_file=lambda path:binary)
    monkeypatch.setattr(module,"REPO",repo);monkeypatch.setattr(sys,"prefix",str(prefix))
    monkeypatch.setattr(module.importlib.metadata,"distribution",distribution)
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name.split(".")[0] in {"torch","local_splat3_tiles","local_splat3_projection"}:
            pytest.fail("Metadata recorder attempted native/Torch import: "+name)
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",guarded_import)
    return SimpleNamespace(module=module,root=root,prefix=prefix,repo=repo,manifest=manifest,lock=lock,
        initial=initial,unrelated_lines=unrelated_lines,protected=protected,binaries=binaries,wheels=wheels,lookups=lookups)


def test_real_round4_recorder_merges_only_new_modules_and_preserves_old_artifacts(round4_record_fixture):
    f=round4_record_fixture
    original_manifest=f.manifest.read_bytes();original_lock=f.lock.read_bytes()
    records=f.module.record_round4_native_runtime(f.root)
    assert f.lookups==["local-splat3-tiles","local-splat3-projection"]
    saved=json.loads(f.manifest.read_text(encoding="utf-8"))
    assert set(records)=={"local_splat3_tiles","local_splat3_projection"}
    for key in ("dataset","spz","local_splat3_stats","local_splat3_render","unknown"):
        assert saved[key]==f.initial[key]
    for key,value in f.initial["versions"].items():
        if key!="local-splat3-tiles":assert saved["versions"][key]==value
    for key,value in f.initial["artifacts"].items():assert saved["artifacts"][key]==value
    for suffix in ("tiles","projection"):
        name="local_splat3_"+suffix;record=records[name]
        assert saved[name]==record and record["version"]=="0.1.0"
        assert record["binary_sha256"]==hashlib.sha256(f.binaries[suffix].read_bytes()).hexdigest()
        assert record["wheel_sha256"]==hashlib.sha256(f.wheels[suffix].read_bytes()).hexdigest()
        assert record["original_postshot_code"] is False
        assert record["source_sha256"]
        # Fresh arbitrary bytes are recorded as identity, never as GPU-tested.
        assert "test" not in record and "tests" not in record
        assert "identity" in record["validation_scope"]
    lines=f.lock.read_text(encoding="utf-8").splitlines()
    assert all(line in lines for line in f.unrelated_lines)
    assert [line for line in lines if line.lower().replace("_","-").startswith("local-splat3-tiles==")]==["local-splat3-tiles==0.1.0"]
    assert [line for line in lines if line.lower().replace("_","-").startswith("local-splat3-projection==")]==["local-splat3-projection==0.1.0"]
    assert all(path.read_bytes()==content for path,content in f.protected.items())
    receipt=json.loads((f.repo/"evidence/splat3/round4-runtime.json").read_text())
    backup=Path(receipt["backup"])
    assert (backup/f.manifest.name).read_bytes()==original_manifest
    assert (backup/f.lock.name).read_bytes()==original_lock
    saved_bytes=f.manifest.read_bytes();lock_bytes=f.lock.read_bytes()
    # Repeat is idempotent for live metadata and never overwrites first backup.
    assert f.module.record_round4_native_runtime(f.root)==records
    assert f.manifest.read_bytes()==saved_bytes and f.lock.read_bytes()==lock_bytes
    assert (backup/f.manifest.name).read_bytes()==original_manifest
    assert (backup/f.lock.name).read_bytes()==original_lock
    assert all(path.read_bytes()==content for path,content in f.protected.items())


@pytest.mark.parametrize("failure", ["missing_distribution","missing_wheel","outside_binary","two_binaries"])
def test_real_recorder_rejects_incomplete_or_nonprivate_second_module_before_changing_pins(monkeypatch,round4_record_fixture,failure):
    f=round4_record_fixture;manifest=f.manifest.read_bytes();lock=f.lock.read_bytes()
    original=f.module.importlib.metadata.distribution
    if failure=="missing_wheel":f.wheels["projection"].unlink()
    else:
        def distribution(name):
            if name!="local-splat3-projection":return original(name)
            if failure=="missing_distribution":raise importlib.metadata.PackageNotFoundError(name)
            if failure=="two_binaries":return SimpleNamespace(files=[Path("one.pyd"),Path("two.pyd")],locate_file=lambda p:f.prefix/p)
            return SimpleNamespace(files=[Path("global.pyd")],locate_file=lambda p:f.root.parent/"global-python"/p)
        monkeypatch.setattr(f.module.importlib.metadata,"distribution",distribution)
    expected={"missing_distribution":importlib.metadata.PackageNotFoundError,"missing_wheel":FileNotFoundError,
              "outside_binary":ValueError,"two_binaries":RuntimeError}[failure]
    with pytest.raises(expected):f.module.record_round4_native_runtime(f.root)
    assert f.manifest.read_bytes()==manifest and f.lock.read_bytes()==lock
    assert not (f.repo/"evidence/splat3/round4-runtime.json").exists()
    for path,content in f.protected.items():
        if failure=="missing_wheel" and path==f.wheels["projection"]:continue
        assert path.read_bytes()==content


def test_real_recorder_rejects_global_python_before_even_backup_creation(monkeypatch,round4_record_fixture):
    f=round4_record_fixture;before={str(p.relative_to(f.root)):p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(sys,"prefix",str(f.root.parent/"global-python"))
    with pytest.raises(ValueError):f.module.record_round4_native_runtime(f.root)
    after={str(p.relative_to(f.root)):p.read_bytes() for p in f.root.rglob("*") if p.is_file()}
    assert before==after
    assert not (f.root/"evidence/runtime").exists()
    assert f.lookups==[]

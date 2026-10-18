"""Portable runtime contracts and release import boundaries; CPU only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gsstudio.infrastructure.runtime import configuration as pipeline
from local_postshot import runtime as engine

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def portable(tmp_path, monkeypatch):
    for name in set(pipeline.ENV_PATHS.values()) | {'GSSTUDIO_RUNTIME', 'LOCAL_POSTSHOT_RUNTIME',
            'GSSTUDIO_RUNTIME_OVERRIDE', 'GSSTUDIO_SETTINGS_DIR', 'CUDA_HOME'}:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    monkeypatch.setenv('GSSTUDIO_SETTINGS_DIR', str(tmp_path / 'settings'))
    runtime = tmp_path / 'relocated app/runtime/runtime.json'
    runtime.parent.mkdir(parents=True)
    fields = dict(release=False, user_data_root='${user_data_root}',
        python='python/python.exe', preprocess_python='python/python.exe',
        engine_root='../engine', pipeline_root='../pipeline', torch_home='../models/torch',
        data_root='${user_data_root}/projects', cache_root='${user_data_root}/cache',
        log_root='${user_data_root}/logs', gpu_lock_path='${user_data_root}/cache/gpu.lock',
        media_helper=None, mediasdk_root=None, realityscan=None,
        torch_extensions_dir='../engine/extensions')
    runtime.write_text(json.dumps(fields), encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME', str(runtime))
    return runtime


def test_portable_parity_same_interpreter_and_single_lock(portable, tmp_path, monkeypatch):
    cfg = pipeline.runtime_config()
    assert cfg == engine.config()
    assert cfg['python'] == cfg['preprocess_python'] == str(portable.parent / 'python/python.exe')
    assert cfg['torch_home'] == str(portable.parents[1] / 'models/torch')
    base = tmp_path / 'local/GSStudio'
    for key, relative in (('data_root','projects'), ('cache_root','cache'),
                          ('log_root','logs'), ('gpu_lock_path','cache/gpu.lock')):
        assert cfg[key] == str(base / relative)
    lock = cfg['gpu_lock_path']
    monkeypatch.setenv('GSSTUDIO_DATA_ROOT', str(tmp_path / 'external-projects'))
    assert engine.config()['gpu_lock_path'] == pipeline.runtime_config()['gpu_lock_path'] == lock
    assert engine.config()['data_root'] == str(tmp_path / 'external-projects')
    assert not base.exists(), 'Reading configuration must not create mutable state'


def test_external_override_saved_settings_and_env_precedence(portable, tmp_path, monkeypatch):
    override = tmp_path / 'external/runtime.override.json'
    override.parent.mkdir()
    override.write_text(json.dumps(dict(media_helper='sdk/bin/helper.exe', mediasdk_root='sdk',
        realityscan='rs/RealityScan.exe', data_root='projects', torch_home='models')), encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME_OVERRIDE', str(override))
    cfg = pipeline.runtime_config()
    assert cfg == engine.config()
    assert cfg['media_helper'] == str(override.parent / 'sdk/bin/helper.exe')
    assert cfg['data_root'] == str(override.parent / 'projects')
    settings = tmp_path / 'settings/settings.json'
    settings.parent.mkdir()
    settings.write_text(json.dumps(dict(schema_version=1, data_root=str(tmp_path / 'saved'))))
    assert pipeline.runtime_config()['data_root'] == engine.config()['data_root'] == str(tmp_path / 'saved')
    for key, name in pipeline.ENV_PATHS.items():
        monkeypatch.setenv(name, str(tmp_path / 'env' / key))
    for cfg in (pipeline.runtime_config(), engine.config()):
        for key in pipeline.ENV_PATHS:
            assert cfg[key] == str(tmp_path / 'env' / key)
        from local_postshot.studio_bridge import child_environment
        for env in (pipeline.child_environment(cfg), child_environment(cfg)):
            for key, name in pipeline.ENV_PATHS.items():
                assert env[name] == str(tmp_path / 'env' / key)


def test_default_override_and_no_machine_fallback(portable, tmp_path, monkeypatch):
    default_override = tmp_path / 'local/GSStudio/runtime.override.json'
    default_override.parent.mkdir(parents=True)
    default_override.write_text(json.dumps(dict(realityscan='installed/RealityScan.exe')))
    assert engine.config()['realityscan'] == str(default_override.parent / 'installed/RealityScan.exe')
    monkeypatch.delenv('GSSTUDIO_RUNTIME')
    monkeypatch.setattr(pipeline, '_adjacent_runtime', lambda: tmp_path / 'absent/runtime.json')
    monkeypatch.setattr(engine, '_adjacent_runtime', lambda: tmp_path / 'absent/runtime.json')
    assert pipeline.runtime_config() == engine.config()
    assert pipeline.runtime_config()['data_root'] == str(tmp_path / 'local/GSStudio/projects')


def test_missing_explicit_config_and_relative_env_fail(portable, tmp_path, monkeypatch):
    monkeypatch.setenv('GSSTUDIO_RUNTIME_OVERRIDE', str(tmp_path / 'missing.json'))
    for resolver in (engine.config, pipeline.runtime_config):
        with pytest.raises(FileNotFoundError, match='override|runtime'):
            resolver()
    monkeypatch.delenv('GSSTUDIO_RUNTIME_OVERRIDE')
    monkeypatch.setenv('GSSTUDIO_DATA_ROOT', 'relative')
    for resolver in (engine.config, pipeline.runtime_config):
        with pytest.raises(ValueError, match='absolute'):
            resolver()


def test_commercial_dependency_never_uses_path(portable, tmp_path, monkeypatch):
    from gsstudio.infrastructure.adapters import realityscan
    fake = tmp_path / 'RealityScan.exe'; fake.touch()
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(RuntimeError, match='explicit'):
        realityscan.realityscan_executable()
    monkeypatch.setenv('GSSTUDIO_REALITYSCAN_CLI', str(fake))
    assert realityscan.realityscan_executable() == fake


def test_release_guard_blocks_missing_binary_before_backend_exec(portable, tmp_path):
    fields = json.loads(portable.read_text())
    fields.update(release=True, engine_root=str(REPO / 'engine'))
    portable.write_text(json.dumps(fields))
    fake = tmp_path / 'fake'
    for folder in ('gsplat/cuda', 'torch/utils'):
        directory = fake / folder
        directory.mkdir(parents=True)
        (directory / '__init__.py').touch()
        (directory.parent / '__init__.py').touch()
    # Present but invalid PE bytes exercise the load-failure guard without CUDA.
    from local_postshot.release_runtime import REQUIRED_BINARIES
    for name in REQUIRED_BINARIES:
        binary = fake / (name.replace('.', '/') + '.pyd')
        binary.parent.mkdir(parents=True, exist_ok=True)
        if '.' in name:
            (binary.parent / '__init__.py').touch()
        binary.write_bytes(b'synthetic invalid extension; never a GPU task')
    sentinel = tmp_path / 'jit-was-executed'
    (fake / 'gsplat/cuda/_backend.py').write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n")
    (fake / 'torch/utils/cpp_extension.py').write_text("load=load_inline=_jit_compile=lambda *a,**k:None\n")
    (fake / 'torch/hub.py').write_text("download_url_to_file=lambda *a,**k:None\n")
    script = """
import sys
sys.path[:0]=sys.argv[1:3]
import local_postshot
try:
    import gsplat.cuda._backend
except RuntimeError as exc:
    assert 'gsplat.csrc' in str(exc) and 'JIT' in str(exc),str(exc)
else:
    raise AssertionError('missing gsplat binary accepted')
from torch.utils import cpp_extension
from torch import hub
for func in (cpp_extension.load,cpp_extension.load_inline,cpp_extension._jit_compile,hub.download_url_to_file):
    try: func()
    except RuntimeError as exc: assert 'disabled' in str(exc)
    else: raise AssertionError('release JIT/download allowed')
print('PASS')
"""
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(REPO / 'engine'), str(fake)],
                          capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'PASS'
    assert not sentinel.exists()
    (fake / 'gsplat/csrc.pyd').unlink()
    proc = subprocess.run([sys.executable, '-I', '-B', '-c',
        'import sys;sys.path[:0]=sys.argv[1:3];import local_postshot',
        str(REPO / 'engine'), str(fake)], capture_output=True, text=True, timeout=20)
    assert proc.returncode != 0 and 'required precompiled binary gsplat.csrc is missing' in proc.stderr


def test_selection_release_ignores_mutable_extension_cache(portable, tmp_path, monkeypatch):
    import types
    from local_postshot import studio_selection
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(__version__='fake-torch'))
    monkeypatch.setattr('local_postshot.studio_bridge.config',
                        lambda: dict(release=True, torch_extensions_dir=str(tmp_path / 'mutable')))
    source, root = studio_selection.build_location()
    assert root.is_relative_to(source.parent / 'extensions')
    assert not root.is_relative_to(tmp_path)


def test_versions_keep_internal_components():
    import gsstudio
    import local_postshot
    assert gsstudio.__version__ == '0.3.0rc1'
    assert local_postshot.__version__ == '0.1.0'
    for filename in ('desktop/package.json', 'desktop/src-tauri/tauri.conf.json'):
        assert json.loads((REPO / filename).read_text())['version'] == '0.3.0-rc.1'


def test_user_data_root_override_retargets_inherited_tokens(portable, tmp_path, monkeypatch):
    override = tmp_path / 'override.json'
    override.write_text(json.dumps(dict(user_data_root='new-data')))
    monkeypatch.setenv('GSSTUDIO_RUNTIME_OVERRIDE', str(override))
    cfg = engine.config()
    assert cfg == pipeline.runtime_config()
    assert cfg['user_data_root'] == str(tmp_path / 'new-data')
    assert cfg['data_root'] == str(tmp_path / 'new-data/projects')
    assert cfg['gpu_lock_path'] == str(tmp_path / 'new-data/cache/gpu.lock')


@pytest.mark.parametrize('shell', ['pwsh', 'powershell'])
def test_package_cli_allows_optional_sdk_and_preserves_env(portable, tmp_path, shell):
    """Exercise the real PS7/PS5.1 wrapper, argv serialization and exit code."""
    import shutil
    host = shutil.which(shell)
    if not host:
        pytest.skip(f'{shell} is unavailable')
    package = tmp_path / "cli 中文 space"
    package.mkdir()
    shutil.copy2(REPO / 'desktop/scripts/gsstudio-cli.ps1', package / 'gsstudio.ps1')
    module = package / 'pipeline/gsstudio/infrastructure/runtime'
    module.mkdir(parents=True)
    for directory in (module.parent.parent, module.parent, module):
        (directory / '__init__.py').touch()
    shutil.copy2(REPO / 'src/gsstudio/infrastructure/runtime/configuration.py', module / 'configuration.py')
    (module.parent.parent / '__main__.py').write_text(
        "import json,os,sys\nprint(json.dumps(dict(argv=sys.argv[1:],data=os.environ['GSSTUDIO_DATA_ROOT'],"
        "lock=os.environ['GSSTUDIO_GPU_LOCK_PATH'],helper=os.environ.get('GSSTUDIO_MEDIA_HELPER'))))\nsys.exit(42)\n")
    fields = json.loads(portable.read_text())
    fields.update(python=sys.executable, preprocess_python=sys.executable,
                  engine_root=str(REPO / 'engine'), pipeline_root=str(package / 'pipeline'))
    portable.write_text(json.dumps(fields))
    wrapper = package / 'invoke.ps1'
    wrapper.write_text("""$Before = $env:GSSTUDIO_DATA_ROOT
. (Join-Path $PSScriptRoot 'gsstudio.ps1') '中文 space' '' 'quoted "word"' 'C:\\tail space\\'
if ($env:GSSTUDIO_DATA_ROOT -ne $Before) { throw 'Caller environment mutated' }
exit $LASTEXITCODE
""", encoding='utf-8-sig')
    env = os.environ.copy()
    env['GSSTUDIO_DATA_ROOT'] = str(tmp_path / 'explicit-projects')
    env['GSSTUDIO_GPU_LOCK_PATH'] = str(tmp_path / 'explicit.lock')
    proc = subprocess.run([host, '-NoProfile', '-File', str(wrapper)],
                          env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert proc.returncode == 42, proc.stderr
    data = json.loads(proc.stdout)
    assert data['argv'] == ['中文 space', '', 'quoted "word"', 'C:\\tail space\\']
    assert data['data'] == env['GSSTUDIO_DATA_ROOT']
    assert data['lock'] == env['GSSTUDIO_GPU_LOCK_PATH']
    assert data['helper'] is None



def test_release_alias_override_cannot_split_interpreters(portable):
    fields = json.loads(portable.read_text())
    fields.update(release=True, preprocess_python='different/python.exe')
    portable.write_text(json.dumps(fields))
    for resolver in (engine.config, pipeline.runtime_config):
        with pytest.raises(ValueError, match='same interpreter'):
            resolver()

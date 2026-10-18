"""CPU evidence for explicit-root private children; no writes to a live package."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from gsstudio.infrastructure.adapters import gsplat
from gsstudio.infrastructure.runtime.configuration import (
    child_environment, isolated_module_command, runtime_config,
)

REPO = Path(__file__).resolve().parents[2]
PACKAGE = Path('E:/AI/GSStudio/app')
VALIDATION_RUNTIME = Path('E:/AI/GSStudio/coordination/validation-runtime.json')


def private_config():
    selected = Path(os.environ.get('GSSTUDIO_RUNTIME') or VALIDATION_RUNTIME)
    cfg = json.loads(selected.read_text(encoding='utf-8-sig'))
    cfg['runtime_path'] = str(selected)
    for name in ('python', 'preprocess_python'):
        assert Path(cfg[name]).is_file(), f'Required private runtime missing: {name}'
    return cfg


def poisoned_environment(tmp_path):
    env = os.environ.copy()
    roots = [tmp_path / name for name in ('cwd', 'global-python', 'pythonpath', 'userbase')]
    for root in roots:
        root.mkdir()
        for name in ('isolation_poison', 'sitecustomize', 'usercustomize', 'json', 'argparse'):
            (root / f'{name}.py').write_text("raise RuntimeError('POISON IMPORT')\n", encoding='utf-8')
        for name in ('gsstudio', 'local_postshot'):
            (root / name).mkdir()
            (root / name / '__init__.py').write_text("raise RuntimeError('POISON IMPORT')\n", encoding='utf-8')
    user_site = roots[3] / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'site-packages'
    user_site.mkdir(parents=True)
    (user_site / 'isolation_poison.py').write_text("raise RuntimeError('POISON USER')\n")
    env.update(PYTHONPATH=os.pathsep.join(map(str, roots)), PYTHONHOME=str(roots[1]),
               PYTHONUSERBASE=str(roots[3]), PYTHONSTARTUP=str(roots[0] / 'sitecustomize.py'),
               PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return env, roots


@pytest.mark.parametrize('kind', ['python', 'preprocess_python'])
def test_real_private_child_isolated_argv_and_poison_rejected(tmp_path, kind, monkeypatch):
    cfg = private_config()
    env, poison = poisoned_environment(tmp_path)
    pipeline = tmp_path / 'pipeline 中文 space'; pipeline.mkdir()
    engine = tmp_path / 'engine'; engine.mkdir()
    probe = '''import importlib.util,json,os,sys
print(json.dumps(dict(isolated=sys.flags.isolated,no_user_site=sys.flags.no_user_site,
 ignore_environment=sys.flags.ignore_environment,argv=sys.argv[1:],path=sys.path,
 poison=importlib.util.find_spec('isolation_poison') is not None,
 torch_loaded='torch' in sys.modules,runtime=os.environ.get('GSSTUDIO_RUNTIME'))))
'''
    (pipeline / 'isolation_probe.py').write_text(probe, encoding='utf-8')
    args = ['中文 space', "O'Brien", '', 'double"quote', 'C:\\tail space\\']
    command_cfg = dict(cfg, python=cfg[kind], pipeline_root=str(pipeline), engine_root=str(engine))
    command = isolated_module_command(command_cfg, 'isolation_probe', *args)
    proc = subprocess.run(command, cwd=poison[0], env=env, capture_output=True,
                          text=True, encoding='utf-8', timeout=30)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data['isolated'] == data['no_user_site'] == data['ignore_environment'] == 1
    assert data['argv'] == args and not data['poison'] and not data['torch_loaded']
    assert not any(str(root).casefold() in str(path).casefold()
                   for root in poison for path in data['path'])


def test_real_bridge_child_flags_and_jsonl_identity(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / 'engine'))
    from local_postshot import studio_bridge as bridge
    cfg = private_config()
    pipeline = tmp_path / 'pipeline'; worker = pipeline / 'gsstudio/interfaces/worker'
    worker.mkdir(parents=True)
    for directory in (pipeline / 'gsstudio', worker.parent, worker):
        (directory / '__init__.py').touch()
    (worker / 'main.py').write_text('''import importlib.util,json,os,sys
r=json.loads(sys.stdin.readline())
print(json.dumps(dict(type='result',id=r['id'],result=dict(isolated=sys.flags.isolated,
 argv=sys.argv[1:],poison=importlib.util.find_spec('isolation_poison') is not None,
 torch_home=os.environ.get('TORCH_HOME'),lock=os.environ.get('GSSTUDIO_GPU_LOCK_PATH')))))
''', encoding='utf-8')
    cfg.update(pipeline_root=str(pipeline), data_root=str(tmp_path/'absent-projects'),
               gpu_lock_path=str(tmp_path/'unused.lock'))
    env, _ = poisoned_environment(tmp_path)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE'):
        monkeypatch.setenv(key, env[key])
    monkeypatch.setattr(bridge, 'config', lambda: cfg)
    result = bridge.call('studio.catalog', {}, request_id="中文 O'Brien")
    assert result['isolated'] == 1 and not result['poison'] and result['argv'] == []
    assert result['torch_home'] == cfg['torch_home']
    # Explicit source lock overrides remain supported; package CLI sets the configured lock.
    assert result['lock'] == os.environ['GSSTUDIO_GPU_LOCK_PATH']


def test_actual_core_to_preprocess_catalog_cpu_only(tmp_path):
    cfg = private_config()
    cfg.update(data_root=str(tmp_path/'absent-projects'), gpu_lock_path=str(tmp_path/'unused.lock'))
    runtime = tmp_path / 'runtime.json'; cfg['runtime_path'] = str(runtime)
    runtime.write_text(json.dumps(cfg), encoding='utf-8')
    env, poison = poisoned_environment(tmp_path)
    env['GSSTUDIO_RUNTIME'] = str(runtime)
    env.pop('GSSTUDIO_DATA_ROOT', None); env.pop('GSSTUDIO_GPU_LOCK_PATH', None)
    script = '''import json,sys
sys.path.insert(0,sys.argv[1]);del sys.argv[1:]
from local_postshot.studio_bridge import call
result=call('studio.catalog',{},request_id='cpu-catalog-中文')
print(json.dumps(dict(result=result,isolated=sys.flags.isolated,torch_loaded='torch' in sys.modules)))
'''
    proc = subprocess.run([cfg['python'], '-I', '-B', '-u', '-c', script, cfg['engine_root']],
                          env=env, cwd=poison[0], capture_output=True, text=True,
                          encoding='utf-8', timeout=60)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data['isolated'] == 1 and not data['torch_loaded']
    assert all(data['result'][key] == [] for key in ('locations','scenes','captures','runs','external_imports'))
    assert not Path(cfg['data_root']).exists() and not Path(cfg['gpu_lock_path']).exists()


def test_actual_core_trainer_help_uses_new_builder(tmp_path, monkeypatch):
    cfg = private_config()
    monkeypatch.setattr('gsstudio.infrastructure.runtime.configuration.runtime_config', lambda: cfg)
    env, poison = poisoned_environment(tmp_path)
    env = child_environment(cfg, core=True) | {key: env[key] for key in ('PYTHONHOME','PYTHONPATH','PYTHONUSERBASE')}
    command = gsplat.native_training_command('--help')
    assert command[1:5] == ['-I','-B','-u','-c']
    proc = subprocess.run(command, env=env, cwd=poison[0], capture_output=True,
                          text=True, encoding='utf-8', timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert '--dataset' in proc.stdout and '--resume' in proc.stdout
    diagnostic = gsplat.trainer_diagnostic_command()
    assert diagnostic[1:5] == ['-I','-B','-u','-c']
    assert diagnostic[7] == 'gsstudio.interfaces.trainer_diagnostic'
    # Diagnostic execution is deliberately excluded: its CLI performs GPU work.


@pytest.mark.parametrize('key', ['python', 'engine_root', 'pipeline_root'])
def test_missing_explicit_runtime_paths_fail_without_fallback(tmp_path, key, monkeypatch):
    cfg = dict(python=sys.executable, engine_root=str(tmp_path), pipeline_root=str(tmp_path))
    cfg.pop(key)
    monkeypatch.setenv('GSSTUDIO_GSPLAT_PYTHON', sys.executable)
    with pytest.raises(RuntimeError, match='explicit'):
        isolated_module_command(cfg, 'unused')


def test_package_cli_bootstrap_help_read_only(tmp_path):
    """Run the exact template bootstrap with private package Python; never stage it."""
    runtime = PACKAGE / 'runtime/runtime.json'
    assert runtime.is_file(), 'Built directory package required for CPU help proof'
    cfg = json.loads(runtime.read_text(encoding='utf-8-sig'))
    for key in ('python','preprocess_python','engine_root','pipeline_root'):
        cfg[key] = str((runtime.parent / cfg[key]).resolve())
        assert Path(cfg[key]).is_relative_to(PACKAGE.resolve())
    template = (REPO/'desktop/scripts/gsstudio-cli.ps1').read_text(encoding='utf-8-sig')
    bootstrap = re.search(r"\$Bootstrap = @'\n(.*?)\n'@", template, re.S).group(1)
    env, poison = poisoned_environment(tmp_path)
    env.update(GSSTUDIO_RUNTIME=str(runtime), GSSTUDIO_DATA_ROOT=cfg['data_root'],
               GSSTUDIO_GPU_LOCK_PATH=cfg['gpu_lock_path'])
    # -B also prevents bytecode during private-site startup, before bootstrap runs.
    base = [cfg['preprocess_python'],'-I','-B','-u','-c',bootstrap,cfg['pipeline_root'],str(runtime)]
    for args in (['--help'], ['status','--help'], ['mask','--help'],
                 ['mask-review','--help'], ['mask-finalize','--help'],
                 ['train-control','--help'], ['train-options','--help'],
                 ['external','inspect','--help'], ['external','prepare','--help'],
                 ['external','train','--help']):
        proc = subprocess.run(base + args, env=env, cwd=poison[0], capture_output=True,
                              text=True, encoding='utf-8', timeout=45)
        assert proc.returncode == 0, (args, proc.stderr)
        assert 'Usage' in proc.stdout, (args, proc.stdout)
    # The packaged trainer remains independently CPU-launchable as well.
    proc = subprocess.run([cfg['python'],'-I','-B','-u','-c',
                           'import runpy,sys;sys.path[:0]=sys.argv[1:3];del sys.argv[1:3];'
                           'runpy.run_module("local_postshot.studio_training",run_name="__main__",alter_sys=True)',
                           cfg['engine_root'],cfg['pipeline_root'],'--help'],
                          env=env,cwd=poison[0],capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert '--dataset' in proc.stdout


def test_template_bootstrap_actual_argv_env_and_exit(tmp_path):
    cfg = private_config()
    pipeline = tmp_path / "pipeline 中文 O'Brien"
    runtime = tmp_path / 'runtime.json'
    module = pipeline / 'gsstudio/infrastructure/runtime'
    module.mkdir(parents=True)
    for directory in (pipeline/'gsstudio', module.parent, module):
        (directory/'__init__.py').touch()
    (module/'configuration.py').write_bytes((REPO/'src/gsstudio/infrastructure/runtime/configuration.py').read_bytes())
    (pipeline/'gsstudio/__main__.py').write_text('''import json,os,sys
print(json.dumps(dict(argv=sys.argv[1:],isolated=sys.flags.isolated,env={k:os.environ.get(k) for k in (
 'GSSTUDIO_RUNTIME','GSSTUDIO_DATA_ROOT','GSSTUDIO_GPU_LOCK_PATH','TORCH_HOME','GSSTUDIO_MEDIA_HELPER',
 'INSTA360_MEDIA_SDK_ROOT','GSSTUDIO_REALITYSCAN_CLI','GSSTUDIO_REALITYSCAN_EXPORT_PARAMS','PATH')})))
sys.exit(42)
''', encoding='utf-8')
    fields = dict(python=cfg['python'],preprocess_python=cfg['preprocess_python'],
                  pipeline_root=str(pipeline),engine_root=str(tmp_path/'engine'),
                  data_root=str(tmp_path/'configured-projects'),gpu_lock_path=str(tmp_path/'shared.lock'),
                  torch_home=str(tmp_path/'torch-cache'),media_helper=str(tmp_path/'helper.exe'),
                  mediasdk_root=str(tmp_path/'sdk'),realityscan=str(tmp_path/'rs.exe'),
                  realityscan_export_params=str(tmp_path/'export.xml'),
                  ffmpeg=str(tmp_path/'tools/ffmpeg.exe'),ffprobe=str(tmp_path/'tools/ffprobe.exe'),
                  colmap=str(tmp_path/'colmap/colmap.exe'))
    runtime.write_text(json.dumps(fields),encoding='utf-8')
    template = (REPO/'desktop/scripts/gsstudio-cli.ps1').read_text(encoding='utf-8-sig')
    bootstrap = re.search(r"\$Bootstrap = @'\n(.*?)\n'@", template, re.S).group(1)
    env = os.environ.copy()
    for key in ('GSSTUDIO_MEDIA_HELPER','INSTA360_MEDIA_SDK_ROOT','GSSTUDIO_REALITYSCAN_CLI',
                'GSSTUDIO_REALITYSCAN_EXPORT_PARAMS','TORCH_HOME'):
        env[key] = str(tmp_path/'explicit-env'/key)
    # Explicit environment wins over file configuration in every launcher.
    env.update(GSSTUDIO_RUNTIME='wrong-source.json', GSSTUDIO_DATA_ROOT=fields['data_root'],
               GSSTUDIO_GPU_LOCK_PATH=fields['gpu_lock_path'])
    args = ['中文 space',"O'Brien",'','quoted "word"','C:\\tail space\\']
    proc = subprocess.run([cfg['preprocess_python'],'-I','-u','-c',bootstrap,str(pipeline),str(runtime),*args],
                          env=env,capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert proc.returncode == 42,proc.stderr
    data=json.loads(proc.stdout)
    assert data['isolated'] == 1 and data['argv'] == args
    for field,key in (('data_root','GSSTUDIO_DATA_ROOT'),('gpu_lock_path','GSSTUDIO_GPU_LOCK_PATH'),
                      ('torch_home','TORCH_HOME'),('media_helper','GSSTUDIO_MEDIA_HELPER'),
                      ('mediasdk_root','INSTA360_MEDIA_SDK_ROOT'),('realityscan','GSSTUDIO_REALITYSCAN_CLI'),
                      ('realityscan_export_params','GSSTUDIO_REALITYSCAN_EXPORT_PARAMS')):
        expected = fields[field] if key in ('GSSTUDIO_DATA_ROOT','GSSTUDIO_GPU_LOCK_PATH') else str(tmp_path/'explicit-env'/key)
        assert Path(data['env'][key]) == Path(expected)
    assert data['env']['GSSTUDIO_RUNTIME'] == str(runtime)
    assert data['env']['PATH'].startswith(str(tmp_path/'tools') + os.pathsep)

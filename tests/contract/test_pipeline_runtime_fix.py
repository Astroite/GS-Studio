"""Actual private processes, canonical relative config and activation guard."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from gsstudio.infrastructure.runtime.configuration import runtime_config, child_environment


def test_canonical_relative_runtime_and_cache_lock(monkeypatch,tmp_path):
    from local_postshot.runtime import config
    from local_postshot.studio_bridge import child_environment as bridge_env
    path=tmp_path/'runtime.json'
    fields={'python':'core/python.exe','preprocess_python':'prep/python.exe',
        'pipeline_root':'../pipeline','engine_root':'../engine','data_root':'../projects',
        'torch_home':'../cache/model-cache','torch_extensions_dir':'../cache/torch_extensions',
        'gpu_lock_path':'../cache/gpu.lock','media_helper':'../tools/insta360/bin/gsstudio-media-helper.exe',
        'mediasdk_root':'../tools/insta360','realityscan_export_params':'../tools/rs/export.xml',
        'ffmpeg':'../tools/ffmpeg/ffmpeg.exe','ffprobe':'../tools/ffmpeg/ffprobe.exe',
        'colmap':'../tools/colmap/bin/colmap.exe'}
    path.write_text(json.dumps(fields),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',str(path))
    monkeypatch.setenv('LOCAL_POSTSHOT_RUNTIME',str(tmp_path/'wrong.json'))
    monkeypatch.delenv('GSSTUDIO_GPU_LOCK_PATH',raising=False)
    monkeypatch.delenv('GSSTUDIO_MEDIA_HELPER',raising=False)
    for cfg in (config(),runtime_config()):
        for key,value in fields.items():
            assert cfg[key] == str((path.parent/value).resolve())
        for env in (bridge_env(cfg),child_environment(cfg),child_environment(cfg,core=True)):
            assert env['TORCH_HOME'] == str((tmp_path/'../cache/model-cache').resolve())
            assert env['GSSTUDIO_GPU_LOCK_PATH'] == str((tmp_path/'../cache/gpu.lock').resolve())
            assert env['GSSTUDIO_MEDIA_HELPER'] == cfg['media_helper']


def test_actual_isolated_core_parent_catalog_and_torch_home(tmp_path):
    cfg=runtime_config()
    core=Path(cfg['python']);prep=Path(cfg['preprocess_python'])
    assert core.is_file() and prep.is_file(), 'Private runtimes are required for this integration test'
    cfg=dict(cfg,data_root=str(tmp_path/'read-only-projects'),
             gpu_lock_path=str(tmp_path/'unused-gpu.lock'))
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps(cfg),encoding='utf-8')
    script='''import sys,json,subprocess,os
sys.path.insert(0,sys.argv[1])
from local_postshot.studio_bridge import call,child_environment,config
cfg=config()
result=call('studio.catalog',{},request_id='isolated-catalog')
child=subprocess.run([cfg['preprocess_python'],'-I','-c','import os,json;print(json.dumps(dict(torch_home=os.environ.get("TORCH_HOME"),helper=os.environ.get("GSSTUDIO_MEDIA_HELPER"))))'],env=child_environment(cfg),capture_output=True,text=True,check=True)
print(json.dumps(dict(result=result,child=json.loads(child.stdout),torch_loaded='torch' in sys.modules)))
'''
    env={k:v for k,v in os.environ.items() if k.upper() in {'SYSTEMROOT','WINDIR','TEMP','TMP','COMSPEC','PATH'}}
    env['GSSTUDIO_RUNTIME']=str(runtime)
    process=subprocess.run([str(core),'-I','-c',script,cfg['engine_root']],
                           env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
    assert process.returncode == 0,process.stderr
    data=json.loads(process.stdout)
    assert not data['torch_loaded']
    assert data['result']['locations'] == [] and data['result']['runs'] == []
    assert data['child']['torch_home'] == cfg['torch_home']
    assert Path(data['child']['helper']) == Path(cfg['media_helper'])
    assert not Path(cfg['data_root']).exists() and not Path(cfg['gpu_lock_path']).exists()


def test_prepared_launcher_guard_and_no_qt_import(tmp_path,monkeypatch):
    from gsstudio.interfaces import launcher
    monkeypatch.delenv('GSSTUDIO_RUNTIME', raising=False)
    with pytest.raises(SystemExit) as exit_info:
        launcher.main(['--app-root',str(tmp_path/'absent')])
    assert exit_info.value.code == 2
    root=tmp_path/'package';(root/'runtime').mkdir(parents=True)
    (root/'GS Studio.exe').touch();(root/'runtime/runtime.json').write_text('{}',encoding='utf-8')
    calls=[]
    monkeypatch.setattr(launcher.subprocess,'Popen',lambda args,**kwargs:calls.append((args,kwargs)))
    assert launcher.main(['--app-root',str(root)]) == 0
    assert calls[0][0] == [str(root/'GS Studio.exe')]
    assert calls[0][1]['env']['GSSTUDIO_RUNTIME'] == str(root/'runtime/runtime.json')
    monkeypatch.setenv('GSSTUDIO_APP_ROOT', str(root))
    monkeypatch.setenv('GSSTUDIO_RUNTIME', str(tmp_path/'explicit-runtime.json'))
    assert launcher.main([]) == 0
    assert calls[-1][0] == [str(root/'GS Studio.exe')]
    assert calls[-1][1]['env']['GSSTUDIO_RUNTIME'] == str(tmp_path/'explicit-runtime.json')

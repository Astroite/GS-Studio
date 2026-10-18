"""Inactive previews must invoke the locked core; live trainers consume the queue."""
import subprocess
import sys
from types import SimpleNamespace
import pytest
from gsstudio.application import studio
from gsstudio.pipeline.training.data import json_write
from gsstudio.infrastructure.runtime import configuration

def config(tmp_path):
    return dict(python=sys.executable,engine_root=str(tmp_path),pipeline_root=str(tmp_path),
                runtime_path=str(tmp_path/'runtime.json'),gpu_lock_path=str(tmp_path/'gpu.lock'),data_root=str(tmp_path))


@pytest.mark.parametrize('status',['running','suspended','waiting_gpu','paused','stopped','succeeded'])
def test_preview_routes_by_saved_state(tmp_path,monkeypatch,status):
    monkeypatch.setattr(studio,'_experiment',lambda r:(tmp_path,{'package_sha256':'f'*64}))
    monkeypatch.setattr(studio,'_package_for_dispatch',lambda d:(tmp_path,{'images':[{'image':'images/a.png'}]}))
    json_write(tmp_path/'training.json',{'status':status})
    cfg=config(tmp_path)
    monkeypatch.setattr(configuration,'runtime_config',lambda:cfg)
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda argv,**kw:calls.append((argv,kw)) or SimpleNamespace(returncode=0,stderr=''))
    request=dict(action='preview.request',root=str(tmp_path),package_sha256='f'*64,image='images/a.png',request_id='a'*32)
    handled,result=studio.execute_additional(request)
    assert handled and result['request_id']=='a'*32
    assert bool(calls)==(status in {'paused','stopped','succeeded'})
    if calls:
        argv,kwargs=calls[0]
        assert argv[0]==sys.executable and '-I' in argv and '-B' in argv
        assert 'local_postshot.studio_saved_preview' in argv and kwargs['timeout']==300
    assert (tmp_path/'preview-request.json').is_file()


def test_failed_saved_preview_is_reported(tmp_path,monkeypatch):
    monkeypatch.setattr(studio,'_experiment',lambda r:(tmp_path,{'package_sha256':'f'*64}))
    monkeypatch.setattr(studio,'_package_for_dispatch',lambda d:(tmp_path,{'images':[{'image':'images/a.png'}]}))
    json_write(tmp_path/'training.json',{'status':'succeeded'})
    monkeypatch.setattr(configuration,'runtime_config',lambda:config(tmp_path))
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='Checkpoint identity changed'))
    with pytest.raises(RuntimeError,match='Checkpoint identity changed'):
        studio.execute_additional(dict(action='preview.request',root=str(tmp_path),package_sha256='f'*64,image='images/a.png'))

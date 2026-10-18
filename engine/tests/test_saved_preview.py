"""Saved-view identity and lock guards; CUDA publication is separately exercised."""
import json
from pathlib import Path
import importlib.util

import pytest
from local_postshot import project, studio_saved_preview, studio_training, training
from local_postshot.studio_dataset import prepare
from local_postshot.locks import operation_lock, is_active

spec = importlib.util.spec_from_file_location('saved_preview_fixture', Path(__file__).parents[1]/'scripts/studio_synthetic.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setenv('GSSTUDIO_GPU_LOCK_PATH', str(tmp_path/'gpu.lock'))
    package = tmp_path/'package'
    fixtures.make_package(package)
    p = prepare(package, tmp_path/'experiment')
    output = Path(p['path'])
    (output/'checkpoints').mkdir(exist_ok=True)
    (output/'checkpoints/latest.pt').write_bytes(b'CPU guard fixture, never loaded')
    request = dict(schema_version=1, request_id='b'*32,
                   package_sha256=p['studio_dataset']['package_sha256'],
                   image=p['reconstruction']['cameras'][4]['image'])
    meta = dict(status='succeeded', completed_steps=10, generation_id='g',
                package_sha256=request['package_sha256'], engine=studio_training.engine_identity(),
                checkpoint_sha256=project.digest(output/'checkpoints/latest.pt'))
    for name,value in [('preview-request',request),('training',meta),('config',{'steps':10})]:
        project.atomic_json(output/(name+'.json'), value)
    return output,p,request,meta


@pytest.mark.parametrize('change,match',[
    ('active','inactive'), ('package','identity'), ('camera','outside'),
    ('manifest','manifest changed'), ('camera_pose','camera/split'),
    ('source','source image changed'), ('checkpoint','checkpoint changed'), ('engine','engine identity')])
def test_changed_saved_evidence_rejected_before_loading_checkpoint(saved, monkeypatch, change, match):
    output,p,request,meta = saved
    if change=='active':meta['status']='running'
    if change=='package':request['package_sha256']='0'*64
    if change=='camera':request['image']='images/foreign.png'
    if change=='manifest':
        path=Path(p['studio_dataset']['package'])/'dataset.json'
        path.write_bytes(path.read_bytes()+b' ')
    if change=='camera_pose':
        p['reconstruction']['cameras'][4]['w2c'][0][3]+=1
        project.save(p)
    if change=='source':
        path=Path(p['studio_dataset']['package'])/request['image']
        path.write_bytes(path.read_bytes()+b'changed')
    if change=='checkpoint':(output/'checkpoints/latest.pt').write_bytes(b'tampered')
    if change=='engine':meta['engine']['name']='foreign-engine'
    project.atomic_json(output/'preview-request.json',request)
    project.atomic_json(output/'training.json',meta)
    monkeypatch.setattr(training,'load_checkpoint',lambda *a,**kw:pytest.fail('Invalid evidence reached checkpoint loader'))
    with pytest.raises(ValueError,match=match):
        studio_saved_preview.render_saved(output,request['request_id'])
    assert not is_active(output) and not (output/'preview.json').exists()


def test_superseded_request_does_not_load_or_publish(saved,monkeypatch):
    output,p,request,meta=saved
    monkeypatch.setattr(training,'load_checkpoint',lambda *a,**kw:pytest.fail('Stale request loaded checkpoint'))
    assert studio_saved_preview.render_saved(output,'a'*32)['status']=='superseded'
    assert not (output/'preview.json').exists()


@pytest.mark.parametrize('status',['paused','stopped','succeeded'])
def test_saved_checkpoint_validation_precedes_gpu_transfer(saved,monkeypatch,status):
    output,p,request,meta=saved
    meta['status']=status
    project.atomic_json(output/'training.json',meta)
    monkeypatch.setattr(training,'load_checkpoint',lambda *a,**kw:{'schema':0})
    with pytest.raises(ValueError,match='Incomplete independent core checkpoint'):
        studio_saved_preview.render_saved(output,request['request_id'])
    assert not is_active(output)

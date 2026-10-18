"""CPU JSONL contracts, including real process containment and multiplexing."""
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'engine'))
    from local_postshot import worker,studio_bridge
    return worker,studio_bridge


def _lines(stream):
    return [json.loads(x) for x in stream.getvalue().splitlines()]


def test_catalog_and_controls_respond_during_long_job(engine,monkeypatch,tmp_path):
    module,bridge=engine
    started,release=threading.Event(),threading.Event()
    def fake(method,params,*,request_id,emit=None,cancel=None):
        if params.get('action')=='create_run':
            started.set()
            while not release.wait(.01):
                if cancel.is_set():
                    raise bridge.BridgeError('CANCELLED','cancelled')
            return {'run_id':'run-one','next':'mask_review'}
        return {'read':request_id,'action':params.get('action')}
    monkeypatch.setattr(bridge,'call',fake)
    stream=io.StringIO();worker=module.Worker(stream)
    worker.receive({'id':'job','method':'studio.operation','params':{'root':str(tmp_path),'action':'create_run'}})
    assert started.wait(1)
    for rid,method,params in [('catalog','studio.catalog',{}),
                              ('control','studio.operation',{'action':'control_training'}),
                              ('preview','studio.operation',{'action':'preview.read'})]:
        worker.receive({'id':rid,'method':method,'params':params})
    for thread in list(worker.read_threads): thread.join(2)
    assert {'catalog','control','preview'} <= {x['id'] for x in _lines(stream)}
    assert worker.active['id']=='job'
    worker.receive({'id':'stale','method':'cancel','params':{'job_id':'job','generation_id':'old'}})
    assert not worker.control.is_set()
    active=worker.active.copy()
    worker.receive({'id':'cancel','method':'cancel','params':{'job_id':'job','generation_id':active['generation_id']}})
    worker.thread.join(2)
    terminals=[x for x in _lines(stream) if x['id']=='job' and x['type'] in {'error','result'}]
    assert len(terminals)==1 and terminals[0]['error']['code']=='CANCELLED'
    assert worker.active is None


def test_real_preprocess_child_catalog_is_structured_and_preserves_id(engine,monkeypatch,tmp_path):
    _,bridge=engine
    cfg={'runtime_path':str(tmp_path/'runtime.json'),'preprocess_python':sys.executable,
         'pipeline_root':str(Path(__file__).resolve().parents[2]/'src'),
         'data_root':str(tmp_path/'projects'),'gpu_lock_path':str(tmp_path/'gpu.lock')}
    Path(cfg['runtime_path']).write_text(json.dumps(cfg),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',cfg['runtime_path'])
    result=bridge.call('studio.catalog',{},request_id='中文-id')
    assert result['root']==str(Path(cfg['data_root']).resolve())
    assert all(result[k]==[] for k in ('locations','scenes','captures','runs','external_imports'))


@pytest.mark.parametrize('mode,expected',[('plain','CHILD_PROTOCOL'),('exit','CHILD_FAILED'),
                                         ('failure','KNOWN_FAILURE'),('wrong_id','CHILD_PROTOCOL'),
                                         ('zero_without_result','CHILD_FAILED'),('duplicate','CHILD_PROTOCOL')])
def test_child_failure_never_guesses_success(engine,monkeypatch,tmp_path,mode,expected):
    _,bridge=engine
    cfg={'runtime_path':str(tmp_path/'runtime.json'),'preprocess_python':sys.executable,
         'pipeline_root':str(Path(__file__).resolve().parents[2]/'src'),
         'data_root':str(tmp_path),'gpu_lock_path':str(tmp_path/'gpu.lock')}
    monkeypatch.setattr(bridge,'config',lambda:cfg)
    real=subprocess.Popen
    script="""import sys,json
r=json.loads(sys.stdin.readline()); mode=sys.argv[1]
if mode=='plain': print('success 100%')
elif mode=='failure': print(json.dumps({'type':'error','id':r['id'],'error':{'code':'KNOWN_FAILURE','message':'failed'}}))
elif mode=='wrong_id': print(json.dumps({'type':'result','id':'wrong','result':{}}))
elif mode=='duplicate':
 for i in range(2): print(json.dumps({'type':'result','id':r['id'],'result':{}}))
sys.exit(9 if mode=='exit' else 0)
"""
    monkeypatch.setattr(bridge.subprocess,'Popen',lambda args,**kwargs:real([sys.executable,'-c',script,mode],**kwargs))
    with pytest.raises(bridge.BridgeError) as error:
        bridge.call('studio.catalog',{},request_id='r')
    assert error.value.code==expected


def test_cancel_contains_and_terminates_descendants(engine,monkeypatch,tmp_path):
    psutil=pytest.importorskip('psutil')
    _,bridge=engine
    cfg={'runtime_path':str(tmp_path/'runtime.json'),'preprocess_python':sys.executable,
         'pipeline_root':str(Path(__file__).resolve().parents[2]/'src'),
         'data_root':str(tmp_path),'gpu_lock_path':str(tmp_path/'gpu.lock')}
    monkeypatch.setattr(bridge,'config',lambda:cfg)
    pid_file=tmp_path/'child.pid';real=subprocess.Popen
    script="""import subprocess,sys,pathlib,time
sys.stdin.readline()
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
pathlib.Path(sys.argv[1]).write_text(str(p.pid))
time.sleep(60)
"""
    monkeypatch.setattr(bridge.subprocess,'Popen',lambda args,**kwargs:real([sys.executable,'-c',script,str(pid_file)],**kwargs))
    cancel=threading.Event();errors=[]
    def run():
        try: bridge.call('studio.operation',{'action':'create_run'},request_id='r',cancel=cancel)
        except bridge.BridgeError as exc: errors.append(exc.code)
    thread=threading.Thread(target=run);thread.start()
    deadline=time.monotonic()+5
    while not pid_file.exists() and time.monotonic()<deadline: time.sleep(.02)
    assert pid_file.exists()
    pid=int(pid_file.read_text());cancel.set();thread.join(5)
    deadline=time.monotonic()+3
    while psutil.pid_exists(pid) and time.monotonic()<deadline: time.sleep(.02)
    assert errors==['CANCELLED'] and not thread.is_alive()
    assert not psutil.pid_exists(pid)


def test_shared_gpu_lock_between_core_and_pipeline(engine,tmp_path):
    _,bridge=engine
    from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
    with bridge.gpu_session():
        with pytest.raises(TimeoutError):
            with gpu_session(timeout=.05):
                raise AssertionError('Two runtimes acquired one GPU concurrently')


def test_runtime_prefers_gsstudio_and_isolates_site_paths(engine,monkeypatch,tmp_path):
    _,bridge=engine
    from local_postshot.runtime import config
    a,b=tmp_path/'a.json',tmp_path/'b.json'
    a.write_text(json.dumps({'data_root':'projects','preprocess_python':'prep/python.exe'}),encoding='utf-8')
    b.write_text(json.dumps({'data_root':'wrong'}),encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_RUNTIME',str(a));monkeypatch.setenv('LOCAL_POSTSHOT_RUNTIME',str(b))
    monkeypatch.setenv('PYTHONPATH','D:/other-cu129/Lib/site-packages')
    monkeypatch.setenv('PYTHONHOME','wrong')
    cfg=config();env=bridge.child_environment(cfg)
    assert cfg['data_root']==str(tmp_path/'projects')
    assert 'site-packages' not in env['PYTHONPATH'] and 'PYTHONHOME' not in env
    assert env['PYTHONNOUSERSITE']=='1'


def test_run_stage_stop_is_generation_bound_and_does_not_hard_cancel(engine,monkeypatch,tmp_path):
    module,bridge=engine
    started,release=threading.Event(),threading.Event();paths=[]
    def fake(method,params,*,request_id,emit=None,cancel=None):
        path=Path(params['stop_file']);paths.append(path);started.set()
        assert release.wait(3)
        assert path.is_file() and not cancel.is_set()
        return {'run_id':'r','next':'resume'}
    monkeypatch.setattr(bridge,'call',fake)
    stream=io.StringIO();worker=module.Worker(stream)
    worker.receive({'id':'job','method':'studio.operation','params':{'action':'resume_run','root':str(tmp_path)}})
    assert started.wait(1)
    active=worker.active.copy()
    worker.receive({'id':'old','method':'studio.operation','params':{'action':'control_run','job_id':'job',
                    'generation_id':'old','control_action':'stop_after_stage'}})
    assert not paths[0].exists()
    worker.receive({'id':'stop','method':'studio.operation','params':{'action':'control_run','job_id':'job',
                    'generation_id':active['generation_id'],'control_action':'stop_after_stage'}})
    assert paths[0].exists() and not worker.control.is_set()
    release.set();worker.thread.join(3)
    assert next(x for x in _lines(stream) if x['id']=='job')['result']['next']=='resume'
    assert not paths[0].parent.exists()


def test_worker_export_preserves_selected_deletions_and_protects_sources(engine,monkeypatch,tmp_path):
    import numpy as np
    module,_=engine
    from local_postshot import project,studio_edit
    from local_postshot.formats import write_ply,load_cloud
    p=project.create('CPU editor',tmp_path/'project')
    cloud={'means':np.array([[-.5,0,-3],[.5,0,-3],[1.5,0,-3],[2.5,0,-3]],np.float32),
           'scales':np.full((4,3),-3,np.float32),'quats':np.tile([1.,0,0,0],(4,1)),
           'opacities':np.zeros(4,np.float32),'sh':np.zeros((4,16,3),np.float32)}
    source=Path(p['path'])/'models'/'source.ply';write_ply(source,cloud)
    record=project.model_record(source,4);p['models']=[record]
    package=tmp_path/'package';package.mkdir();p['studio_dataset']={'package':str(package)}
    project.save(p)
    before=source.read_bytes()
    # Renderer boundary is explicitly simulated; picking/deletion/history/export are real CPU operations.
    monkeypatch.setattr(studio_edit,'_render',lambda cloud,camera,policy:np.zeros((32,32,3),np.float32))
    w=module.Worker(io.StringIO())
    params={'path':p['path'],'model_id':record['id'],'action':'select_rect','args':{
        'xray':True,'camera':{'width':32,'height':32,'K':[[10,0,16],[0,10,16],[0,0,1]],
                             'w2c':np.diag([1.,-1.,-1.,1.]).tolist()},'rectangle':[13,15,16,17]}}
    result=w._dispatch('editor.operation',params,lambda *a:None)
    assert result['selection_count']==1
    w._dispatch('editor.operation',{**params,'action':'delete','args':{}},lambda *a:None)
    destination=tmp_path/'edited.ply'
    exported=w._dispatch('model.export',{'path':p['path'],'model_id':record['id'],'file':str(destination)},lambda *a:None)
    assert exported['count']==3 and len(load_cloud(destination)['means'])==3
    assert source.read_bytes()==before
    for path in (source,package/'bad.ply',Path(p['path'])/'checkpoints'/'bad.ply'):
        with pytest.raises(ValueError,match='immutable'):
            w._dispatch('model.export',{'path':p['path'],'model_id':record['id'],'file':str(path),'bake_edits':False},lambda *a:None)


def test_direct_training_eof_cancels_identity_bound_gpu_wait(engine,tmp_path):
    _,bridge=engine
    from local_postshot.studio_control import TrainingControl,read_control_state
    from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
    output=tmp_path/'experiment';output.mkdir()
    controller=TrainingControl(output,'a'*64,waiting_for_gpu=True)
    shutdown=threading.Event();shutdown.set()
    with gpu_session():
        with bridge.training_control_scope(shutdown):
            with pytest.raises(bridge.BridgeError,match='GPU wait cancelled'):
                with bridge.gpu_session(cancel=controller):
                    raise AssertionError('Cancelled worker acquired GPU')
    state=read_control_state(output)
    request=json.loads((output/'control-request.json').read_text())
    assert state['status']=='stopping'
    assert request['generation_id']==controller.generation_id and request['package_sha256']=='a'*64

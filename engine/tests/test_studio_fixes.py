"""Second-pass contracts; all tests here are offline CPU tests."""
import copy
import json
from pathlib import Path
import random
import threading

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from local_postshot import project
from local_postshot.studio_auxiliary import masked_statistics,InvalidMask,StudioAuxiliary
from local_postshot.studio_checkpoint import assert_restored
from local_postshot.studio_control import TrainingControl
from local_postshot.studio_dataset import prepare,sparse_depth_availability,validate_package
from test_studio_core import fixtures


@pytest.fixture(autouse=True)
def isolated_cpu_gpu_lock(tmp_path,monkeypatch):
    monkeypatch.setenv('GSSTUDIO_GPU_LOCK_PATH',str(tmp_path/'cpu-only-gpu.lock'))


@pytest.mark.parametrize('kind',['uniform11','gaussian11'])
@pytest.mark.parametrize('padding',['valid','same'])
def test_ssim_uses_only_full_footprints(kind,padding):
    torch.manual_seed(12)
    x=torch.rand(1,31,33,3,dtype=torch.double,requires_grad=True)
    y=torch.rand_like(x)
    keep=torch.ones(1,31,33,dtype=torch.bool);keep[:,15,16]=False
    actual=masked_statistics(x,y,keep,kind=kind,padding=padding)
    # Independent explicit window reference: no renormalization, every tap kept.
    a,b=x.permute(0,3,1,2),y.permute(0,3,1,2)
    pad=5 if padding=='same' else 0
    a=F.pad(a,[pad]*4);b=F.pad(b,[pad]*4)
    mask=F.pad(keep,[pad]*4)
    axis=torch.arange(-5,6,dtype=torch.double)
    axis=torch.exp(-.5*(axis/1.5)**2) if kind=='gaussian11' else torch.ones(11,dtype=torch.double)
    kernel=(axis[:,None]*axis[None,:]);kernel/=kernel.sum()
    scores=[]
    for v in range(a.shape[-2]-10):
        for u in range(a.shape[-1]-10):
            if not mask[:,v:v+11,u:u+11].all():continue
            xx=a[:,:,v:v+11,u:u+11];yy=b[:,:,v:v+11,u:u+11]
            mx=(xx*kernel).sum((-1,-2));my=(yy*kernel).sum((-1,-2))
            vx=(xx.square()*kernel).sum((-1,-2))-mx.square()
            vy=(yy.square()*kernel).sum((-1,-2))-my.square()
            cov=(xx*yy*kernel).sum((-1,-2))-mx*my
            scores.append(((2*mx*my+.01**2)*(2*cov+.03**2))/((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
    torch.testing.assert_close(actual['ssim'],torch.stack(scores).mean(),rtol=0,atol=3e-15)
    # Ignored RGB/NaNs cannot affect values or propagate nonfinite gradients.
    nx=torch.where(keep[...,None],x.detach(),float('nan')).requires_grad_()
    ny=torch.where(keep[...,None],y,float('nan'))
    changed=masked_statistics(nx,ny,keep,kind=kind,padding=padding)
    for key in ('ssim','l1','mse','dssim_map'):
        torch.testing.assert_close(actual[key],changed[key],rtol=0,atol=0)
    (changed['l1']+1-changed['ssim']).backward()
    assert torch.isfinite(nx.grad).all() and nx.grad[~keep].count_nonzero()==0
    # A single ignored center excludes a full11x11 set of SSIM centers.
    assert actual['dssim_map'][0,10:21,11:22].count_nonzero()==0


def test_control_io_is_throttled_but_step_is_always_current(tmp_path,monkeypatch):
    import local_postshot.studio_control as module
    clock=[100.]
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    c=TrainingControl(tmp_path,'a'*64)
    reads=[];writes=[]
    read=module._read_object;write=module.atomic_json
    monkeypatch.setattr(module,'_read_object',lambda p:(reads.append(p),read(p))[1])
    monkeypatch.setattr(module,'atomic_json',lambda *a,**k:(writes.append(a[0]),write(*a,**k))[1])
    for step in range(1000):c.observe(step,lambda:None)
    assert c._last_step==999 and len(reads)==1 and not writes
    request={'schema_version':1,'request_id':'b'*32,'package_sha256':'a'*64,'generation_id':c.generation_id,'action':'checkpoint'}
    write(tmp_path/'control-request.json',request)
    assert c.observe(1000,lambda:pytest.fail('Request not yet observed')) is None
    clock[0]+=.151
    saves=[];c.observe(1001,lambda:saves.append(1001))
    assert saves==[1001] and len(reads)==2 and len(writes)==1
    state=json.loads((tmp_path/'control-state.json').read_text())
    assert state['step']==1001 and state['message']=='checkpoint_saved'


def test_studio_observer_1000_boundaries_no_per_step_publication_or_signature(tmp_path,monkeypatch):
    import local_postshot.training as training
    import local_postshot.studio_training as studio
    import local_postshot.studio_dataset as dataset
    from local_postshot.studio_preview import Preview
    clock=[100.]
    monkeypatch.setattr(studio.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(Preview,'tick',lambda *a,**k:None)
    package=fixtures.make_package(tmp_path/'package')
    writes=[];write=project.atomic_json
    def counted(file,*a,**kw):
        if Path(file).name=='training.json':writes.append(a[0].copy())
        return write(file,*a,**kw)
    monkeypatch.setattr(project,'atomic_json',counted)
    def fake_train(path,emit,control,pause,*,observer,checkpoint_observer,**kw):
        p=project.load(path)
        (Path(path)/'checkpoints/latest.pt').write_bytes(b'CPU synthetic checkpoint')
        checkpoint_observer(0)
        monkeypatch.setattr(dataset,'signature',lambda *a:pytest.fail('Per-step OS signature queried'))
        for step in range(1000):observer({}, {'step':step,'target':2000,'project':p},lambda:None)
        raise RuntimeError('CPU terminal evidence')
    monkeypatch.setattr(training,'train',fake_train)
    with pytest.raises(RuntimeError,match='CPU terminal'):studio.run(package,tmp_path/'out',steps=2000,emit=lambda *a:None)
    assert len(writes)<=6
    assert writes[-1]['status']=='failed' and writes[-1]['completed_steps']==999
    assert writes[-1]['checkpoint_sha256']==project.digest(tmp_path/'out/checkpoints/latest.pt')


def test_preview_request_poll_is_throttled(tmp_path,monkeypatch):
    import local_postshot.studio_preview as module
    package=fixtures.make_package(tmp_path/'package');p=prepare(package,tmp_path/'out')
    p['studio_training']={'preview_every':0}
    clock=[100.];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    preview=module.Preview(p,'generation')
    original=Path.read_text;reads=[]
    def counted(path,*a,**kw):
        if path.name=='preview-request.json':reads.append(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',counted)
    for step in range(1000):preview.tick({},step,2000)
    assert len(reads)==1
    clock[0]+=.151;preview.tick({},1001,2000)
    assert len(reads)==2


@pytest.mark.parametrize('empty_file',[False,True])
def test_depth_availability_and_effective_config(tmp_path,monkeypatch,empty_file):
    import local_postshot.training as training
    from local_postshot.studio_training import run
    package=fixtures.make_package(tmp_path/'package')
    meta=json.loads((package/'dataset.json').read_text())
    if empty_file:
        with np.load(package/'sparse_depth.npz') as d:arrays={k:d[k][:0] for k in d.files}
        np.savez(package/'sparse_depth.npz',**arrays)
    else:(package/'sparse_depth.npz').unlink()
    meta['sparse_depth_observations']=0
    fixtures.rehash(package,meta)
    info=sparse_depth_availability(package)
    assert not info['available'] and info['observations']==0 and info['reason']=='no_valid_training_anchors'
    def fail(*a,**k):raise RuntimeError('CPU stop before training')
    monkeypatch.setattr(training,'train',fail)
    with pytest.raises(RuntimeError,match='CPU stop'):run(package,tmp_path/'out',steps=1,emit=lambda *a:None)
    cfg=json.loads((tmp_path/'out/config.json').read_text())
    assert cfg['sparse_depth_requested'] and not cfg['use_sparse_depth'] and not cfg['sparse_depth_available']
    assert cfg['sparse_depth_status']=='unavailable'


def test_external_cancel_while_waiting_for_gpu_is_cpu_only(tmp_path,monkeypatch):
    import local_postshot.studio_bridge as bridge
    from local_postshot.studio_training import run
    package=fixtures.make_package(tmp_path/'package')
    lock=tmp_path/'occupied-gpu.lock';lock.write_bytes(b'0')
    monkeypatch.setenv('GSSTUDIO_GPU_LOCK_PATH',str(lock))
    event=threading.Event();event.set()
    # Hold the actual OS byte lock. Cancellation must be checked before trying
    # to acquire it and must never enter the GPU train body.
    import msvcrt
    with lock.open('r+b') as f:
        msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            value=run(package,tmp_path/'out',steps=1,external_control=event,emit=lambda *a:None)
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
    assert value['status']=='stopped' and value['completed_steps']==0
    assert value['reason']=='gpu_wait_cancelled' and not value['recoverable']


def test_exact_boundary_audit_detects_missing_or_zeroed_optimizer_aux(tmp_path,monkeypatch):
    from local_postshot.training import checkpoint_state,Sampler,atomic_checkpoint,load_checkpoint
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    package=fixtures.make_package(tmp_path/'package');p=prepare(package,tmp_path/'out')
    cfg={'photo_comp':True,'use_bilateral_grid':True,'use_sparse_depth':True}
    p['studio_training']=cfg
    aux=StudioAuxiliary(p,'cpu')
    params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.rand(shape)) for k,shape in
        {'means':(4,3),'scales':(4,3),'quats':(4,4),'opacities':(4,),'sh0':(4,1,3),'shN':(4,15,3)}.items()})
    optimizers={k:torch.optim.Adam([v]) for k,v in params.items()}
    schedulers={'means':torch.optim.lr_scheduler.ExponentialLR(optimizers['means'],.9)}
    for v in params.values():v.square().sum().backward()
    for o in optimizers.values():o.step();o.zero_grad(set_to_none=True)
    schedulers['means'].step()
    aux.photometric.grid.sum().backward();aux.step()
    sampler=Sampler([0,1,2,3]);sampler.next();sampler.next()
    state=checkpoint_state(params,optimizers,schedulers,{'accumulator':torch.rand(4)},{},sampler,2,p['settings'],
                           p['studio_dataset']['identity'],1.,auxiliary=aux,experiment_identity=cfg)
    file=tmp_path/'boundary.pt';atomic_checkpoint(file,state)
    saved=load_checkpoint(file,map_location='cpu');assert_restored(saved,state)
    # Reconstruct each state with the same production load_state_dict methods.
    fresh=torch.nn.ParameterDict({k:torch.nn.Parameter(v.clone()) for k,v in saved['parameters'].items()})
    restored={k:torch.optim.Adam([v]) for k,v in fresh.items()}
    schedule={'means':torch.optim.lr_scheduler.ExponentialLR(restored['means'],.9)}
    for k,o in restored.items():o.load_state_dict(saved['optimizers'][k])
    for k,s in schedule.items():s.load_state_dict(saved['schedulers'][k])
    loaded_aux=StudioAuxiliary(p,'cpu');loaded_aux.load_state_dict(saved['studio_auxiliary'])
    loaded_sampler=Sampler([0,1,2,3]);loaded_sampler.load_state_dict(saved['sampler'])
    random.setstate(saved['rng_python']);np.random.set_state(saved['rng_numpy']);torch.set_rng_state(saved['rng_torch'])
    current=checkpoint_state(fresh,restored,schedule,saved['strategy_state'],{},loaded_sampler,2,p['settings'],
                            p['studio_dataset']['identity'],1.,auxiliary=loaded_aux,experiment_identity=cfg)
    assert_restored(saved,current)
    for key in ('optimizers','studio_auxiliary','sampler','rng_torch','parameters','strategy_state'):
        bad=copy.deepcopy(current);bad[key]={}
        with pytest.raises(AssertionError,match=key):assert_restored(saved,bad)
    bad=copy.deepcopy(current)
    bad['optimizers']['quats']['state'][0]['exp_avg'].zero_()
    with pytest.raises(AssertionError,match='exp_avg'):assert_restored(saved,bad)
    bad=copy.deepcopy(current)
    bad['studio_auxiliary']['optimizer']['state'][0]['exp_avg'].zero_()
    with pytest.raises(AssertionError,match='exp_avg'):assert_restored(saved,bad)


def test_minimum_four_is_a_clear_native_capability_error(tmp_path):
    package=fixtures.make_package(tmp_path/'package')
    with np.load(package/'points.npz') as d:xyz,rgb=d['xyz'][:3],d['rgb'][:3]
    np.savez(package/'points.npz',xyz=xyz,rgb=rgb)
    meta=json.loads((package/'dataset.json').read_text());meta['initial_points']=3
    fixtures.rehash(package,meta)
    with pytest.raises(ValueError,match='at least 4.*kNN k=4'):validate_package(package)


def test_resize_label_cannot_silently_leave_original_pixels(tmp_path):
    package=fixtures.make_package(tmp_path/'package')
    with pytest.raises(ValueError,match='coherently resized immutable package'):
        prepare(package,tmp_path/'out',{'max_image_size':16})
    assert not (tmp_path/'out/project.json').exists()


def test_clean_derivative_retains_committed_history_source_and_undo(tmp_path):
    from test_studio_core import edit_project
    from local_postshot.studio_edit import dispatch,model_cloud,cleaned_history_record
    p,m,cloud=edit_project(tmp_path)
    original_path=m['path'];sha=project.digest(original_path)
    def edit(action,args=None,model_id=None):
        return dispatch({'path':p['path'],'model_id':model_id or m['id'],'action':action,'args':args or {}})
    # Existing inspector transform/crop and deletion stay in the same chain.
    from local_postshot.studio_edit import update_model_edits,_state,_persist
    update_model_edits(p,m,{'transform':{'position':[.1,.2,.3],'rotation':[0.,0.,0.],'scale':2.}})
    project.save(p)
    state,_=_state(p,m);state['selected'][3]=True;_persist(p,m,state)
    edit('delete')
    p=project.load(p['path']);m=p['models'][0]
    record=cleaned_history_record(p,m,np.array([0]),'11111111-1111-1111-1111-111111111111','cleaned')
    p['models'].append(record);project.save(p)
    assert record['path']==original_path and project.digest(original_path)==sha
    assert len(model_cloud(p,record)['means'])==2 and len(model_cloud(p,m)['means'])==3
    edit('undo',model_id=record['id'])
    p=project.load(p['path']);cleaned=p['models'][-1]
    np.testing.assert_array_equal(model_cloud(p,cleaned)['means'],model_cloud(p,p['models'][0])['means'])
    edit('undo',model_id=cleaned['id'])
    p=project.load(p['path'])
    assert len(model_cloud(p,p['models'][-1])['means'])==4
    edit('undo',model_id=cleaned['id'])
    p=project.load(p['path'])
    np.testing.assert_allclose(model_cloud(p,p['models'][-1])['means'],cloud['means'])
    assert project.digest(original_path)==sha

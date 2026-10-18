import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image
import torch

from local_postshot import project
from local_postshot.studio_dataset import prepare,validate_package,validate_project
from local_postshot.studio_auxiliary import (masked_statistics,InvalidMask,BilateralGrid,PanoramaExposure,
                                            select_photometric,StudioAuxiliary)
from local_postshot.studio_control import TrainingControl,submit_control,read_control_state
from local_postshot.studio_edit import dispatch,model_cloud,compositing_weights,center_candidates
from local_postshot.formats import write_ply,read_ply,write_spz,read_spz
from local_postshot.geometry import apply_edits
from local_postshot.studio_training import validate_checkpoint

spec = importlib.util.spec_from_file_location('studio_synthetic',Path(__file__).parents[1]/'scripts/studio_synthetic.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.fixture
def package(tmp_path):
    return fixtures.make_package(tmp_path/'package')


@pytest.fixture(autouse=True)
def isolated_cpu_gpu_lock(tmp_path,monkeypatch):
    # These orchestration tests mock all CUDA work; never wait for or contend
    # with the parent's production GPU slot merely to exercise file lifecycle.
    monkeypatch.setenv('GSSTUDIO_GPU_LOCK_PATH',str(tmp_path/'cpu-only-gpu.lock'))


def test_prepare_coordinate_projection_and_depth_normalization(package,tmp_path):
    meta = validate_package(package)
    p = prepare(package,tmp_path/'experiment',{'scene_normalization':{'scale':2.5,'center':[.1,.2,-.3]}})
    validate_project(p)
    np.testing.assert_array_equal(p['studio_dataset']['train_indices'],[0,1,2,3])
    np.testing.assert_array_equal(p['studio_dataset']['test_indices'],[4,5])
    with np.load(package/'points.npz') as d:
        raw = d['xyz']
    with np.load(Path(p['reconstruction']['path'])/'sparse_points.npz') as d:
        native = d['points']
    for row,camera in zip(meta['images'],p['reconstruction']['cameras']):
        old = raw@np.array(row['world_to_camera'])[:3,:3].T+np.array(row['world_to_camera'])[:3,3]
        new = native@np.array(camera['w2c'])[:3,:3].T+np.array(camera['w2c'])[:3,3]
        np.testing.assert_allclose(new,old*2.5,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose((new@np.array(row['K']).T)[:,:2]/new[:,2:],
                                   (old@np.array(row['K']).T)[:,:2]/old[:,2:],rtol=1e-6)
    with np.load(package/'sparse_depth.npz') as old,np.load(Path(p['reconstruction']['path'])/'sparse_depth.npz') as new:
        np.testing.assert_array_equal(new['depth'],old['depth']*2.5)
    assert prepare(package,tmp_path/'experiment',{'scene_normalization':{'scale':2.5,'center':[.1,.2,-.3]}})['id']==p['id']


@pytest.mark.parametrize('change',['split','time','pose','intrinsics','mask','empty_mask','colors','provenance','depth','depth_split','duplicate','path'])
def test_package_semantic_validation_rejects(package,change):
    meta = json.loads((package/'dataset.json').read_text())
    if change=='split': meta['images'][4]['frame']=0
    if change=='time': meta['images'][1]['timestamp_seconds']=9
    if change=='pose': meta['images'][0]['world_to_camera'][0][0]=2
    if change=='intrinsics': meta['images'][0]['K'][0][0]=-1
    if change in ('mask','empty_mask'):
        Image.new('L',(32,32),128 if change=='mask' else 0).save(package/meta['images'][0]['mask'])
    if change=='colors':
        with np.load(package/'points.npz') as d: xyz,rgb=d['xyz'],d['rgb']
        rgb[0,0]+=1
        np.savez(package/'points.npz',xyz=xyz,rgb=rgb)
    if change=='provenance': meta['initialization_colors']='source_colmap'
    if change in ('depth','depth_split'):
        with np.load(package/'sparse_depth.npz') as d: anchors={k:d[k] for k in d.files}
        anchors['depth' if change=='depth' else 'row_index'][0]=99 if change=='depth' else 4
        np.savez(package/'sparse_depth.npz',**anchors)
    if change=='duplicate': meta['images'][1]['image']=meta['images'][0]['image']
    if change=='path':
        meta['files']['../escape']='0'*64
        project.atomic_json(package/'dataset.json',meta)
    else: fixtures.rehash(package,meta)
    with pytest.raises((ValueError,FileNotFoundError)):
        validate_package(package)


def test_hash_validation_and_cache_detects_changed_sources(package):
    validate_package(package)
    file=package/'images/00000000.png'
    file.write_bytes(file.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash changed'):
        validate_package(package)


def test_cache_detects_same_size_write_with_restored_mtime(package):
    validate_package(package)
    file=package/'images/00000000.png'
    before=file.stat()
    data=bytearray(file.read_bytes());data[-1]^=1
    file.write_bytes(data)
    os.utime(file,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='hash changed'):validate_package(package)


def test_package_is_never_written_and_prepared_artifact_checked(package,tmp_path):
    before = {f:project.digest(f) for f in package.rglob('*') if f.is_file()}
    p=prepare(package,tmp_path/'exp')
    assert before=={f:project.digest(f) for f in before}
    file=Path(p['reconstruction']['path'])/'sparse_points.npz'
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):validate_project(p)
    with pytest.raises(ValueError):prepare(package,package/'bad')


def test_rewriting_prepared_hash_cannot_bless_wrong_initialization(package,tmp_path):
    p=prepare(package,tmp_path/'exp')
    file=Path(p['reconstruction']['path'])/'sparse_points.npz'
    with np.load(file) as data:points,colors=data['points'],data['colors']
    points[0,0]+=1
    np.savez(file,points=points,colors=colors)
    p['reconstruction']['artifacts']['sparse_points.npz']=project.digest(file)
    with pytest.raises(ValueError,match='conversion/provenance'):validate_project(p)


@pytest.mark.parametrize('kind',['uniform11','gaussian11'])
@pytest.mark.parametrize('padding',['same','valid'])
def test_masked_statistics_excludes_values_and_gradients(kind,padding):
    torch.manual_seed(9)
    x=torch.rand((1,25,27,3),dtype=torch.double,requires_grad=True)
    y=torch.rand_like(x)
    keep=torch.ones((1,25,27),dtype=torch.bool);keep[:,5:9,6:10]=False
    a=masked_statistics(x,y,keep,kind=kind,padding=padding)
    changed=torch.where(keep[...,None],x.detach(),999).requires_grad_()
    altered=torch.where(keep[...,None],y,-333)
    b=masked_statistics(changed,altered,keep,kind=kind,padding=padding)
    for k in ('l1','mse','ssim'): torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    loss=a['l1']+1-a['ssim'];loss.backward()
    assert x.grad[~keep].count_nonzero()==0
    assert torch.isfinite(x.grad).all() and x.grad[keep].abs().sum()>0
    # Independent finite difference at a mask boundary pixel.
    position=(0,5,5,0);eps=1e-6
    values=[]
    for sign in (-1,1):
        z=x.detach().clone();z[position]+=sign*eps
        c=masked_statistics(z,y,keep,kind=kind,padding=padding)
        values.append(float(c['l1']+1-c['ssim']))
    assert float(x.grad[position])==pytest.approx((values[1]-values[0])/(2*eps),abs=1e-7)


def test_invalid_masks_explicitly_fail_without_full_window():
    x=torch.rand((1,12,12,3),requires_grad=True)
    keep=torch.zeros((1,12,12),dtype=torch.bool)
    with pytest.raises(InvalidMask):masked_statistics(x,x,keep)
    keep[:,6,6]=True
    with pytest.raises(InvalidMask,match='No full SSIM window'):masked_statistics(x,x.detach(),keep)


@pytest.mark.parametrize('bilateral',[False,True])
def test_photometric_identity_gradient_and_optimizer_resume(package,tmp_path,bilateral):
    p=prepare(package,tmp_path/'exp')
    cfg={'photo_comp':True,'use_bilateral_grid':bilateral,'use_sparse_depth':True}
    p['studio_training']=cfg
    aux=StudioAuxiliary(p,'cpu')
    assert isinstance(aux.photometric,BilateralGrid if bilateral else PanoramaExposure)
    rgb=torch.rand((1,14,16,3),requires_grad=True)
    out=aux.compensate(rgb,2)
    torch.testing.assert_close(out,rgb)
    depth=torch.ones((1,32,32),requires_grad=True)
    loss=(out-.1).square().mean()+aux.extra_loss(depth,2,out)
    loss.backward();aux.step()
    assert rgb.grad.abs().sum()>0 and depth.grad.abs().sum()>0
    assert any(v.abs().sum()>0 for v in aux.optimizer.state[next(iter(aux.photometric.parameters()))].values() if isinstance(v,torch.Tensor))
    saved=copy.deepcopy(aux.state_dict())
    restored=StudioAuxiliary(p,'cpu');restored.load_state_dict(saved)
    torch.testing.assert_close(aux.compensate(rgb,2),restored.compensate(rgb,2),rtol=0,atol=0)
    with pytest.raises(ValueError):restored.compensate(rgb,4)
    wrong=copy.deepcopy(saved);wrong['identity']={**cfg,'photo_comp':False}
    with pytest.raises(ValueError):restored.load_state_dict(wrong)
    assert select_photometric(False,True,[0,1]) is None


def test_bilateral_single_cell_regularization_and_panorama_anchor():
    grid=BilateralGrid(1,spatial=1,bins=1)
    assert torch.isfinite(grid.regularization())
    exposure=PanoramaExposure([2.,0.,1.])
    exposure.values.data[:]=.1
    x=torch.rand(3,4,3)
    torch.testing.assert_close(exposure(x,0),x)
    exposure.regularization().backward()
    assert exposure.values.grad[0].count_nonzero()==0


def test_controls_generation_hot_suspend_and_release_pause(tmp_path):
    c=TrainingControl(tmp_path,'a'*64,io_interval=0)
    saves=[]
    submit_control(tmp_path,'a'*64,'suspend')
    t=threading.Thread(target=lambda:c.observe(7,lambda:saves.append(7)),daemon=True)
    t.start()
    deadline=time.monotonic()+5
    while read_control_state(tmp_path)['status']!='suspended' and time.monotonic()<deadline:time.sleep(.01)
    assert t.is_alive() and not saves
    submit_control(tmp_path,'a'*64,'checkpoint')
    while read_control_state(tmp_path).get('message')!='checkpoint_saved' and time.monotonic()<deadline:time.sleep(.01)
    submit_control(tmp_path,'a'*64,'unpause');t.join(5)
    assert not t.is_alive() and saves==[7]
    submit_control(tmp_path,'a'*64,'pause')
    assert c.observe(8,lambda:saves.append(8))=='pause'
    assert read_control_state(tmp_path)['status']=='paused'
    old=json.loads((tmp_path/'control-request.json').read_text())
    fresh=TrainingControl(tmp_path,'a'*64)
    old['request_id']='new-old-request'
    project.atomic_json(tmp_path/'control-request.json',old)
    assert fresh.observe(0,lambda:pytest.fail('stale request saved')) is None
    assert read_control_state(tmp_path)['message']=='identity_or_action_mismatch'


def test_controls_import_without_training_or_torch():
    code="import sys; from local_postshot.studio_control import TrainingControl; assert 'torch' not in sys.modules; assert 'gsstudio' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def edit_project(tmp_path):
    p=project.create('edit',tmp_path/'edit')
    cloud={'means':np.array([[-1.,0.,3.],[0.,0.,3.],[1.,0.,3.],[0.,0.,4.]],np.float32),
           'scales':np.full((4,3),-2,np.float32),'quats':np.tile([1.,0.,0.,0.],(4,1)).astype(np.float32),
           'sh':np.random.default_rng(3).normal(0,.1,(4,16,3)).astype(np.float32),'opacities':np.zeros(4,np.float32)}
    file=Path(p['path'])/'models/source.ply';write_ply(file,cloud)
    m=project.model_record(file,4);p['models']=[m];project.save(p)
    return p,m,cloud


def test_editor_undo_redo_reopen_preview_export_and_source_identity(tmp_path):
    p,m,cloud=edit_project(tmp_path)
    base={'path':p['path'],'model_id':m['id']}
    def call(action,args=None):return dispatch({**base,'action':action,'args':args or {}})
    transform={'position':[1,2,3],'rotation':[.2,.4,.1],'scale':2}
    call('preview',{'transform':transform})
    loaded=project.load(p['path']);model=loaded['models'][0]
    np.testing.assert_array_equal(model_cloud(loaded,model)['means'],cloud['means'])
    preview=model_cloud(loaded,model,include_preview=True)
    np.testing.assert_allclose(preview['means'],apply_edits(cloud,transform)['means'])
    call('cancel_preview')
    call('preview',{'transform':transform});result=call('commit')
    assert result['can_undo'] and not result['can_redo']
    result=call('undo');assert result['can_redo']
    call('redo')
    # Selection itself remains bounded in the RPC; CPU tests use exact xray
    # center candidates and inject the renderer only, never test fake visibility.
    state,source=__import__('local_postshot.studio_edit',fromlist=['_state'])._state(project.load(p['path']),project.load(p['path'])['models'][0])
    state['selected'][1]=True
    from local_postshot.studio_edit import _persist
    loaded=project.load(p['path']);_persist(loaded,loaded['models'][0],state)
    result=call('delete');assert result['selection_count']==0
    loaded=project.load(p['path']);edited=model_cloud(loaded,loaded['models'][0])
    expected=apply_edits({k:v[[0,2,3]] for k,v in cloud.items()},transform)
    for k in edited:np.testing.assert_allclose(edited[k],expected[k],atol=2e-7)
    output=tmp_path/'export.ply';write_ply(output,edited)
    for k,v in read_ply(output).items():np.testing.assert_allclose(v,expected[k],atol=2e-7)
    call('undo');call('redo')
    assert len(model_cloud(project.load(p['path']),project.load(p['path'])['models'][0])['means'])==3
    with pytest.raises(ValueError):call('preview',{'transform':{'position':[float('nan'),0,0]}})
    # Never silently applies original point indices to a different source.
    write_ply(m['path'],{k:v[[0,1,2]] for k,v in cloud.items()})
    with pytest.raises(ValueError,match='source hash changed'):call('state')


def test_compositing_exact_threshold_and_center_projection():
    alpha=torch.tensor([.8,.02,1/255,.5,.99,.8],dtype=torch.double)
    pixels=torch.tensor([0,0,1,1,2,2])
    expected=torch.tensor([.8,.004,1/255,(1-1/255)*.5,.99,.008],dtype=torch.double)
    torch.testing.assert_close(compositing_weights(alpha,pixels),expected,rtol=1e-13,atol=1e-15)
    assert (compositing_weights(alpha,pixels)>=1/255).tolist()==[True,True,True,True,True,True]
    assert compositing_weights(torch.tensor([.999,.5]),torch.tensor([0,0]))[1]<1/255
    c={'means':np.array([[0,0,3],[0,0,-3],[10,0,3]])}
    camera={'w2c':np.eye(4),'K':[[10,0,5],[0,10,5],[0,0,1]],'width':10,'height':10}
    assert center_candidates(c,camera,[4,4,6,6]).tolist()==[True,False,False]
    assert center_candidates({'means':np.array([[1.47,0,3.]])},camera,[0,0,10,10]).tolist()==[True]


def test_checkpoint_full_state_and_identity_contract(package,tmp_path,monkeypatch):
    from local_postshot.training import checkpoint_state,Sampler,atomic_checkpoint,load_checkpoint
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    cfg={'photo_comp':True,'use_bilateral_grid':False,'use_sparse_depth':True,'steps':10}
    p=prepare(package,tmp_path/'exp');p['studio_training']=cfg
    aux=StudioAuxiliary(p,'cpu')
    params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.zeros(shape)) for k,shape in
                                  {'means':(4,3),'scales':(4,3),'quats':(4,4),'opacities':(4,),
                                   'sh0':(4,1,3),'shN':(4,15,3)}.items()})
    optimizers={k:torch.optim.Adam([v]) for k,v in params.items()}
    sampler=Sampler([0,1,2,3]);sampler.next()
    state=checkpoint_state(params,optimizers,{}, {'s':torch.ones(4)}, {},sampler,3,p['settings'],
                           p['studio_dataset']['identity'],1.,auxiliary=aux,experiment_identity=cfg)
    validate_checkpoint(state,cfg,p['studio_dataset']['identity'],p['settings'])
    file=tmp_path/'checkpoint.pt';atomic_checkpoint(file,state)
    saved=load_checkpoint(file,map_location='cpu')
    validate_checkpoint(saved,cfg,p['studio_dataset']['identity'],p['settings'])
    assert set(['rng_python','rng_numpy','rng_torch','rng_cuda','strategy_state','sampler'])<=saved.keys()
    for key in ('studio_auxiliary','rng_python','sampler','optimizers'):
        bad=copy.deepcopy(state);del bad[key]
        with pytest.raises(ValueError):validate_checkpoint(bad,cfg,p['studio_dataset']['identity'],p['settings'])
    with pytest.raises(ValueError):validate_checkpoint(saved,{**cfg,'photo_comp':False},p['studio_dataset']['identity'],p['settings'])


def test_editor_state_read_does_not_mutate_revision_and_crop_history(tmp_path):
    p,m,cloud=edit_project(tmp_path)
    def call(action,args=None):return dispatch({'path':p['path'],'model_id':m['id'],'action':action,'args':args or {}})
    first=call('state');second=call('state')
    assert first['revision']==second['revision']==0
    crop={'enabled':True,'min':[-.1,-1,2],'max':[.1,1,5],'invert':False}
    call('preview',{'crop':crop});call('commit')
    loaded=project.load(p['path'])
    assert len(model_cloud(loaded,loaded['models'][0])['means'])==2
    call('undo');loaded=project.load(p['path'])
    assert len(model_cloud(loaded,loaded['models'][0])['means'])==4
    call('redo');loaded=project.load(p['path'])
    assert len(model_cloud(loaded,loaded['models'][0])['means'])==2


def test_cli_lifecycle_outputs_and_failure_preserves_checkpoint(package,tmp_path,monkeypatch):
    """CPU orchestration test; CUDA numerical acceptance uses separate script."""
    import local_postshot.training as training
    import local_postshot.studio_training as studio
    from local_postshot.studio_preview import Preview
    from local_postshot.studio_bridge import gpu_session
    monkeypatch.setattr(Preview,'tick',lambda *a,**k:None)
    calls=[]
    def fake_train(path,emit,control,pause,*,resume=False,observer,checkpoint_observer):
        p=project.load(path)
        file=Path(path)/'checkpoints/latest.pt'
        # This is deliberately opaque bytes: lifecycle success must verify
        # hashes/records, not misrepresent this CPU mock as resumable training.
        file.write_bytes(b'last-good-core-checkpoint')
        checkpoint_observer(3)
        calls.append(path)
        observer({}, {'step':3,'target':3,'project':p},lambda:None)
        cloud={'means':np.array([[0,0,-3]],np.float32),'scales':np.full((1,3),-2,np.float32),
               'quats':np.array([[1,0,0,0]],np.float32),'opacities':np.zeros(1,np.float32),'sh':np.zeros((1,16,3),np.float32)}
        file=Path(path)/'models/trained.ply';write_ply(file,cloud)
        m=project.model_record(file,1);m['role']='training'
        p.update(status='trained',models=[m],training={'step':3})
        project.save(p)
        return p
    def fake_evaluate(p,emit,control):
        output=Path(p['path'])/'evaluation';output.mkdir(exist_ok=True)
        result={'schema_version':1,'images':[{'image':r['image'],'psnr':10.,'ssim':.8} for r in validate_package(package)['images'] if r['split']=='validation']}
        project.atomic_json(output/'metrics.json',result)
        return result
    monkeypatch.setattr(training,'train',fake_train)
    monkeypatch.setattr(studio,'evaluate_project',fake_evaluate)
    output=tmp_path/'success'
    result=studio.run(package,output,steps=3,emit=lambda *a:None)
    assert result['status']=='succeeded' and result['steps']==result['completed_steps']==3
    assert result['checkpoint_sha256']==project.digest(output/'checkpoints/latest.pt')
    assert result['model_sha256']==project.digest(output/'model.ply')
    assert result['evaluation_sha256']==project.digest(output/'evaluation/metrics.json')
    assert read_control_state(output)['status']=='succeeded'
    with pytest.raises(ValueError,match='already succeeded'):studio.run(package,output,steps=3)
    failed=tmp_path/'failure'
    def broken(*args,**kwargs):
        Path(args[0],'checkpoints/latest.pt').write_bytes(b'last-good')
        raise RuntimeError('injected OOM boundary')
    monkeypatch.setattr(training,'train',broken)
    with pytest.raises(RuntimeError,match='OOM'):studio.run(package,failed,steps=3,emit=lambda *a:None)
    assert (failed/'checkpoints/latest.pt').read_bytes()==b'last-good'
    evidence=json.loads((failed/'training.json').read_text())
    assert evidence['status']=='failed' and evidence['recoverable']
    assert evidence['checkpoint_sha256']==project.digest(failed/'checkpoints/latest.pt')
    assert read_control_state(failed)['status']=='failed'


def test_preview_camera_identity_hashes_and_no_heldout_fitting(package,tmp_path,monkeypatch):
    from local_postshot.studio_preview import Preview
    import local_postshot.training as training
    p=prepare(package,tmp_path/'exp');p['studio_training']={'preview_every':1}
    observed=[]
    def render(params,camera,degree,**kwargs):
        observed.append(camera)
        return torch.full((1,camera['height'],camera['width'],3),.25),None,None
    monkeypatch.setattr(training,'render_parameters',render)
    preview=Preview(p,'g')
    request={'schema_version':1,'package_sha256':p['studio_dataset']['package_sha256'],
             'request_id':'b'*32,'image':p['reconstruction']['cameras'][4]['image']}
    project.atomic_json(Path(p['path'])/'preview-request.json',request)
    preview.tick({},2,10)
    record=json.loads((Path(p['path'])/'preview.json').read_text())
    assert record['request_id']==request['request_id'] and record['split']=='validation'
    assert record['camera']['w2c']==p['reconstruction']['cameras'][4]['w2c']
    for kind in ('source','render'):
        assert record[kind+'_sha256']==project.digest(Path(p['path'])/record[kind+'_file'])
    assert np.array(Image.open(Path(p['path'])/record['source_file'])).shape==np.array(Image.open(Path(p['path'])/record['render_file'])).shape


def test_heldout_evaluation_respects_masks(package,tmp_path,monkeypatch):
    import local_postshot.training as training
    import local_postshot.studio_training as studio
    p=prepare(package,tmp_path/'exp')
    cloud={'means':np.array([[0,0,-3]],np.float32),'scales':np.full((1,3),-2,np.float32),
           'quats':np.array([[1,0,0,0]],np.float32),'opacities':np.zeros(1,np.float32),'sh':np.zeros((1,16,3),np.float32)}
    write_ply(Path(p['path'])/'model.ply',cloud)
    original_tensor=torch.as_tensor
    monkeypatch.setattr(torch,'as_tensor',lambda data,*a,**k:original_tensor(data,*a,**{key:value for key,value in k.items() if key!='device'}))
    monkeypatch.setattr(training,'tensors_from_cloud',lambda c:c)
    def render(params,camera,degree,**kwargs):
        with Image.open(package/camera['image']) as im:rgb=np.array(im).astype(np.float32)/255
        rgb[:4,:4]=999
        return original_tensor(rgb)[None],None,None
    monkeypatch.setattr(training,'render_parameters',render)
    result=studio.evaluate_project(p)
    assert len(result['images'])==2
    for row in result['images']:
        assert row['psnr']==pytest.approx(120) and row['ssim']==pytest.approx(1,abs=1e-4)


def test_exact_visibility_pairs_preserve_compositor_termination_rules():
    from local_postshot.splat3_stats import contribution_stats
    from local_postshot.splat3_renderer import reference_projected
    # Both overlap the one pixel. Second contribution alpha*T is >1/255,
    # but public gsplat excludes the Gaussian that crosses next-T<=1e-4.
    xy=torch.tensor([[[.5,.5],[.5,.5]]],dtype=torch.float32)
    conics=torch.tensor([[[1.,0,1.],[1.,0,1.]]])
    opacity=torch.tensor([[.99,.999]])
    offsets=torch.zeros((1,1,1),dtype=torch.int32)
    ids=torch.tensor([0,1],dtype=torch.int32)
    public=contribution_stats(xy,conics,opacity,offsets,ids,torch.ones((1,1)),1,1,16,backend='native_cpu_v1')
    assert (public.max_weight>=1/255).tolist()==[True,False]
    native=reference_projected(xy,conics,torch.ones((1,2,3)),opacity,offsets,ids,1,1,16)
    assert (native['stats'].max_weight>=1/255).tolist()==[True,True]


def test_editor_spz_export_and_inspector_share_history(tmp_path):
    p,m,cloud=edit_project(tmp_path)
    transform={'position':[.1,.2,.3],'rotation':[.1,.2,.3],'scale':1.5}
    dispatch({'path':p['path'],'model_id':m['id'],'action':'preview','args':{'transform':transform}})
    dispatch({'path':p['path'],'model_id':m['id'],'action':'commit'})
    changed={**transform,'scale':2.}
    project.update(p['path'],models=[{'id':m['id'],'transform':changed}])
    loaded=project.load(p['path']);baked=model_cloud(loaded,loaded['models'][0])
    np.testing.assert_allclose(baked['means'],apply_edits(cloud,changed)['means'])
    dispatch({'path':p['path'],'model_id':m['id'],'action':'undo'})
    loaded=project.load(p['path']);baked=model_cloud(loaded,loaded['models'][0])
    np.testing.assert_allclose(baked['means'],apply_edits(cloud,transform)['means'])
    pytest.importorskip('spz')
    file=tmp_path/'edited.spz';write_spz(file,baked)
    decoded=read_spz(file)
    np.testing.assert_allclose(decoded['means'],baked['means'],atol=1e-3)
    assert len(decoded['means'])==len(cloud['means'])


def test_waiting_gpu_control_is_bound_and_stop_only(tmp_path):
    c=TrainingControl(tmp_path,'c'*64,waiting_for_gpu=True)
    with pytest.raises(RuntimeError,match='Only stop'):submit_control(tmp_path,'c'*64,'pause')
    with pytest.raises(RuntimeError,match='identity'):submit_control(tmp_path,'d'*64,'stop')
    submit_control(tmp_path,'c'*64,'stop')
    assert c.is_set() and read_control_state(tmp_path)['status']=='stopping'


def test_explicit_step_zero_retry_before_config_and_legacy_rejection(package,tmp_path,monkeypatch):
    import local_postshot.training as training
    from local_postshot.studio_training import run
    def broken(*a,**k):raise RuntimeError('intentional pre-checkpoint failure')
    monkeypatch.setattr(training,'train',broken)
    output=tmp_path/'before-config';output.mkdir()
    project.atomic_json(output/'dispatch.json',{'status':'failed'})
    with pytest.raises(RuntimeError,match='intentional'):
        run(package,output,steps=1,restart_from_step_zero=True,emit=lambda *a:None)
    assert json.loads((output/'training.json').read_text())['status']=='failed'
    assert not (output/'checkpoints/latest.pt').exists()
    # A second explicit retry is safe with matching config and zero evidence.
    with pytest.raises(RuntimeError,match='intentional'):
        run(package,output,steps=1,restart_from_step_zero=True,emit=lambda *a:None)
    old=tmp_path/'old';old.mkdir();(old/'checkpoint.pt').write_bytes(b'old GS state')
    with pytest.raises(ValueError,match='Old GS checkpoints'):
        run(package,old,steps=1,restart_from_step_zero=True)
    assert (old/'checkpoint.pt').read_bytes()==b'old GS state'
    model=tmp_path/'unknown';model.mkdir();(model/'model.ply').write_bytes(b'user model')
    with pytest.raises(ValueError,match='Unrecognized experiment'):
        run(package,model,steps=1,restart_from_step_zero=True)
    assert (model/'model.ply').read_bytes()==b'user model'


def test_cli_rejection_is_jsonl_without_gpu_initialization(tmp_path):
    result=subprocess.run([sys.executable,'-m','local_postshot.studio_training','--dataset',str(tmp_path/'missing'),
                           '--output',str(tmp_path/'out'),'--steps','1'],capture_output=True,text=True,timeout=30)
    assert result.returncode==1
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows)==1 and rows[0]['type']=='error'
    assert not (tmp_path/'out').exists()

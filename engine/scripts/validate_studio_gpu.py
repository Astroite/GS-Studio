"""Parent-scheduled real CUDA acceptance. NEVER runs without --allow-cuda.

Example (after parent grants GPU slot):
  python engine/scripts/validate_studio_gpu.py --allow-cuda --output E:/.../core-gpu-001
No existing outputs or package sources are overwritten. JSON report records
actual equivalence deltas; a failing assertion exits nonzero. CPU mocks are not
used here. Quality v2 defaults; v5/v6 are explicit parent-selected checks.
"""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import traceback


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cuda',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=40)
    parser.add_argument('--recipe',choices=['quality_v2','observed_v5','observed_v6'],default='quality_v2')
    parser.add_argument('--mode',choices=['bilateral','panorama','off','all'],default='bilateral')
    args=parser.parse_args(argv)
    if not args.allow_cuda:
        parser.error('Parent GPU authorization required: --allow-cuda')
    if args.steps<20:
        parser.error('At least20 steps required to exercise controls/resume')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from PIL import Image
    from studio_synthetic import make_package
    from local_postshot import project
    from local_postshot.studio_training import run
    from local_postshot.training import load_checkpoint,render
    from local_postshot.studio_control import submit_control,read_control_state
    from local_postshot.studio_edit import dispatch,model_cloud,project_points
    from local_postshot.formats import read_ply,write_ply,read_spz,write_spz
    from local_postshot.studio_checkpoint import assert_restored
    from studio_selection_cases import run_cases
    package=make_package(root/'package',size=48)
    settings={'max_splats':64,'checkpoint_every':5,'preview_every':10,'sh_degree_interval':5,
              'refine_start':100000,'refine_stop':0,'reset_every':3000}
    report={'scope':'real CUDA synthetic contract; separate real-scene acceptance required',
            'torch':torch.__version__,'recipe':args.recipe,'cases':[]}
    def wait_for(predicate,description,timeout=120):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            try:
                value=predicate()
                if value:return value
            except (FileNotFoundError,PermissionError,json.JSONDecodeError):pass
            time.sleep(.02)
        raise TimeoutError(description)
    def read(path):return json.loads(path.read_text(encoding='utf-8'))
    max_errors={}
    def diagnostic(a,b,path):
        if isinstance(a,torch.Tensor):
            assert a.shape==b.shape and a.dtype==b.dtype,path
            a,b=a.cpu(),b.cpu()
            if a.is_floating_point():
                delta=float((a-b).abs().max()) if a.numel() else 0.
                max_errors[path]=delta
                assert torch.isfinite(a).all() and torch.isfinite(b).all(),path
            else:assert torch.equal(a,b),path
        elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
        elif isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:
                if k in ('anchor_wall_seconds_total','wall_seconds'):continue
                diagnostic(a[k],b[k],path+'.'+str(k))
        elif isinstance(a,(tuple,list)):
            assert type(a)==type(b) and len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):diagnostic(x,y,f'{path}.{i}')
        else:assert a==b,path
    def native_renders(state,p,folder):
        from local_postshot.training import render_parameters
        from local_postshot.studio_bridge import gpu_session
        folder.mkdir(exist_ok=False)
        result=[]
        recipe=state['training_recipe'] or {}
        degree=min(p['settings']['sh_degree'],state['step']//p['settings']['sh_degree_interval'])
        with gpu_session(),torch.no_grad():
            parameters={k:v.cuda() for k,v in state['parameters'].items()}
            for i,camera in enumerate(p['reconstruction']['cameras']):
                rgb,_,_=render_parameters(parameters,camera,degree,projection_policy=recipe.get('projection'))
                pixels=rgb[0].cpu().numpy()
                np.save(folder/f'{i:04d}.npy',pixels,allow_pickle=False)
                result.append(pixels)
        return result
    def render_variance(reference,repeats,resumed):
        rows=[]
        def delta(a,b):
            d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
            return {'max_abs':float(np.abs(d).max()),'rms':float(np.sqrt(np.mean(d*d)))}
        for i,base in enumerate(reference):
            noise=[delta(base,r[i]) for r in repeats]
            noise.append(delta(repeats[0][i],repeats[1][i]))
            observed={k:max(n[k] for n in noise) for k in ('max_abs','rms')}
            error=delta(base,resumed[i])
            # Predeclared rendering criterion. No per-tensor tolerance changes:
            # exact restore is tested BEFORE the next optimizer update.
            limits={'max_abs':min(1/255,3*observed['max_abs']+1e-6),
                    'rms':min(1e-4,3*observed['rms']+1e-7)}
            row={'camera':i,'repeat_continuous':noise,'variance_envelope':observed,
                 'resume_vs_continuous':error,'limits':limits,
                 'passed':all(error[k]<=limits[k] for k in error)}
            rows.append(row)
        return rows
    try:
        report['selection']=run_cases(root/'selection')
        modes=('bilateral','panorama','off') if args.mode=='all' else (args.mode,)
        for mode in modes:
            kw={'steps':args.steps,'recipe':args.recipe,'photo_comp':mode!='off',
                'use_bilateral_grid':mode=='bilateral','use_sparse_depth':True,
                'preview_every':10,'settings':settings,'emit':lambda *a:None}
            continuous=root/(mode+'-continuous')
            interrupted=root/(mode+'-interrupted')
            run(package,continuous,**kw)
            repeats=[]
            for repeat in range(2):
                folder=root/f'{mode}-repeat-{repeat+1}'
                run(package,folder,**kw)
                repeats.append(folder)
            errors=[]
            receipts={}
            def controls():
                try:
                    state=wait_for(lambda:(s if (s:=read_control_state(interrupted)) and s['status']=='running' else None),'running')
                    sha=state['package_sha256']
                    submit_control(interrupted,sha,'suspend')
                    suspended=wait_for(lambda:(s if (s:=read_control_state(interrupted)) and s['status']=='suspended' else None),'suspended')
                    step=suspended['step']
                    submit_control(interrupted,sha,'checkpoint')
                    wait_for(lambda:read_control_state(interrupted).get('message')=='checkpoint_saved','checkpoint')
                    request={'schema_version':1,'package_sha256':sha,'generation_id':suspended['generation_id'],
                             'request_id':'a'*32,'image':'images/00000004.png'}
                    project.atomic_json(interrupted/'preview-request.json',request)
                    preview=wait_for(lambda:(v if (v:=read(interrupted/'preview.json')).get('request_id')==request['request_id'] else None),'suspended heldout preview')
                    assert preview['step']==step and preview['split']=='validation'
                    for key in ('source','render'):
                        assert project.digest(interrupted/preview[key+'_file'])==preview[key+'_sha256']
                    receipts.update(suspended_step=step,preview=preview)
                    submit_control(interrupted,sha,'unpause')
                    receipts['unpaused']=True
                except BaseException as error:errors.append(error)
            monitor=threading.Thread(target=controls,daemon=True)
            monitor.start()
            def pause_boundary(params,context):
                if context['step']>=args.steps//3 and receipts.get('unpaused') and not receipts.get('pause_requested'):
                    wait_for(lambda:read_control_state(interrupted).get('status')=='running','unpause acknowledgement')
                    submit_control(interrupted,project.load(interrupted)['studio_dataset']['package_sha256'],'pause')
                    receipts['pause_requested']=True
                    # Allow the documented IO poll interval at this synthetic
                    # control boundary. This is not a training performance test.
                    time.sleep(.16)
            paused=run(package,interrupted,boundary_observer=pause_boundary,**kw)
            monitor.join(5)
            if errors:raise errors[0]
            assert not monitor.is_alive() and paused['status']=='paused'
            assert 0<paused['completed_steps']<args.steps
            assert paused['checkpoint_sha256']==project.digest(interrupted/'checkpoints/latest.pt')
            # Old generation controls cannot stop the resumed attempt.
            stale=read(interrupted/'control-request.json')
            stale['request_id']='b'*32
            project.atomic_json(interrupted/'control-request.json',stale)
            boundary=load_checkpoint(interrupted/'checkpoints/latest.pt',map_location='cpu')
            import shutil
            shutil.copyfile(interrupted/'checkpoints/latest.pt',interrupted/'restore-input.pt')
            restored=[]
            def restore_boundary(params,context):
                if restored:return
                assert context['step']==boundary['step']
                current=context['snapshot']()
                assert_restored(boundary,current)
                # Negative controls prove missing optimizer/aux restoration is
                # caught even when rendering might barely change in40 steps.
                import copy
                for key in ('optimizers','studio_auxiliary'):
                    broken=copy.deepcopy(current);broken[key]={}
                    try:assert_restored(boundary,broken)
                    except AssertionError:pass
                    else:raise AssertionError('Restore audit missed '+key)
                restored.append({'step':context['step'],'all_state_bit_exact':True,
                                 'negative_controls':['optimizers','studio_auxiliary']})
                project.atomic_json(interrupted/'restore-boundary.json',restored[0])
            completed=run(package,interrupted,resume=True,boundary_observer=restore_boundary,**kw)
            assert restored,'Missing pre-update restoration audit'
            assert completed['status']=='succeeded' and completed['completed_steps']==args.steps
            a,b=(load_checkpoint(folder/'checkpoints/latest.pt',map_location='cpu') for folder in (continuous,interrupted))
            for key in ('parameters','optimizers','schedulers','strategy_state','strategy_config','sampler',
                        'rng_python','rng_numpy','rng_torch','rng_cuda','dataset_key','scene_scale','training_recipe','config'):
                diagnostic(a[key],b[key],mode+'.'+key)
            identities=[dict(state['studio_experiment']) for state in (a,b)]
            for identity in identities:identity.pop('checkpoint')
            diagnostic(*identities,mode+'.experiment_identity_except_output_path')
            for key in ('photometric','optimizer'):
                diagnostic(a['studio_auxiliary'][key],b['studio_auxiliary'][key],mode+'.auxiliary.'+key)
            prepared=project.load(continuous)
            reference=native_renders(a,prepared,continuous/'acceptance-renders')
            repeat_renders=[]
            for folder in repeats:
                state=load_checkpoint(folder/'checkpoints/latest.pt',map_location='cpu')
                diagnostic(a['parameters'],state['parameters'],mode+'.repeat.'+folder.name)
                repeat_renders.append(native_renders(state,prepared,folder/'acceptance-renders'))
            resumed=native_renders(b,prepared,interrupted/'acceptance-renders')
            variance=render_variance(reference,repeat_renders,resumed)
            case={'mode':mode,'paused_steps':paused['completed_steps'],'completed':completed,
                  'controls':receipts,'restore_boundary':restored,'render_variance':variance}
            report['cases'].append(case)
            project.atomic_json(root/'validation.json',report)
            assert all(row['passed'] for row in variance),f'{mode} resumed render exceeds measured repeat variance'
        p=project.load(interrupted)
        model=next(m for m in p['models'] if m.get('role')=='training')
        camera=p['reconstruction']['cameras'][0]
        def edit(action,**kw):return dispatch({'path':p['path'],'model_id':model['id'],'action':action,'args':kw})
        selected=edit('select_rect',camera=camera,rectangle=[0,0,camera['width'],camera['height']],xray=False)
        assert selected['selection_count']>0 and selected.get('native_render')
        precision=render(p['path'],camera,str(root/'reference-render.png'))
        np.testing.assert_array_equal(np.asarray(Image.open(selected['native_render'])),np.asarray(Image.open(precision['path'])))
        cloud=model_cloud(p,model)
        xy,z=project_points(cloud['means'],camera)
        cut=float(np.median(xy[:,0]))
        part=edit('select_rect',camera=camera,rectangle=[0,0,cut,camera['height']],xray=False)
        assert 0<part['selection_count']<len(cloud['means'])
        assert len(read_ply(part['selection_preview'])['means'])<=10000
        edit('delete')
        edit('preview',transform={'position':[.1,.2,-.1],'rotation':[.1,.2,.3],'scale':1.2})
        edit('cancel_preview')
        edit('preview',transform={'position':[.1,.2,-.1],'rotation':[.1,.2,.3],'scale':1.2})
        edit('commit');edit('undo');edit('redo')
        reopened=project.load(p['path']);m=next(m for m in reopened['models'] if m['id']==model['id'])
        baked=model_cloud(reopened,m)
        write_ply(root/'edited.ply',baked);write_spz(root/'edited.spz',baked)
        for k,v in read_ply(root/'edited.ply').items():np.testing.assert_allclose(v,baked[k],atol=2e-7)
        decoded=read_spz(root/'edited.spz')
        assert len(decoded['means'])==len(baked['means'])
        np.testing.assert_allclose(decoded['means'],baked['means'],atol=1e-3)
        exact=edit('render',camera=camera)
        rendered=render(p['path'],camera,str(root/'edited-render.png'))
        np.testing.assert_array_equal(np.asarray(Image.open(exact['native_render'])),np.asarray(Image.open(rendered['path'])))
        report.update(status='passed',equivalence_max_abs=max_errors,editor={'count':len(baked['means']),'ply':str(root/'edited.ply'),'spz':str(root/'edited.spz')})
    except BaseException:
        report.update(status='failed',error=traceback.format_exc(),equivalence_max_abs=max_errors)
        raise
    finally:
        project.atomic_json(root/'validation.json',report)
    print(json.dumps({'status':'passed','report':str(root/'validation.json')}))
    return 0


if __name__=='__main__':
    sys.exit(main())

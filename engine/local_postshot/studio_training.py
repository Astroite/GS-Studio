"""GS experiment CLI over the existing independent core train() loop.

python -m local_postshot.studio_training --dataset PACKAGE --output EXPERIMENT
    [--steps N] [--resume] [--no-photo-comp] [--no-use-bilateral-grid]
    [--no-use-sparse-depth] [--preview-every N] [--restart-from-step-zero]
"""
import argparse
import contextlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np
from PIL import Image
from . import project
from .locks import operation_lock
from .studio_dataset import prepare, validate_project
from .studio_control import TrainingControl


def engine_identity():
    # Freeze code/artifacts relevant to this experiment, independent of mtimes.
    root = Path(__file__).parent
    names = ['training.py','training_recipe.py','training_performance.py','training_state_performance.py',
             'projection_policy.py','geometry.py','formats.py','splat3_strategy.py','splat3_regularization.py',
             'splat3_renderer.py','splat3_stats.py','splat3_projection.py','splat3_tiles.py','splat3_aa.py',
             'studio_dataset.py','studio_auxiliary.py','studio_training.py','studio_preview.py','studio_control.py']
    versions = {}
    for name in ('torch','gsplat','numpy','fused-ssim','local-splat3-render','local-splat3-stats',
                 'local-splat3-projection','local-splat3-tiles'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'name':'local-postshot-independent-gsstudio','schema_version':1,
            'sources':{n:project.digest(root/n) for n in names},'versions':versions}


def validate_checkpoint(state,config,dataset_key,settings,train_indices=None):
    required = {'parameters','optimizers','schedulers','strategy_state','strategy_config','sampler',
                'rng_python','rng_numpy','rng_torch','rng_cuda','config','dataset_key','scene_scale',
                'training_recipe','studio_experiment','studio_auxiliary','step'}
    if not isinstance(state,dict) or not required<=state.keys() or state.get('schema')!=1:
        raise ValueError('Incomplete independent core checkpoint; old GS checkpoints are unsupported')
    if (state['studio_experiment']!=config or state['dataset_key']!=dataset_key or
            state['config']!=settings or not isinstance(state['step'],int) or not 0<=state['step']<=config['steps']):
        raise ValueError('Checkpoint experiment/config/dataset identity mismatch')
    auxiliary = state['studio_auxiliary']
    if not isinstance(auxiliary,dict) or auxiliary.get('identity')!=config or auxiliary.get('schema_version')!=1:
        raise ValueError('Checkpoint auxiliary identity mismatch')
    expected = config['photo_comp']
    if (auxiliary.get('photometric') is not None)!=expected or (auxiliary.get('optimizer') is not None)!=expected:
        raise ValueError('Checkpoint missing full auxiliary parameters/optimizer')
    if set(state['parameters'])!={'means','scales','quats','opacities','sh0','shN'} or set(state['optimizers'])!=set(state['parameters']):
        raise ValueError('Checkpoint Gaussian parameters/optimizers incomplete')
    import torch
    if any(not torch.isfinite(v).all() for v in state['parameters'].values()):
        raise ValueError('Nonfinite checkpoint Gaussian state')
    def finite_tree(value):
        if isinstance(value,torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value,dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value,(tuple,list)):
            return all(finite_tree(v) for v in value)
        return True
    if not all(finite_tree(state[k]) for k in ('optimizers','strategy_state','studio_auxiliary')):
        raise ValueError('Nonfinite checkpoint optimizer/auxiliary/strategy state')
    sampler = state['sampler']
    if (not isinstance(sampler,dict) or not {'indices','rng','order','position'}<=sampler.keys() or
            (train_indices is not None and sampler['indices']!=train_indices) or
            len(set(sampler['indices']))!=len(sampler['indices']) or not sampler['indices'] or
            not isinstance(sampler['position'],int) or not 0<=sampler['position']<=len(sampler['order']) or
            (sampler['order'] and sorted(sampler['order'])!=sorted(sampler['indices']))):
        raise ValueError('Checkpoint sampler identity/state mismatch')
    return state


def evaluate_project(p,emit=None,control=None):
    import torch
    from .training import tensors_from_cloud,render_parameters,ssim
    from .formats import load_cloud
    from .studio_auxiliary import masked_statistics
    meta = validate_project(p)
    emit = emit or (lambda *a:None)
    root = Path(p['path'])
    output = root/'evaluation'
    output.mkdir(exist_ok=True)
    # Evaluation is of the verified raw export; no heldout photometric fitting.
    cloud = load_cloud(root/'model.ply')
    params = tensors_from_cloud(cloud)
    images = []
    indices = p['studio_dataset']['test_indices']
    for number,index in enumerate(indices):
        if control is not None and control.is_set():
            raise InterruptedError('Evaluation cancelled')
        camera = p['reconstruction']['cameras'][index]
        with Image.open(Path(p['studio_dataset']['package'])/camera['image']) as im:
            truth = torch.as_tensor(np.asarray(im.convert('RGB')).copy(),device='cuda',dtype=torch.float32)[None]/255
        with torch.no_grad():
            prediction,_,_ = render_parameters(params,camera,p['settings']['sh_degree'],projection_policy=(p.get('training',{}).get('recipe') or {}).get('projection'))
            if camera['has_mask']:
                with Image.open(Path(p['studio_dataset']['package'])/camera['mask']) as im:
                    keep = torch.as_tensor(np.asarray(im).copy()==255,device='cuda')[None]
                metric = masked_statistics(prediction,truth,keep,kind='uniform11',padding='same')
                mse,score = metric['mse'],metric['ssim']
            else:
                mse,score = (prediction-truth).square().mean(),ssim(prediction,truth)
            row = {'image':meta['images'][index]['image'],'index':index,
                   'psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),'ssim':float(score)}
        file = output/f'{index:08d}-render.png'
        Image.fromarray((prediction[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(file)
        row.update(render_file=file.name,render_sha256=project.digest(file))
        images.append(row)
        emit('progress',{'stage':'evaluation','progress':(number+1)/len(indices),'message':row['image']})
    result = {'schema_version':1,'package_sha256':p['studio_dataset']['package_sha256'],
              'model_sha256':project.digest(root/'model.ply'),'images':images,
              'averages':{k:float(np.mean([r[k] for r in images])) for k in ('psnr','ssim')},
              'metric':'masked sRGB PSNR; full-support uniform11 SSIM; black background; no heldout fitting'}
    project.atomic_json(output/'metrics.json',result)
    return result


def run(dataset,output,steps=None,resume=False,photo_comp=True,use_bilateral_grid=True,use_sparse_depth=True,
        preview_every=0,restart_from_step_zero=False,recipe='quality_v2',emit=None,settings=None,
        external_control=None,external_pause=None,boundary_observer=None):
    from .studio_dataset import validate_package
    dataset,output = Path(dataset).resolve(),Path(output).resolve()
    if output==dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError('Experiment must be outside immutable package')
    if resume and restart_from_step_zero:
        raise ValueError('Resume and step-zero restart are mutually exclusive')
    meta = validate_package(dataset)
    steps = int(steps if steps is not None else max(30000,30*sum(r['split']=='train' for r in meta['images'])))
    if steps<1 or preview_every<0:
        raise ValueError('Positive steps/nonnegative preview interval required')
    output.mkdir(parents=True,exist_ok=True)
    emit = emit or (lambda e,d: print(json.dumps({'type':'event','event':e,'data':d},ensure_ascii=False,allow_nan=False),flush=True))
    with operation_lock(output,'studio_training'):
        if (output/'checkpoint.pt').exists():
            raise ValueError('Old GS checkpoints are unsupported; use a new experiment')
        if not (output/'config.json').is_file() and any((output/n).exists() for n in ('model.ply','checkpoints/latest.pt')):
            raise ValueError('Unrecognized experiment results; existing files will not be overwritten')
        depth_count = meta['_sparse_depth_observations']
        cfg = {'schema_version':1,'package_sha256':meta['_package_sha256'],'steps':steps,
               'photo_comp':bool(photo_comp),'use_bilateral_grid':bool(use_bilateral_grid),
               'use_sparse_depth':bool(use_sparse_depth and depth_count),'preview_every':int(preview_every),
               'sparse_depth_requested':bool(use_sparse_depth),'sparse_depth_available':depth_count>0,
               'sparse_depth_observations':depth_count,
               'photometric_model':'disabled' if not photo_comp else 'bilateral_grid' if use_bilateral_grid else 'panorama_exposure',
               'sparse_depth_status':'disabled' if not use_sparse_depth else 'enabled' if depth_count else 'unavailable',
               'sparse_depth_weight':.1,'training_recipe':recipe,'engine':engine_identity(),
               'checkpoint':str(output/'checkpoints'/'latest.pt')}
        settings = {**(settings or {}),'steps':steps,'training_recipe':recipe}
        if preview_every:
            settings['preview_every'] = preview_every
        p = prepare(dataset,output,settings)
        config_file = output/'config.json'
        previous = json.loads(config_file.read_text(encoding='utf-8')) if config_file.is_file() else None
        checkpoint = output/'checkpoints'/'latest.pt'
        summary_file = output/'training.json'
        prior_summary = json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.is_file() else {}
        if previous is not None and previous!=cfg:
            raise ValueError('Experiment config/package/engine identity changed')
        if prior_summary.get('status')=='succeeded':
            raise ValueError('Experiment already succeeded; create a new experiment')
        if resume:
            if previous is None or not checkpoint.is_file():
                raise ValueError('No independent checkpoint/config available to resume')
            from .training import load_checkpoint
            restored = validate_checkpoint(load_checkpoint(checkpoint,map_location='cpu'),cfg,p['studio_dataset']['identity'],p['settings'],p['studio_dataset']['train_indices'])
            # A crash may leave an atomically newer checkpoint than the summary;
            # an older/same-step hash mismatch is never silently accepted.
            if (prior_summary.get('checkpoint_sha256') and
                    project.digest(checkpoint)!=prior_summary['checkpoint_sha256'] and
                    restored['step']<=prior_summary.get('completed_steps',0)):
                raise ValueError('Checkpoint hash differs from last durable experiment record')
            del restored
        elif restart_from_step_zero:
            if checkpoint.exists() or (output/'model.ply').exists() or prior_summary.get('completed_steps',0)>0:
                raise ValueError('Step-zero restart conflicts with checkpoint/result evidence')
        elif previous is not None:
            raise ValueError('Existing experiment requires explicit checkpoint resume or verified step-zero restart')
        p['studio_training'] = cfg
        project.atomic_json(config_file,cfg)
        project.save(p)
        controller = TrainingControl(output,cfg['package_sha256'],waiting_for_gpu=True)
        completed = int(prior_summary.get('completed_steps',0)) if resume else 0
        checkpoint_digest = project.digest(checkpoint) if checkpoint.is_file() else None
        last_record = float('-inf')
        def record(status,*,force=True,refresh_checkpoint=False,**extra):
            nonlocal checkpoint_digest,last_record
            now = time.monotonic()
            if not force and now-last_record < .15:
                return
            if refresh_checkpoint:
                checkpoint_digest = project.digest(checkpoint) if checkpoint.is_file() else None
            value = {**cfg,'status':status,'steps':steps,'completed_steps':completed,'target':steps,
                     'checkpoint':str(checkpoint) if checkpoint_digest else None,
                     'checkpoint_sha256':checkpoint_digest,
                     'generation_id':controller.generation_id,**extra}
            project.atomic_json(summary_file,value)
            last_record = now
            return value
        record('waiting_gpu')
        controller.on_event = lambda:record(controller.status)
        try:
            from .studio_bridge import gpu_session
            from .studio_preview import Preview
            from .training import train
            previews = Preview(p,controller.generation_id,emit)
            last_validation = [time.monotonic()]
            def observer(params,context,save):
                nonlocal completed
                completed = context['step']
                if boundary_observer is not None:
                    boundary_observer(params,context)
                # Core state and experiment records have separate responsibilities.
                p.update(context['project'])
                if time.monotonic()-last_validation[0]>=10 or completed==steps:
                    validate_project(p)
                    last_validation[0] = time.monotonic()
                def tick():
                    previews.tick(params,completed,steps)
                    if controller.status=='suspended':
                        record('suspended',force=False)
                    if external_control is not None and external_control.is_set():
                        return 'pause' if external_pause is not None and external_pause.is_set() else 'stop'
                tick()
                if completed==steps:
                    previews.tick(params,completed,steps,force=True)
                if external_control is not None and external_control.is_set():
                    save()
                    return 'pause' if external_pause is not None and external_pause.is_set() else 'stop'
                revision = controller.revision
                action = controller.observe(completed,save,target=steps,tick=tick)
                record('paused' if action=='pause' else 'stopped' if action=='stop' else controller.status,
                       force=controller.revision!=revision)
                return action
            def checkpoint_written(step):
                nonlocal completed
                completed = step
                record(controller.status,refresh_checkpoint=True)
            class WaitControl:
                def is_set(self):
                    if external_control is not None and external_control.is_set():
                        controller.status = 'stopping'
                        return True
                    return controller.is_set()
            with gpu_session(cancel=WaitControl()):
                controller.start_running()
                record('running')
                p = train(output,emit,threading.Event(),threading.Event(),resume=resume,observer=observer,
                          checkpoint_observer=checkpoint_written)
                completed = p['training']['step']
                source = next(m for m in p['models'] if m.get('role')=='training')
                destination = output/'model.ply'
                temp = output/'model.ply.tmp'
                expected = project.digest(source['path'])
                shutil.copyfile(source['path'],temp)
                if project.digest(temp)!=expected:
                    raise ValueError('Export copy hash verification failed')
                temp.replace(destination)
                if project.digest(destination)!=expected:
                    raise ValueError('Export hash verification failed')
                if p['status'] in ('paused','stopped','cancelled'):
                    status = 'paused' if p['status']=='paused' else 'stopped'
                    controller.on_event = None
                    controller.complete(status,completed)
                    return record(status,refresh_checkpoint=True,model_sha256=expected,evaluation_status='not_run')
                if completed!=steps or p['status']!='trained':
                    raise RuntimeError('Core training did not reach requested target')
                record('evaluating',refresh_checkpoint=True,model_sha256=expected,evaluation_status='running')
                class EvaluationControl:
                    def is_set(self):
                        return controller.observe(completed,lambda:None,target=steps) in ('pause','stop')
                evaluate_project(p,emit,EvaluationControl())
                # Revalidate package and export after evaluation before success.
                validate_project(p)
                if project.digest(destination)!=expected:
                    raise ValueError('Model changed during evaluation')
                result = record('succeeded',model_sha256=expected,evaluation_status='completed',
                                evaluation_sha256=project.digest(output/'evaluation'/'metrics.json'))
                controller.on_event = None
                controller.complete('succeeded',completed)
                return result
        except BaseException as error:
            waiting_cancelled = controller.status=='stopping' and completed==0 and not checkpoint.exists()
            controlled = controller.status in ('paused','stopping')
            status = 'paused' if controller.status=='paused' else 'stopped' if controlled else 'failed'
            result = record(status,refresh_checkpoint=True,reason='gpu_wait_cancelled' if waiting_cancelled else str(error),
                            evaluation_status='not_completed',recoverable=checkpoint.is_file())
            if (output/'model.ply').is_file():
                result['model_sha256'] = project.digest(output/'model.ply')
                project.atomic_json(summary_file,result)
            if not controlled:
                project.atomic_json(output/'failure.json',result)
            controller.on_event = None
            controller.complete(status,completed)
            if controlled:
                return result
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--no-photo-comp',dest='photo_comp',action='store_false')
    parser.add_argument('--no-use-bilateral-grid',dest='use_bilateral_grid',action='store_false')
    parser.add_argument('--no-use-sparse-depth',dest='use_sparse_depth',action='store_false')
    parser.add_argument('--preview-every',type=int,default=0)
    parser.add_argument('--restart-from-step-zero',action='store_true')
    parser.add_argument('--recipe',choices=['quality_v2','observed_v5','observed_v6'],default='quality_v2')
    args = vars(parser.parse_args(argv))
    try:
        protocol = sys.stdout
        def emit(event,data):
            print(json.dumps({'type':'event','event':event,'data':data},ensure_ascii=False,allow_nan=False),file=protocol,flush=True)
        with contextlib.redirect_stdout(sys.stderr):
            result = run(**args,emit=emit)
        print(json.dumps({'type':'result','result':result},ensure_ascii=False,allow_nan=False),flush=True)
        return 0
    except Exception as error:
        print(json.dumps({'type':'error','error':{'code':type(error).__name__,'message':str(error)}},ensure_ascii=False),flush=True)
        return 1


if __name__=='__main__':
    sys.exit(main())

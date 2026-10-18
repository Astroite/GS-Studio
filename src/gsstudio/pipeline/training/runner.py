"""Backend dispatch; the dataset and validation cameras are backend independent."""
from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.runtime.gpu_lock import gpu_session
from gsstudio.infrastructure.adapters.gsplat import native_training_command
from gsstudio.infrastructure.runtime.processes import run_logged
from gsstudio.pipeline.training.data import json_write, validate_package
from gsstudio.pipeline.training.presets import prepare_training, depth_availability


def _verify_attempt_history(output: Path, history: object) -> list[dict]:
    if not isinstance(history, list):
        raise RuntimeError('Experiment attempt history is invalid')
    for attempt in history:
        if not isinstance(attempt, dict):
            raise RuntimeError('Experiment attempt history is invalid')
        for kind in ('dispatch', 'log', 'training', 'failure'):
            evidence = attempt.get(kind)
            if evidence is None:
                continue
            if not isinstance(evidence, dict) or not isinstance(evidence.get('path'), str):
                raise RuntimeError('Experiment attempt evidence is invalid')
            name = evidence['path']
            if Path(name).name != name or name in {'.', '..'}:
                raise RuntimeError('Experiment attempt evidence path is unsafe')
            path = output / name
            if (not path.is_file() or path.stat().st_size != evidence.get('bytes') or
                    sha256_file(path) != evidence.get('sha256')):
                raise RuntimeError(f'Experiment attempt evidence changed: {name}')
    return history


def _archive_attempt(output: Path, previous: dict) -> list[dict]:
    history = list(_verify_attempt_history(output, previous.get('attempt_history', [])))
    index = 1
    while any((output / f'{name}-attempt-{index:03d}.{suffix}').exists()
              for name, suffix in (('dispatch', 'json'), ('train', 'log'),
                                   ('training', 'json'), ('failure', 'json'))):
        index += 1
    entry: dict = {'number': index, 'status': previous.get('status')}
    for kind, source, name in (
        ('dispatch', output / 'dispatch.json', f'dispatch-attempt-{index:03d}.json'),
        ('log', output / 'train.log', f'train-attempt-{index:03d}.log'),
        ('training', output / 'training.json', f'training-attempt-{index:03d}.json'),
        ('failure', output / 'failure.json', f'failure-attempt-{index:03d}.json'),
    ):
        if not source.is_file():
            continue
        destination = output / name
        temporary = output / f'.{name}.{uuid.uuid4().hex}.tmp'
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        entry[kind] = {'path': name, 'bytes': destination.stat().st_size,
                       'sha256': sha256_file(destination)}
    history.append(entry)
    return history


def verify_evaluation(output: Path, meta: dict) -> str:
    path = output / 'evaluation' / 'metrics.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    expected = sorted(row['image'] for row in meta['images'] if row['split'] == 'validation')
    rows = report.get('images') if isinstance(report, dict) else None
    if (not isinstance(report, dict) or report.get('schema_version') != 1 or
            not isinstance(rows, list) or len(rows) != len(expected) or
            any(not isinstance(row, dict) or not isinstance(row.get('image'), str) for row in rows) or
            sorted(row['image'] for row in rows) != expected or
            any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k])
                for row in rows for k in ('psnr','ssim'))):
        raise RuntimeError('Training evaluation evidence is incomplete or mismatched')
    return sha256_file(path)


ENGINE_ID = "local_postshot.studio_training"
CHECKPOINT = "checkpoints/latest.pt"


def require_backend(backend):
    if backend == 'postshot':
        raise ValueError('Postshot backend is retired. Use --backend gsplat with the independent GS Studio engine; create a new experiment.')
    if backend != 'gsplat':
        raise ValueError('backend must be gsplat (independent GS Studio engine)')


def _verify_identity(output, record, training, *, require_checkpoint=True):
    config = json.loads((output/'config.json').read_text(encoding='utf-8'))
    for key, expected in (('package_sha256',record['package_sha256']),
                          ('steps',record['requested_steps']),('photo_comp',record['photo_comp']),
                          ('use_bilateral_grid',record['use_bilateral_grid']),
                          ('sparse_depth_requested',record['sparse_depth_requested']),
                          ('use_sparse_depth',record['use_sparse_depth']),
                          ('sparse_depth_available',record['sparse_depth_available']),
                          ('sparse_depth_observations',record['sparse_depth_observations']),
                          ('sparse_depth_status',record['sparse_depth_status'])):
        if any(doc.get(key) != expected or type(doc.get(key)) is not type(expected)
               for doc in (config, training)):
            raise RuntimeError(f'Independent training identity differs: {key}')
    command = record['command']
    package = Path(command[command.index('--dataset')+1])
    actual = depth_availability(package)
    if (actual['package_sha256'] != record['package_sha256'] or
            actual['available'] != record['sparse_depth_available'] or
            actual['observations'] != record['sparse_depth_observations'] or
            record['use_sparse_depth'] != (record['sparse_depth_requested'] and actual['available'])):
        raise RuntimeError('Independent sparse-depth effective state differs from validated anchors')
    recipe=record.get('training_recipe')
    if recipe and (config.get('training_recipe')!=recipe or training.get('training_recipe')!=recipe):
        raise RuntimeError('Independent training recipe identity differs')
    engine = config.get('engine')
    if (not isinstance(engine,dict) or engine.get('name')!='local-postshot-independent-gsstudio'
            or engine.get('schema_version')!=1 or not engine.get('sources') or not engine.get('versions')
            or training.get('engine')!=engine):
        raise RuntimeError('Retired or unknown training engine identity; create a new experiment')
    checkpoint = output/CHECKPOINT
    if not require_checkpoint:
        if (checkpoint.exists() or training.get('checkpoint') is not None or
                not isinstance(config.get('checkpoint'),str) or
                (output/config['checkpoint']).resolve()!=checkpoint.resolve()):
            raise RuntimeError('Cancelled GPU wait checkpoint identity conflicts with evidence')
        return
    if (not checkpoint.is_file() or
            any(not isinstance(doc.get('checkpoint'),str) or
                (output/doc['checkpoint']).resolve()!=checkpoint.resolve() for doc in (config,training))):
        raise RuntimeError('Missing current checkpoints/latest.pt identity; old checkpoints are unsupported')
    if training.get('checkpoint_sha256') != sha256_file(checkpoint):
        raise RuntimeError('Checkpoint hash differs from training evidence')


def verify_completed(output, meta, record):
    if record.get('engine')!=ENGINE_ID or record.get('checkpoint_format')!=CHECKPOINT:
        raise RuntimeError('Retired experiment identity cannot be adopted')
    training=json.loads((output/'training.json').read_text(encoding='utf-8'))
    _verify_identity(output,record,training)
    if (record.get('status')!='succeeded' or training.get('status')!='succeeded'
            or training.get('completed_steps')!=record['requested_steps']
            or training.get('evaluation_status')!='completed'
            or verify_evaluation(output,meta)!=record.get('evaluation_sha256')
            or training.get('model_sha256')!=record.get('model_sha256')
            or sha256_file(output/'model.ply')!=record.get('model_sha256')):
        raise RuntimeError('Completed independent experiment evidence differs from dispatch')
    return training


def verify_dispatch_settings(record, package, **options):
    """Crash adoption uses the same defaults/derived identity as a new dispatch."""
    package, meta, settings = prepare_training(package, **options)
    expected = {**{k:v for k,v in settings.items() if k!='steps'},
                'requested_steps':settings['steps'], 'backend':'gsplat',
                'engine':ENGINE_ID, 'checkpoint_format':CHECKPOINT}
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f'Completed training configuration identity differs: {key}')
    return meta


def train_package(package: Path, output: Path, backend='gsplat', steps=None,
                  photo_comp=None, resume=False, dry_run=False,
                  use_bilateral_grid=None, use_sparse_depth=None,
                  preview_every=0, training_recipe='quality_v2', sh_degree=3,
                  restart_from_step_zero=False, training_preset='auto', source_kind=None):
    require_backend(backend)
    if restart_from_step_zero and not resume:
        raise ValueError('Step-zero restart requires an explicit resume of a failed experiment')
    package,output=package.resolve(),output.resolve()
    if package==output or package in output.parents or output in package.parents:
        raise ValueError('Training output must be separate from the immutable package')
    package,meta,settings=prepare_training(package, source_kind=source_kind,
        training_preset=training_preset,steps=steps,training_recipe=training_recipe,
        photo_comp=photo_comp,use_bilateral_grid=use_bilateral_grid,
        use_sparse_depth=use_sparse_depth,sh_degree=sh_degree)
    if package==output or package in output.parents or output in package.parents:
        raise ValueError('Training output must be separate from the derived immutable package')
    steps,recipe=settings['steps'],settings['training_recipe']
    photo_comp,use_bilateral_grid=settings['photo_comp'],settings['use_bilateral_grid']
    requested_depth=settings['sparse_depth_requested']
    if type(preview_every)!=int or preview_every<0:
        raise ValueError('preview_every must be nonnegative')
    previous=None
    if output.exists():
        if not (output/'dispatch.json').is_file():
            raise FileExistsError(f'Unrecognized experiment directory: {output}')
        previous=json.loads((output/'dispatch.json').read_text(encoding='utf-8'))
        if previous.get('engine')!=ENGINE_ID or previous.get('checkpoint_format')!=CHECKPOINT:
            raise RuntimeError('Retired experiment/checkpoint identity format; create a new experiment')
        if not resume and previous.get('status')!='prepared':
            raise FileExistsError(f'Experiment already executed: {output}')
    if resume and previous is None:
        raise FileNotFoundError(f'No experiment to resume: {output}')
    if resume and previous.get('status')=='succeeded':
        raise RuntimeError('A completed experiment cannot be resumed')
    if previous:
        expected={'backend':backend,'engine':ENGINE_ID,'checkpoint_format':CHECKPOINT,
                  'package_sha256':sha256_file(package/'dataset.json'),'requested_steps':steps,
                  'photo_comp':bool(photo_comp),'use_bilateral_grid':bool(use_bilateral_grid),
                  'use_sparse_depth':settings['use_sparse_depth'],'training_recipe':recipe,'sh_degree':3,
                  **{k:v for k,v in settings.items() if k!='steps'}}
        for key,value in expected.items():
            if previous.get(key)!=value:
                raise RuntimeError(f'Experiment identity changed: {key}')
    output.mkdir(parents=True,exist_ok=True)
    restart=False
    if resume and not (output/CHECKPOINT).is_file():
        summary_path=output/'training.json'
        summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
        if ((output/'checkpoint.pt').exists() or (output/'model.ply').exists() or
                summary.get('completed_steps',0)>0 or previous.get('completed_steps',0)>0):
            raise RuntimeError('Current checkpoint missing; old checkpoints cannot be resumed')
        restart=previous.get('status') in {'failed','prepared','stopped'}
        if not restart:
            raise RuntimeError('No recoverable checkpoint or verified step-zero restart')
        if not restart_from_step_zero:
            raise RuntimeError('No checkpoint exists; explicitly set restart_from_step_zero=true or --restart-from-step-zero after reviewing the failure')
    elif restart_from_step_zero:
        raise ValueError('Step-zero restart cannot replace an existing checkpoint')
    command=native_training_command('--dataset',str(package),'--output',str(output),'--steps',str(steps))
    command+=['--recipe',recipe]
    for enabled,flag in ((photo_comp,'--no-photo-comp'),(use_bilateral_grid,'--no-use-bilateral-grid'),
                         (requested_depth,'--no-use-sparse-depth')):
        if not enabled:
            command.append(flag)
    if restart:
        command.append('--restart-from-step-zero')
    elif resume:
        command.append('--resume')
    if preview_every:
        command+=['--preview-every',str(preview_every)]
    record=dict(backend=backend,engine=ENGINE_ID,checkpoint_format=CHECKPOINT,command=command,
                requested_steps=steps,status='prepared',
                preview_every=preview_every,**{k:v for k,v in settings.items() if k!='steps'})
    if restart:
        record['restart_from_step_zero']=True
    if previous:
        for key in ('backend','engine','checkpoint_format','package_sha256','requested_steps','photo_comp',
                    'use_bilateral_grid','use_sparse_depth','training_recipe','sh_degree',
                    'source_kind','training_preset','requested_training_preset',
                    'source_package_sha256','sparse_depth_requested','sparse_depth_available',
                    'sparse_depth_observations','sparse_depth_status','max_image_size','seed','max_splats'):
            if previous.get(key)!=record[key]:
                raise RuntimeError(f'Experiment identity changed: {key}')
    if dry_run:
        if not previous:
            json_write(output/'dispatch.json',record)
        return record
    if resume:
        record['attempt_history']=_archive_attempt(output,previous)
    json_write(output/'dispatch.json',record)
    try:
        from gsstudio.infrastructure.runtime.configuration import child_environment
        # Child owns ONE shared GPU lock, including direct CLI launches.
        resources=run_logged(command,output/'train.log',cwd=output,monitor_gpu=True,gpu_index=0,
                             env=child_environment(core=True))
        training=json.loads((output/'training.json').read_text(encoding='utf-8'))
        status=training.get('status')
        waiting_cancelled=status=='stopped' and training.get('reason')=='gpu_wait_cancelled'
        if waiting_cancelled and not (output/CHECKPOINT).is_file():
            if training.get('completed_steps',-1)!=0 or (output/'model.ply').exists():
                raise RuntimeError('GPU wait cancellation conflicts with training evidence')
            _verify_identity(output,record,training,require_checkpoint=False)
        else:
            _verify_identity(output,record,training)
        completed=training.get('completed_steps')
        if type(completed)!=int or not 0<=completed<=steps:
            raise RuntimeError('Invalid completed training budget')
        if status in {'paused','stopped'}:
            record.update(status=status,resources=resources,completed_steps=completed,
                          checkpoint=str(output/CHECKPOINT) if (output/CHECKPOINT).is_file() else None,
                          stop_reason=training.get('reason'))
        elif status=='succeeded':
            if completed!=steps or training.get('evaluation_status')!='completed':
                raise RuntimeError('Independent training or evaluation incomplete')
            evaluation_sha256=verify_evaluation(output,meta)
            if training.get('model_sha256')!=sha256_file(output/'model.ply'):
                raise RuntimeError('Training PLY hash differs from verified export')
            record.update(status='succeeded',resources=resources,completed_steps=completed,
                          model_sha256=training['model_sha256'],evaluation_sha256=evaluation_sha256,
                          checkpoint=str(output/CHECKPOINT))
        else:
            raise RuntimeError('Independent training did not publish a completed status')
    except Exception as error:
        record.update(status='failed',error=str(error))
        if getattr(error,'metrics',None):
            record['resources']=error.metrics
        json_write(output/'dispatch.json',record)
        raise
    json_write(output/'dispatch.json',record)
    return record

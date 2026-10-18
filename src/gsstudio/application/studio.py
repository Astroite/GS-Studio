"""Bounded desktop operations over the same authoritative headless manifests."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import uuid

from gsstudio.application import browse
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.persistence.manifests import canonical_hash, load_capture_manifest, load_model, save_yaml
from gsstudio.infrastructure.persistence.live_json import read_json
from gsstudio.infrastructure.persistence.run_repository import load_run
from gsstudio.infrastructure.paths import ensure_within, location_dir, scene_dir
from gsstudio.infrastructure.runtime.run_lock import run_session

READ_ACTIONS = frozenset({'capture_config','source_hashes','run_config','masks.review','masks.read',
    'qa_report','segment_choices','experiments','training_state','preview.read',
    'verified_models','model_open','list_external_imports','inspect_external_import','sources.list','experiment.read',
    'run_status'})
CONTROL_ACTIONS = frozenset({'control_training','preview.request'})


def catalog(root: Path) -> dict:
    root = root.resolve()
    locations = browse.list_locations(root)
    scenes, captures, runs = [], [], []
    for location in locations:
        if not location.readable:
            continue
        for scene in browse.list_scenes(root, location.id):
            scenes.append(scene.to_dict())
            if scene.readable:
                captures.extend(x.to_dict() for x in browse.list_captures(root, location.id, scene.id))
                runs.extend(x.to_dict() for x in browse.list_runs(root, location.id, scene.id))
    from gsstudio.application.external_imports import list_external_imports
    locations=[x.to_dict() for x in locations]
    for entries in (locations,scenes,captures,runs):
        for entry in entries:
            path=Path(entry['manifest_path'])
            if path.is_file():
                entry['manifest_sha256']=sha256_file(path)
    return dict(root=str(root), locations=locations, scenes=scenes,
                captures=captures, runs=runs, external_imports=list_external_imports(root)['imports'])


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path resolves outside the project root')
    ensure_within(path, root)
    return path


def _scene(r):
    return browse.require_scene(Path(r['root']), r['location_id'], r['scene_id'])


def _run(r):
    scene = _scene(r)
    run = load_run(scene, r['run_id'])
    return scene, run, _inside(scene/run.id, Path(r['root']))


def _capture(r):
    path = _scene(r)/'captures'/f"{r['capture_id']}.yaml"
    path = _inside(path, _scene(r)/'captures')
    return path, load_capture_manifest(path)


def _check_hash(path, expected):
    if not isinstance(expected,str) or sha256_file(path) != expected:
        raise RuntimeError(f'Stale identity; reload before editing: {path}')


def _unconsumed(scene, capture_id):
    for path in scene.glob('*/manifest.yaml'):
        run = load_run(scene,path.parent.name)
        if run.config.capture_id == capture_id:
            raise RuntimeError('Capture is consumed by a Run; create a new Capture for changed settings')


def _mask_dataset(r):
    from gsstudio.pipeline.masks.review import MaskReviewDataset
    _, _, work = _run(r)
    attempt = r.get('attempt','primary')
    if attempt not in {'primary','repair'}:
        raise ValueError('Attempt must be primary or repair')
    dataset = _inside(work/f'reconstruction-{attempt}',work)
    from gsstudio.pipeline.masks.masking import validate_mask_filter
    filtered = validate_mask_filter(dataset,verify_hashes=False)
    review = MaskReviewDataset(dataset,lazy_stats=True)
    review.expected_source_hashes = {row['image']:row['image_sha256'] for row in filtered['accepted']}
    identity = _mask_identity(review)
    return review, identity, work


def _mask_identity(review):
    return canonical_hash({'mask_filter_sha256':sha256_file(review.path/'mask-filter.json'),
                           'review_sha256':sha256_file(review.review_path) if review.review_path.is_file() else None,
                           'masks':[(x.output_mask,review.mask_sha256(x.image_id)) for x in review.items]})


def _mask_row(review, item):
    image, mask = review.media_path(item.image_id,'image'), review.media_path(item.image_id,'mask')
    image_sha256=sha256_file(image)
    if image_sha256!=review.expected_source_hashes[item.output_image]:
        raise RuntimeError('Accepted mask-filter source image hash changed')
    width,height,ignored=review._media_stats(image,mask)
    return dict(image_id=item.image_id, source_image=item.source_image, output_image=item.output_image,
                image_path=str(image), mask_path=str(mask), width=width,height=height,
                masked_fraction=ignored/(width*height), image_sha256=image_sha256,
                mask_sha256=sha256_file(mask),review=review.review_for(item.image_id))


def _mask_commit(r):
    import cv2
    import numpy as np
    from gsstudio.infrastructure.adapters.image_io import read_cv_image
    from gsstudio.pipeline.masks.masking import atomic_imwrite
    edits = r['edits']
    if not isinstance(edits,list) or not 1 <= len(edits) <= 100:
        raise ValueError('Provide 1..100 explicit review edits')
    _, _, work = _run(r)
    with run_session(work):
        review, identity, _ = _mask_dataset(r)
        if identity != r.get('dataset_identity'):
            raise RuntimeError('Mask dataset identity changed; reload review')
        if (review.path/'mask-final.json').exists():
            raise RuntimeError('Finalized masks are immutable; create a new Run')
        staged, seen = [], set()
        for edit in edits:
            image_id = edit['image_id']
            if image_id in seen:
                raise ValueError('Duplicate review image ID')
            seen.add(image_id)
            item = review.get_item(image_id)
            image, mask = review.media_path(image_id,'image'), review.media_path(image_id,'mask')
            _check_hash(image, edit.get('image_sha256'))
            if edit['image_sha256']!=review.expected_source_hashes[item.output_image]:
                raise RuntimeError('Source image differs from immutable mask-filter identity')
            _check_hash(mask, edit.get('mask_sha256'))
            from gsstudio.pipeline.masks.review import EDITABLE_REVIEW_STATUSES
            if edit.get('status') not in EDITABLE_REVIEW_STATUSES or len(edit.get('note','')) > 2000:
                raise ValueError('Explicit valid review status and bounded note required')
            polygons = edit.get('polygons',[])
            if not isinstance(polygons,list) or len(polygons)>100:
                raise ValueError('At most 100 polygons per image')
            pixels = read_cv_image(str(mask),cv2.IMREAD_GRAYSCALE).copy() if polygons else None
            for polygon in polygons:
                points = np.asarray(polygon['points'],dtype=float)
                if (points.ndim!=2 or points.shape[1]!=2 or not 3<=len(points)<=10000
                        or not np.isfinite(points).all() or (points<0).any()
                        or (points[:,0]>=item.width).any() or (points[:,1]>=item.height).any()
                        or type(polygon.get('ignore')) is not bool):
                    raise ValueError('Invalid native-pixel polygon')
                white_ignore = review.mask_semantics=='white_is_ignored'
                value = 255 if polygon['ignore']==white_ignore else 0
                cv2.fillPoly(pixels,[np.rint(points).astype(np.int32)],value)
            staged.append((edit,pixels))
        # Validate every stale identity before publishing any mask/review changes.
        updated=[]
        for edit,pixels in staged:
            image_id=edit['image_id']
            if pixels is not None:
                atomic_imwrite(review.media_path(image_id,'mask'),pixels)
            review.update_review(image_id,edit['status'],edit.get('note',''))
            updated.append(_mask_row(review,review.get_item(image_id)))
        return {'updated':updated,'dataset_identity':_mask_identity(review)}


def _experiment(r):
    output = _inside(r['output'],Path(r['root']))
    dispatch = read_json(output/'dispatch.json')
    if dispatch.get('backend') != 'gsplat' or not isinstance(dispatch.get('package_sha256'),str):
        raise RuntimeError('Not a current independent engine experiment')
    return output,dispatch


def _package_for_dispatch(dispatch):
    from gsstudio.pipeline.training.data import validate_package
    command=dispatch['command']
    package=Path(command[command.index('--dataset')+1])
    if sha256_file(package/'dataset.json')!=dispatch['package_sha256']:
        raise RuntimeError('Experiment package manifest identity changed')
    return package,validate_package(package)


def _verified_output(root, output, expected_digest=None):
    from gsstudio.application._shared import _verify_model
    from gsstudio.pipeline.training.runner import _verify_identity,verify_evaluation,ENGINE_ID
    output,dispatch=_experiment({'root':str(root),'output':str(output)})
    if dispatch.get('status')!='succeeded' or dispatch.get('engine')!=ENGINE_ID:
        raise RuntimeError('Experiment has no current verified successful dispatch')
    training=_optional_json(output/'training.json')
    _verify_identity(output,dispatch,training)
    _,meta=_package_for_dispatch(dispatch)
    if (training.get('status')!='succeeded' or training.get('evaluation_status')!='completed'
            or training.get('completed_steps')!=dispatch['requested_steps']
            or verify_evaluation(output,meta)!=dispatch.get('evaluation_sha256')
            or training.get('model_sha256')!=dispatch.get('model_sha256')
            or (expected_digest is not None and expected_digest!=dispatch.get('model_sha256'))):
        raise RuntimeError('Experiment success/metrics/model identity differs')
    _verify_model(output/'model.ply',dispatch['model_sha256'])
    try:
        project=read_json(output/'project.json')
    except FileNotFoundError as error:
        raise RuntimeError('Verified independent project state is missing') from error
    models=[row for row in project.get('models',[]) if row.get('role')=='training']
    if len(models)!=1 or project.get('studio_dataset',{}).get('package_sha256')!=dispatch['package_sha256']:
        raise RuntimeError('Independent project model/package identity differs')
    source=_inside(output/models[0]['path'],output)
    if sha256_file(source)!=dispatch['model_sha256']:
        raise RuntimeError('Independent project training source model identity differs')
    return output,dispatch


def _optional_json(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def execute_additional(r: dict, *, sink=None) -> tuple[bool, dict | None]:
    action = r['action']
    root = Path(r['root'])
    if action=='training_options':
        from gsstudio.application.training_options import training_options
        keys = ('import_id','location_id','scene_id','run_id','segment','duration_seconds',
                'training_preset','steps','training_recipe','photo_comp','use_bilateral_grid',
                'use_sparse_depth','sh_degree')
        return True, training_options(root, **{k:r[k] for k in keys if k in r})
    if action=='catalog.build':
        from gsstudio.infrastructure.persistence.catalog import build_catalog
        destination=_inside(root/'catalog'/'catalog.sqlite',root)
        return True,{'path':str(destination),'counts':build_catalog(root,destination)}
    if action=='run_status':
        from gsstudio.application.run_status import read_run_detail
        return True,read_run_detail(root,r['location_id'],r['scene_id'],r['run_id']).to_dict()
    if action=='run_stage':
        from gsstudio.application.runs import run_stage,resume_preprocess
        if r['stage']=='preprocess':
            resume_preprocess(root,r['location_id'],r['scene_id'],r['run_id'],sink=sink,resume=r.get('resume',False))
        else:
            run_stage(root,r['location_id'],r['scene_id'],r['run_id'],r['stage'],resume=r.get('resume',False),sink=sink)
        _,run,_=_run(r)
        return True,{'run_id':run.id,'status':run.status.value,'manifest':run.model_dump(mode='json')}
    if action in {'capture_config','normalize_capture','source_hashes'}:
        path,capture = _capture(r)
        if action=='source_hashes':
            rows=[]
            for source in capture.source.files:
                row=source.model_dump(mode='json')
                source_path=Path(source.windows_path)
                row['present']=source_path.is_file()
                if r.get('verify',False):
                    row['matches']=row['present'] and source_path.stat().st_size==source.byte_size and sha256_file(source_path)==source.sha256
                rows.append(row)
            return True,{'sources':rows}
        if action=='normalize_capture':
            _check_hash(path,r.get('expected_sha256'))
            _unconsumed(_scene(r),capture.id)
            from gsstudio.domain.models import CaptureManifest
            data=capture.model_dump(mode='json')
            data['normalization'].update(r.get('normalization',{}))
            for name in ('start_seconds','end_seconds'):
                if name in r:
                    data['selection'][name]=r[name]
            if 'horizontal_fov' in r:
                data['horizontal_fov_degrees']=r['horizontal_fov']
            capture=CaptureManifest.model_validate(data)
            if capture.source.probe.duration_seconds and capture.selection.end_seconds>capture.source.probe.duration_seconds+.5:
                raise ValueError('Selection exceeds source duration')
            save_yaml(path,capture)
        return True,{'capture':capture.model_dump(mode='json'),'manifest_sha256':sha256_file(path)}
    if action in {'update_hierarchy','delete_hierarchy'}:
        from gsstudio.domain.models import LocationManifest,SceneManifest,CaptureManifest
        if r.get('capture_id'):
            path,model=_capture(r)
            cls=CaptureManifest
        elif r.get('scene_id'):
            path=_scene(r)/'scene.yaml'; cls=SceneManifest
        else:
            path=browse.require_location(root,r['location_id'])/'location.yaml'; cls=LocationManifest
        _inside(path,root)
        _check_hash(path,r.get('expected_sha256'))
        model=load_model(path,cls)
        if action=='delete_hierarchy':
            if r.get('capture_id'):
                _unconsumed(_scene(r),model.id)
            elif any(p.is_file() and p!=path for p in path.parent.rglob('*')):
                raise RuntimeError('Only empty hierarchy entries can be deleted; no recursive data deletion')
            path.unlink()
            return True,{'deleted':str(path)}
        data=model.model_dump(mode='json')
        if r.get('name') is not None:
            if cls is CaptureManifest:
                raise ValueError('Capture IDs are immutable; use notes for its label')
            data['display_name']=r['name']
        if 'notes' in r:
            key='description' if cls is SceneManifest else 'notes'
            if key not in cls.model_fields:
                raise ValueError('This manifest has no notes field')
            data[key]=r['notes']
        save_yaml(path,cls.model_validate(data))
        return True,{'manifest_path':str(path),'sha256':sha256_file(path)}
    if action=='run_config':
        _,run,_=_run(r)
        return True,{'manifest':run.model_dump(mode='json'),'config_hash':run.config_hash}
    if action in {'masks.review','masks.read'}:
        review,identity,_=_mask_dataset(r)
        if action=='masks.read':
            return True,{**_mask_row(review,review.get_item(r['image_id'])),
                         'dataset_identity':identity,'mask_semantics':review.mask_semantics}
        offset,limit=r.get('offset',0),r.get('limit',50)
        if type(offset)!=int or offset<0 or type(limit)!=int or not 1<=limit<=100:
            raise ValueError('Use offset>=0 and limit 1..100')
        items=[x for x in review.items if not r.get('status') or review.review_for(x.image_id)['status']==r['status']]
        return True,{'items':[_mask_row(review,x) for x in items[offset:offset+limit]],
                     'total':len(items),'offset':offset,'limit':limit,'dataset_identity':identity,
                     'mask_semantics':review.mask_semantics,
                     'summary':dict(Counter(review.review_for(x.image_id)['status'] for x in review.items))}
    if action=='masks.commit':
        return True,_mask_commit(r)
    if action in {'qa_report','experiments'}:
        scene,run,work=_run(r)
        if action=='experiments':
            return True,{'experiments':run.metrics.get('training_experiments',[])}
        path=scene/'qa'/f'{run.id}.md'
        return True,{'path':str(path),'text':path.read_text(encoding='utf-8'),
                     'manifest_status':run.status.value}
    if action in {'training_state','experiment.read','sources.list','preview.request','preview.read'}:
        output,dispatch=_experiment(r)
        if r.get('package_sha256') is not None and r['package_sha256']!=dispatch['package_sha256']:
            raise RuntimeError('Experiment package identity differs')
        if action in {'training_state','experiment.read'}:
            log=output/'train.log'
            tail=''
            if log.is_file():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-32768))
                    tail=stream.read(32768).decode('utf-8',errors='replace')
            return True,{'state':_optional_json(output/'control-state.json'),'dispatch':dispatch,
                         'training':_optional_json(output/'training.json'),
                         'config':_optional_json(output/'config.json'),
                         'metrics':_optional_json(output/'evaluation'/'metrics.json'),
                         'log_path':str(log),'log_tail':tail}
        if action=='sources.list':
            package,meta=_package_for_dispatch(dispatch)
            import numpy as np
            p=_optional_json(output/'project.json')
            norm={'scale':1.,'center':[0.,0.,0.]}
            if p:
                if p.get('studio_dataset',{}).get('package_sha256')!=dispatch['package_sha256']:
                    raise RuntimeError('Prepared project package identity differs')
                norm=p['studio_dataset']['normalization']
            scale,center=float(norm['scale']),np.asarray(norm['center'],float)
            if scale<=0 or not np.isfinite(scale) or center.shape!=(3,) or not np.isfinite(center).all():
                raise RuntimeError('Invalid prepared scene normalization')
            offset,limit=r.get('offset',0),r.get('limit',2000)
            if type(offset)!=int or offset<0 or type(limit)!=int or not 1<=limit<=2000:
                raise ValueError('Use offset>=0 and limit 1..2000')
            cameras=[]
            for row in meta['images'][offset:offset+limit]:
                w2c=np.asarray(row['world_to_camera'],float)@np.diag([1.,-1.,-1.,1.])
                w2c[:3,3]=scale*(w2c[:3,3]+w2c[:3,:3]@center)
                cameras.append({**row,'source_path':str(package/row['image']),
                                'mask_path':str(package/row['mask']),'w2c':w2c.tolist()})
            return True,{'package_sha256':dispatch['package_sha256'],'total':len(meta['images']),
                         'offset':offset,'limit':limit,'cameras':cameras,
                         'coordinate_system':'RUB world; RDF camera'}
        if action=='preview.request':
            if r.get('package_sha256')!=dispatch['package_sha256']:
                raise RuntimeError('Preview package identity mismatch')
            from gsstudio.pipeline.training.data import json_write,validate_package
            package,meta=_package_for_dispatch(dispatch)
            if r['image'] not in {x['image'] for x in meta['images']}:
                raise ValueError('Preview camera outside immutable package')
            from gsstudio.pipeline.training.preview import REQUEST_ID
            rid=r.get('request_id',uuid.uuid4().hex)
            if not isinstance(rid,str) or not REQUEST_ID.fullmatch(rid):
                raise ValueError('Preview request_id must be 32 lowercase hexadecimal characters')
            request={'schema_version':1,'request_id':rid,'package_sha256':dispatch['package_sha256'],'image':r['image']}
            json_write(output/'preview-request.json',request)
            saved=_optional_json(output/'training.json')
            if saved and saved.get('status') in {'succeeded','paused','stopped'}:
                import subprocess, os
                from gsstudio.infrastructure.runtime.configuration import runtime_config, isolated_module_command, child_environment
                cfg=runtime_config()
                child=subprocess.run(isolated_module_command(cfg,'local_postshot.studio_saved_preview',
                    '--output',str(output),'--request-id',rid),env=child_environment(cfg,core=True),
                    capture_output=True,text=True,encoding='utf-8',timeout=300,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if child.returncode:
                    raise RuntimeError('Saved camera preview failed: '+child.stderr.strip())
            return True,request
        preview=_optional_json(output/'preview.json')
        if preview is None or (r.get('request_id') and preview.get('request_id')!=r['request_id']):
            return True,{'preview':None}
        if preview.get('schema_version')!=1:
            raise RuntimeError('Unsupported preview schema')
        if preview.get('package_sha256',dispatch['package_sha256'])!=dispatch['package_sha256']:
            raise RuntimeError('Preview package identity differs')
        preview['package_sha256']=dispatch['package_sha256']
        for label in ('source','render'):
            path=_inside(output/preview[label+'_file'],output)
            _check_hash(path,preview[label+'_sha256'])
            preview[label+'_path']=str(path)
        return True,{'preview':preview}
    if action=='model_open':
        if r.get('import_id'):
            from gsstudio.application.external_imports import verify_external_import
            record,external=verify_external_import(root,r['import_id'])
            output=_inside(r['output'],external/'experiments')
            ledger=_optional_json(output.with_suffix('.json'))
            if (not ledger or ledger.get('status')!='succeeded' or
                    ledger.get('package_sha256')!=record['package_sha256']):
                raise RuntimeError('External experiment ledger is not verified successful')
            selected,dispatch=_verified_output(root,output)
            if dispatch['package_sha256']!=record['package_sha256']:
                raise RuntimeError('Model differs from external import')
            return True,{'path':str(selected/'model.ply'),'project_path':str(selected/'project.json'),
                         'model_sha256':dispatch['model_sha256']}
        if r.get('output'):
            # The explicit experiment already selects one Run ledger entry.
            # Verify it once; listing every verified model first repeated the
            # full depth/provenance and checkpoint checks for the same output.
            _,run,_=_run(r)
            # CLI experiments can use a custom directory within the project
            # root. Ownership comes from the exact successful Run ledger row.
            output=_inside(r['output'],root)
            entries=[m for m in run.metrics.get('training_experiments',[])
                     if m.get('status')=='succeeded' and m.get('backend')=='gsplat'
                     and m.get('model_sha256') and Path(str(m.get('output',''))).resolve()==output]
            if len(entries)!=1:
                raise ValueError('Select exactly one verified experiment output')
            selected,dispatch=_verified_output(root,output,entries[0]['model_sha256'])
            return True,{'path':str(selected/'model.ply'),'project_path':str(selected/'project.json'),
                         'model_sha256':dispatch['model_sha256']}
        from gsstudio.application.delivery import verified_models
        models=verified_models(root,r['location_id'],r['scene_id'],r['run_id'])
        if len(models)!=1:
            raise ValueError('Select exactly one verified experiment output')
        model=models[0]
        _verified_output(root,Path(model['path']).parent,model['sha256'])
        p=Path(model['path']).parent/'project.json'
        return True,{'path':model['path'],'project_path':str(p) if p.is_file() else None,'model_sha256':model['sha256']}
    if action=='select_sharp':
        from gsstudio.pipeline.input.sharp_select import run_select_sharp
        from inspect import signature
        options={k:v for k,v in r.items() if k in signature(run_select_sharp).parameters}
        options['sources']=[Path(x) for x in r['sources']]
        options['out']=Path(r['out']).resolve()
        for source in options['sources']:
            if source.resolve()==options['out'] or options['out'] in source.resolve().parents:
                raise ValueError('Select-sharp output must not contain original sources')
        return True,run_select_sharp(**options) or {'stage':r.get('stage','all')}
    return False,None

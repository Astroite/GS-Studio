"""Independent persistent editing; all masks use original source point indices.

Public RPC: dispatch({path, model_id, action, args?}, emit=None, control=None).
Export hook: model_cloud(project, model_record), then existing write_ply/spz.
Native render and export share the same materialization and SH transforms.
"""
import copy
import json
import math
from pathlib import Path
import uuid
from collections import OrderedDict

import numpy as np
from PIL import Image

from . import project
from .formats import load_cloud, write_ply
from .geometry import apply_edits
from .locks import operation_lock

_sources = OrderedDict()


def _source(model):
    from .studio_dataset import signature
    file = Path(model['path']).resolve()
    stamp = signature(file)
    cached = _sources.get(str(file))
    if not cached or cached[0] != stamp:
        sha = project.digest(file)
        cloud = load_cloud(file)
        if signature(file) != stamp:
            raise ValueError('Source model changed while reading')
        cached = (stamp,sha,cloud)
        _sources[str(file)] = cached
        while len(_sources)>2:
            _sources.popitem(last=False)
    _sources.move_to_end(str(file))
    return cached[1],cached[2]


def _base(model):
    return {'transform':copy.deepcopy(model.get('transform')), 'crop':copy.deepcopy(model.get('crop'))}


def _state(p,model):
    sha,cloud = _source(model)
    n = len(cloud['means'])
    record = model.get('editor')
    if not record:
        return {'source_sha256':sha,'keep':np.ones(n,bool),'selected':np.zeros(n,bool),
                'current':_base(model),'history':[],'future':[],'preview':None,'revision':0},cloud
    if record['source_sha256'] != sha:
        raise ValueError('Editor source hash changed; edits cannot be rebased implicitly')
    file = Path(p['path'])/record['state_file']
    if Path(record['state_file']).is_absolute() or Path(p['path']).resolve() not in file.resolve().parents:
        raise ValueError('Unsafe editor state path')
    if project.digest(file) != record['state_sha256']:
        raise ValueError('Editor history hash changed')
    with np.load(file,allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta['source_sha256'] != sha or meta['count'] != n or meta['schema_version'] != 1:
            raise ValueError('Editor state identity mismatch')
        def unpack(a):
            if a.shape != ((n+7)//8,):
                raise ValueError('Invalid editor index mask')
            return np.unpackbits(a,count=n).astype(bool)
        state = {'source_sha256':sha,'keep':unpack(data['keep']),'selected':unpack(data['selected']),
                 'current':meta['current'],'preview':meta['preview'],'revision':meta['revision']}
        for key in ('history','future'):
            state[key] = [(unpack(a),s) for a,s in zip(data[key],meta[key])]
            if len(data[key]) != len(meta[key]):
                raise ValueError('Invalid editor history')
    if state['current'] != _base(model):
        raise ValueError('Model transform/crop changed outside edit history')
    return state,cloud


def _persist(p,model,state,save_project=True):
    if not isinstance(model['id'],str) or any(c not in '0123456789abcdefABCDEF-' for c in model['id']):
        raise ValueError('Unsafe model id')
    root = Path(p['path'])/'edits'/model['id']
    root.mkdir(parents=True,exist_ok=True)
    state['revision'] += 1
    file = root/(uuid.uuid4().hex+'.npz')
    temp = file.with_suffix('.tmp')
    n = len(state['keep'])
    meta = {k:state[k] for k in ('source_sha256','current','preview','revision')}
    meta.update(schema_version=1,count=n)
    arrays = {k:np.packbits(state[k]) for k in ('keep','selected')}
    for key in ('history','future'):
        meta[key] = [s for _,s in state[key]]
        arrays[key] = np.stack([np.packbits(k) for k,_ in state[key]]) if state[key] else np.empty((0,(n+7)//8),np.uint8)
    import os
    with temp.open('xb') as f:
        np.savez_compressed(f, meta=json.dumps(meta,allow_nan=False), **arrays)
        f.flush(); os.fsync(f.fileno())
    temp.replace(file)
    model.update(copy.deepcopy(state['current']))
    model['editor'] = {'schema_version':1,'source_sha256':state['source_sha256'],
                       'state_file':file.relative_to(p['path']).as_posix(), 'state_sha256':project.digest(file),
                       'revision':state['revision'], 'deleted_count':int((~state['keep']).sum())}
    if save_project:
        project.save(p)


def _remember(state):
    state['history'].append((state['keep'].copy(),copy.deepcopy(state['current'])))
    state['future'].clear()


def _materialize(cloud,state,preview=False):
    result = {k:v[state['keep']] for k,v in cloud.items()}
    result['_indices'] = np.flatnonzero(state['keep'])
    edit = state['preview'] if preview and state['preview'] is not None else state['current']
    return apply_edits(result,edit['transform'],edit['crop'])


def model_cloud(p,model,*,include_preview=False,with_indices=False):
    state,cloud = _state(p,model)
    result = _materialize(cloud,state,include_preview)
    if not with_indices:
        result.pop('_indices')
    return result


# Pipeline worker uses this public export hook name.
edited_cloud = model_cloud


def cleaned_history_record(p,source,removed_indices,mid,name):
    """Independent cleaned model, same immutable source and complete undo chain."""
    state,cloud = _state(p,source)
    if state['preview'] is not None:
        raise ValueError('Commit or cancel the editor preview before cleaning')
    removed = np.asarray(removed_indices,dtype=np.int64)
    if removed.ndim!=1 or (removed<0).any() or (removed>=len(state['keep'])).any():
        raise ValueError('Invalid original point indices for cleaning')
    _remember(state)
    state['keep'][removed] = False
    state['selected'][:] = False
    material = _materialize(cloud,state)
    if not len(material['means']):
        raise ValueError('Cleaning removes every Gaussian')
    record = project.model_record(source['path'],len(material['means']),name,mid)
    _persist(p,record,state,save_project=False)
    return record


def update_model_edits(p,model,changes):
    """project.update hook: keep inspector edits in this same undo history."""
    state,cloud = _state(p,model)
    current = {**state['current'],**copy.deepcopy(changes)}
    trial = {**state,'current':current,'preview':None}
    if not len(_materialize(cloud,trial)['means']):
        raise ValueError('Edits remove every Gaussian')
    _remember(state)
    state['current'],state['preview'] = current,None
    state['selected'][:] = False
    _persist(p,model,state,save_project=False)


def validate_camera(camera):
    c = copy.deepcopy(camera)
    if 'w2c' not in c and 'world_to_camera' in c:
        c['w2c'] = c['world_to_camera']
    K,pose = np.asarray(c['K'],float),np.asarray(c['w2c'],float)
    if (K.shape != (3,3) or pose.shape != (4,4) or not np.isfinite(K).all() or
            not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]) or
            not np.allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(pose[:3,:3]),1,atol=1e-5) or
            not np.allclose(K[2],[0,0,1]) or min(K[0,0],K[1,1])<=0 or
            K[0,1]!=0 or K[1,0]!=0 or
            any(isinstance(c[k],bool) or not isinstance(c[k],int) or not 0<c[k]<=16384 for k in ('width','height'))):
        raise ValueError('Invalid RDF projection camera')
    return c


def project_points(xyz,camera):
    pose,K = np.asarray(camera['w2c']),np.asarray(camera['K'])
    camera_xyz = xyz@pose[:3,:3].T+pose[:3,3]
    z = camera_xyz[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        xy = (camera_xyz@K.T)[:,:2]/z[:,None]
    return xy,z


def compositing_weights(alpha,pixels):
    """Front-to-back alpha*T, pairs sorted by pixel then depth (CPU/GPU)."""
    import torch
    if not alpha.numel():
        return alpha
    # Double prefix logs avoid catastrophic cancellation between distant pixels.
    logs = torch.log1p(-alpha.double().clamp(max=.999))
    prefix = torch.cumsum(logs,0)
    positions = torch.arange(len(alpha),device=alpha.device)
    first = torch.cat([torch.ones(1,device=alpha.device,dtype=torch.bool),pixels[1:] != pixels[:-1]])
    starts = torch.cummax(torch.where(first,positions,0),0).values
    previous = torch.where(starts>0,prefix[(starts-1).clamp_min(0)],0)
    return alpha*torch.exp(prefix-logs-previous).to(alpha.dtype)


def center_candidates(cloud,camera,rectangle):
    rectangle = np.asarray(rectangle,float)
    if rectangle.shape != (4,) or not np.isfinite(rectangle).all():
        raise ValueError('Rectangle requires four finite pixel coordinates')
    xy,z = project_points(cloud['means'],camera)
    x0,y0,x1,y1 = rectangle
    return ((z>0)&(xy[:,0]>=max(0,min(x0,x1)))&(xy[:,0]<=max(x0,x1))&(xy[:,0]<camera['width'])&
            (xy[:,1]>=max(0,min(y0,y1)))&(xy[:,1]<=max(y0,y1))&(xy[:,1]<camera['height']))


def native_selection(cloud,camera,rectangle,policy=None,control=None,*,tile_size=96):
    """Old GS compositing visibility, using the SAME native projection policy.

    Memory is O(full render metadata + N maxima), independent of pixel/splat
    intersection count. One bounded pixel chunk per cancellable launch.
    Never replaces failed visibility with nearest-center/depth-only picking.
    Precision render is produced from these same parameters and camera.
    """
    import torch
    from .training import tensors_from_cloud,render_parameters
    center_candidates(cloud,camera,rectangle)  # validate finite rectangle
    if control is not None and control.is_set():
        raise InterruptedError('Editor selection cancelled')
    params = tensors_from_cloud({k:v for k,v in cloud.items() if k != '_indices'})
    degree = round(math.sqrt(cloud['sh'].shape[1])-1)
    x0,y0,x1,y1 = rectangle
    left,right = max(0,math.ceil(min(x0,x1)-.5)),min(camera['width'],math.floor(max(x0,x1)-.5)+1)
    top,bottom = max(0,math.ceil(min(y0,y1)-.5)),min(camera['height'],math.floor(max(y0,y1)-.5)+1)
    with torch.no_grad():
        image,_,full_meta = render_parameters(params,camera,degree,projection_policy=policy)
        xy,z = full_meta['means2d'][0],full_meta['depths'][0]
        cand = ((z>0)&(xy[:,0]>=max(0,min(x0,x1)))&(xy[:,0]<=max(x0,x1))&(xy[:,0]<camera['width'])&
                (xy[:,1]>=max(0,min(y0,y1)))&(xy[:,1]<=max(y0,y1))&(xy[:,1]<camera['height']))
        accepted = torch.zeros_like(cand)
        for ty in range(top,bottom,tile_size):
            for tx in range(left,right,tile_size):
                if control is not None and control.is_set():
                    torch.cuda.current_stream().synchronize()
                    raise InterruptedError('Editor selection cancelled')
                # Full-frame projection, original pixel coordinates and original
                # depth-sorted tile lists. The editor helper bounds pixel work
                # only; it never reprojects/crops K or masks error-map inputs.
                from .studio_selection import roi_max
                weights = roi_max(full_meta,(tx,ty,min(tx+tile_size,right),min(ty+tile_size,bottom)))
                accepted |= cand & (weights>=1/255)
                # Bound queued work as well as allocations. Cancellation and
                # GPU-lock release cannot leave earlier ROI chunks running.
                torch.cuda.current_stream().synchronize()
    return accepted.cpu().numpy(),image[0].clamp(0,1).cpu().numpy()


def _render(cloud,camera,policy):
    import torch
    from .training import tensors_from_cloud,render_parameters
    with torch.no_grad():
        image,_,_ = render_parameters(tensors_from_cloud(cloud),camera,round(math.sqrt(cloud['sh'].shape[1])-1),projection_policy=policy)
    return image[0].clamp(0,1).cpu().numpy()


def dispatch(params,emit=None,control=None):
    with operation_lock(params['path'],'editor.operation'):
        p = project.load(params['path'])
        model = next((m for m in p['models'] if m['id']==params['model_id']),None)
        if model is None:
            raise ValueError('Unknown model')
        state,cloud = _state(p,model)
        action,args = params['action'],params.get('args') or {}
        policy = (p.get('training',{}).get('recipe') or {}).get('projection')
        image = None
        material = _materialize(cloud,state,True)
        ids = material.pop('_indices')
        if not len(ids):
            raise ValueError('Edits remove all points')
        if action == 'select_rect':
            camera = validate_camera(args['camera'])
            if args.get('xray',False):
                selected = center_candidates(material,camera,args['rectangle'])
                from .studio_bridge import gpu_session
                with gpu_session(cancel=control):
                    image = _render(material,camera,policy)
            else:
                from .studio_bridge import gpu_session
                with gpu_session(cancel=control):
                    selected,image = native_selection(material,camera,args['rectangle'],policy,control)
            state['selected'][:] = False
            state['selected'][ids[selected]] = True
        elif action == 'clear':
            state['selected'][:] = False
        elif action == 'invert':
            state['selected'][ids] = ~state['selected'][ids]
        elif action == 'delete':
            if state['preview'] is not None:
                raise ValueError('Commit or cancel preview before deleting')
            if state['selected'].any():
                retained = state['keep'] & ~state['selected']
                trial = {**state,'keep':retained}
                if not len(_materialize(cloud,trial)['means']):
                    raise ValueError('Cannot delete every visible Gaussian')
                _remember(state)
                state['keep'] = retained
                state['selected'][:] = False
        elif action == 'preview':
            edit = {**copy.deepcopy(state['current']),**{k:args[k] for k in ('transform','crop') if k in args}}
            trial = {**state,'preview':edit}
            if not len(_materialize(cloud,trial,True)['means']):
                raise ValueError('Preview removes all Gaussians')
            state['preview'] = edit
            state['selected'][:] = False
        elif action == 'commit':
            if state['preview'] is not None:
                _remember(state)
                state['current'],state['preview'] = state['preview'],None
        elif action == 'cancel_preview':
            state['preview'] = None
        elif action in ('undo','redo'):
            source,destination = ('history','future') if action=='undo' else ('future','history')
            if state[source]:
                state[destination].append((state['keep'].copy(),copy.deepcopy(state['current'])))
                state['keep'],state['current'] = state[source].pop()
            state['preview'] = None
            state['selected'][:] = False
        elif action not in ('state','render'):
            raise ValueError('Unknown editor action')
        material = _materialize(cloud,state,True)
        ids = material.pop('_indices')
        if action == 'render' or ('camera' in args and image is None):
            camera = validate_camera(args['camera'])
            from .studio_bridge import gpu_session
            with gpu_session(cancel=control):
                image = _render(material,camera,policy)
        # Revisioned artifacts make concurrent readers safe; JSON stays bounded.
        if action not in ('state','render'):
            _persist(p,model,state)
        folder = Path(p['path'])/'edits'/model['id']
        folder.mkdir(parents=True,exist_ok=True)
        selected = state['selected'][ids]
        result = {'project':p,'selection_count':int(selected.sum()),'revision':state['revision'],
                  'can_undo':bool(state['history']),'can_redo':bool(state['future'])}
        if selected.any():
            which = np.flatnonzero(selected)
            if len(which)>10000:
                which = which[np.linspace(0,len(which)-1,10000,dtype=int)]
            file = folder/f'selection-{state["revision"]}.ply'
            write_ply(file,{k:v[which] for k,v in material.items()})
            result['selection_preview'] = str(file)
        if image is not None:
            file = folder/f'native-{state["revision"]}.png'
            Image.fromarray((image*255).round().astype(np.uint8)).save(file)
            result['native_render'] = str(file)
        if emit:
            emit('editor', {k:v for k,v in result.items() if k!='project'})
        return result

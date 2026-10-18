"""Render an inactive GS experiment from its verified full checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np

from . import project
from .locks import operation_lock
from .studio_preview import Preview, REQUEST_ID


def render_saved(output, request_id):
    output = Path(output).resolve()
    if not REQUEST_ID.fullmatch(request_id):
        raise ValueError('Invalid saved preview request identity')
    from .studio_bridge import gpu_session
    # Serialize previews before taking the project lock. A resumed trainer's
    # nonblocking project lock still prevents reading a changing checkpoint.
    with gpu_session(), operation_lock(output, 'saved_training_preview'):
        request = json.loads((output/'preview-request.json').read_text(encoding='utf-8'))
        if request.get('request_id') != request_id:
            return {'status':'superseded', 'request_id':request_id}
        p = project.load(output)
        meta = json.loads((output/'training.json').read_text(encoding='utf-8'))
        cfg = json.loads((output/'config.json').read_text(encoding='utf-8'))
        if meta.get('status') not in {'succeeded','paused','stopped'}:
            raise ValueError('Saved preview requires an inactive saved experiment')
        identity = p['studio_dataset']['package_sha256']
        if request.get('schema_version') != 1 or request.get('package_sha256') != identity or meta.get('package_sha256') != identity:
            raise ValueError('Saved preview package identity differs')
        rows = [row for row in p['reconstruction']['cameras'] if row['image'] == request.get('image')]
        if len(rows) != 1:
            raise ValueError('Saved preview camera is outside the experiment')
        package = Path(p['studio_dataset']['package']).resolve()
        if project.digest(package/'dataset.json') != identity:
            raise ValueError('Saved preview dataset manifest changed')
        dataset = json.loads((package/'dataset.json').read_text(encoding='utf-8'))
        # Rendering reads no initialization points, depth or optimizer updates.
        # Bind every camera to the hashed manifest and checkpoint's normalized
        # dataset identity without repeating training-observation reconstruction.
        from .geometry import CV_TO_WORLD
        bridge = copy.deepcopy(p['studio_dataset'])
        key = bridge.pop('identity')
        if project.identity(bridge) != key or key != p['reconstruction']['key']:
            raise ValueError('Saved preview prepared dataset identity changed')
        cameras = p['reconstruction']['cameras']
        if len(cameras) != len(dataset['images']):
            raise ValueError('Saved preview camera inventory changed')
        norm = bridge['normalization']
        for i,(camera,row) in enumerate(zip(cameras,dataset['images'])):
            pose = np.asarray(row['world_to_camera']) @ CV_TO_WORLD
            pose[:3,3] = norm['scale']*(pose[:3,3]+pose[:3,:3]@np.asarray(norm['center']))
            if (any(camera.get(k)!=v for k,v in row.items()) or
                    camera['name'] != row['image'] or camera['row_index'] != i or
                    not np.array_equal(camera['w2c'],pose)):
                raise ValueError('Saved preview camera/split identity changed')
        source = (package/rows[0]['image']).resolve()
        if not source.is_relative_to(package) or project.digest(source) != dataset['files'].get(rows[0]['image']):
            raise ValueError('Saved preview source image changed')
        checkpoint = output/'checkpoints/latest.pt'
        if project.digest(checkpoint) != meta.get('checkpoint_sha256'):
            raise ValueError('Saved preview checkpoint changed')
        from .studio_training import engine_identity, validate_checkpoint
        from .training import load_checkpoint
        if meta.get('engine') != engine_identity():
            raise ValueError('Saved preview engine identity differs')
        state = load_checkpoint(checkpoint, map_location='cpu')
        validate_checkpoint(state, cfg, p['studio_dataset']['identity'], p['settings'], p['studio_dataset']['train_indices'])
        if state['step'] != meta.get('completed_steps'):
            raise ValueError('Saved preview checkpoint step differs')
        current = json.loads((output/'preview-request.json').read_text(encoding='utf-8'))
        if current != request:
            return {'status':'superseded', 'request_id':request_id}
        params = {key:value.to('cuda') for key,value in state['parameters'].items()}
        # A concurrently queued different camera must never be rendered without
        # its own source verification. A same-camera newer request is safe.
        view_project = copy.deepcopy(p)
        view_project['reconstruction']['cameras'] = rows
        view_project['studio_dataset']['train_indices'] = [0]
        preview = Preview(view_project, meta['generation_id'])
        preview.last_request = None
        preview.tick(params, state['step'], cfg['steps'], force=True)
        return {'status':'rendered', 'request_id':request_id, 'step':state['step']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--request-id', required=True)
    args = parser.parse_args()
    print(json.dumps(render_saved(args.output, args.request_id)))


if __name__ == '__main__':
    main()

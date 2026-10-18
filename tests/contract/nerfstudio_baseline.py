"""Explicit old-environment fixture generator; never used as the test oracle at runtime.

Run with the original preprocessing Python, -B and CUDA_VISIBLE_DEVICES=-1.
All output is synthetic and must go to a new evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

from nerfstudio.data.utils import colmap_parsing_utils as parser
from nerfstudio.process_data import colmap_utils, equirect_utils


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), data=value.tolist())
    if hasattr(value, '_asdict'):
        return {key: encode(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def write_model(path, cameras, images, points):
    path.mkdir()
    for kind, records in (('cameras', cameras), ('images', images), ('points3D', points)):
        for suffix in ('text', 'binary'):
            extension = 'txt' if suffix == 'text' else 'bin'
            getattr(parser, f'write_{kind}_{suffix}')(records, path / f'{kind}.{extension}')


def generate(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only generator')
    if importlib.metadata.version('nerfstudio') != '1.1.5':
        raise RuntimeError('Use the accepted Nerfstudio 1.1.5 preprocessing environment')
    output.mkdir(parents=True, exist_ok=False)
    params = {
        'SIMPLE_PINHOLE': [27.5, 16.25, 8.75],
        'PINHOLE': [27.5, 28.25, 16.25, 8.75],
        'SIMPLE_RADIAL': [27.5, 16.25, 8.75, -.125],
        'RADIAL': [27.5, 16.25, 8.75, -.125, .0625],
        'OPENCV': [27.5, 28.25, 16.25, 8.75, -.125, .0625, -.03125, .015625],
        'OPENCV_FISHEYE': [27.5, 28.25, 16.25, 8.75, -.125, .0625, -.03125, .015625],
        'FULL_OPENCV': [27.5, 28.25, 16.25, 8.75, -.125, .0625, -.03125, .015625, -.0078125, .00390625, -.001953125, .0009765625],
        'FOV': [27.5, 28.25, 16.25, 8.75, .125],
        'SIMPLE_RADIAL_FISHEYE': [27.5, 16.25, 8.75, -.125],
        'RADIAL_FISHEYE': [27.5, 16.25, 8.75, -.125, .0625],
        'THIN_PRISM_FISHEYE': [27.5, 28.25, 16.25, 8.75, -.125, .0625, -.03125, .015625, -.0078125, .00390625, -.001953125, .0009765625],
    }
    cameras = {model.model_id + 1: parser.Camera(
        id=model.model_id + 1, model=model.model_name, width=36, height=18,
        params=np.array(params[model.model_name]))
        for model in sorted(parser.CAMERA_MODELS, key=lambda item: item.model_id)}
    quaternions = [np.array([.8, .2, -.3, .45]), np.array([1., 0, 0, 0]),
                   np.array([-.8, -.2, .3, -.45]), np.array([.5, .5, .5, .5])]
    quaternions = [q / np.linalg.norm(q) for q in quaternions]
    point_id = 2**32 + 17
    images = {image_id: parser.Image(
        id=image_id, qvec=q, tvec=np.array([1.25 * index, -2.5, .375 - index]),
        camera_id=2, name=f'view_{index:02d}/frame_{index + 1:06d}.jpg',
        xys=np.array([[.125, 1.5], [17.75, 8.125], [35.5, 17.25]]) if index < 3 else np.empty((0, 2)),
        point3D_ids=np.array([point_id, -1, 42]) if index < 3 else np.array([], dtype=np.int64))
        for index, (image_id, q) in enumerate(zip([17, 3, 31, 100], quaternions))}
    points = {
        point_id: parser.Point3D(id=point_id, xyz=np.array([1.125, -2.25, 4.5]),
            rgb=np.array([0, 127, 255]), error=.375, image_ids=np.array([17, 3, 31]), point2D_idxs=np.array([0, 0, 0])),
        42: parser.Point3D(id=42, xyz=np.array([-.125, .25, 3.75]),
            rgb=np.array([254, 1, 128]), error=.125, image_ids=np.array([17, 3]), point2D_idxs=np.array([2, 2])),
        900: parser.Point3D(id=900, xyz=np.array([0., -0., -.5]),
            rgb=np.array([11, 22, 33]), error=0., image_ids=np.array([], dtype=np.int64), point2D_idxs=np.array([], dtype=np.int64)),
    }
    write_model(output/'parsing', cameras, images, points)
    expected = {}
    for kind in ('cameras', 'images', 'points3D'):
        for suffix in ('text', 'binary'):
            extension = 'txt' if suffix == 'text' else 'bin'
            expected[f'{kind}_{suffix}'] = encode(getattr(parser, f'read_{kind}_{suffix}')(
                output/'parsing'/f'{kind}.{extension}'))
    expected['models'] = encode([tuple(model) for model in sorted(parser.CAMERA_MODELS)])
    expected['fields'] = {name: list(getattr(parser, name)._fields) for name in ('CameraModel', 'Camera', 'Image', 'Point3D')}
    expected['rotations'] = {str(key): parser.qvec2rotmat(image.qvec).tolist() for key, image in images.items()}
    # Preserve the baseline's behavior for non-unit input as well.
    expected['nonunit_rotation'] = parser.qvec2rotmat(np.array([.8, .2, -.3, .45])).tolist()
    expected['camera_params'] = {}
    for camera in list(cameras.values()) + [cameras[2]._replace(model='UNKNOWN')]:
        try:
            result = dict(result=colmap_utils.parse_colmap_camera_params(camera))
        except NotImplementedError as error:
            result = dict(error=type(error).__name__, message=str(error))
        expected['camera_params'][camera.model] = result
    save_json(output/'expected.json', expected)

    projection = np.random.default_rng(20261017).random((2, 3, 18, 36), dtype=np.float32)
    projection[0, :, :, :18] = 0
    projection[0, :, :, 18:] = 1
    np.save(output/'projection-input.npy', projection)
    specs = [(120, 0, 0, 16, 16), (120, -180, -45, 16, 16),
             (110, 180, 45, 12, 18), (84, 37, -22, 13, 17), (110, -91, 82, 18, 18)]
    projected = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for index, spec in enumerate(specs):
            projected[f'case_{index}'] = equirect_utils.equirect2persp(torch.from_numpy(projection), *spec).numpy()
    np.savez(output/'projection-results.npz', **projected)
    save_json(output/'projection-specs.json', specs)

    cases = []
    accepted = [camera for camera in cameras.values() if 'result' in expected['camera_params'][camera.model]]
    for name, single, keep, selected, single_mode, text_points, extras in [
        ('single_default', True, False, [cameras[2]], True, False, False),
        ('single_per_frame', True, False, [cameras[2]], False, False, True),
        ('single_original', True, True, [cameras[2]], True, True, True),
        ('multiple_default', False, False, accepted, True, False, True),
        ('multiple_original', False, True, accepted, False, True, False),
        ('nonone_camera_id', True, False, [cameras[4]], True, False, False),
    ]:
        case = output/'conversion'/name
        case.mkdir(parents=True)
        selected_cameras = {1 if single and name != 'nonone_camera_id' else cam.id: cam._replace(
            id=1 if single and name != 'nonone_camera_id' else cam.id) for cam in selected}
        selected_ids = list(selected_cameras)
        selected_images = {key: image._replace(camera_id=selected_ids[index % len(selected_ids)])
                           for index, (key, image) in enumerate(images.items())}
        # Use every accepted model in the multi-camera export, including both fisheye reductions.
        if not single:
            template = images[17]
            selected_images = {index + 1: template._replace(id=index + 1, camera_id=cam_id,
                qvec=quaternions[index % len(quaternions)],
                name=f'view_{index:02d}/frame_{index + 1:06d}.jpg')
                for index, cam_id in enumerate(selected_ids)}
        write_model(case/'model', selected_cameras, selected_images, points)
        if text_points:
            # Synthetic fixture only: force the baseline's documented text fallback.
            (case/'model'/'points3D.bin').unlink()
        target = case/'expected'
        target.mkdir()
        options = dict(keep_original_world_coordinate=keep, use_single_camera_mode=single_mode,
                       ply_filename='synthetic-points.ply')
        if extras:
            options.update(camera_mask_path=case/'masks'/'camera.png',
                image_id_to_depth_path={key: case/'depth'/f'{key}.png' for key in selected_images},
                image_rename_map={image.name: f'renamed/{key}.jpg' for key, image in selected_images.items()})
        count = colmap_utils.colmap_to_json(case/'model', target, **options)
        # Track text instead of a .ply extension (repository ignores production PLY artifacts).
        (target/'synthetic-points.ply').rename(target/'sparse-ply.txt')
        cases.append(dict(name=name, keep_original_world_coordinate=keep, use_single_camera_mode=single_mode,
                          extras=extras, count=count))
    save_json(output/'conversion-cases.json', cases)

    source_root = Path(parser.__file__).parents[2]
    source_paths = ['data/utils/colmap_parsing_utils.py', 'process_data/colmap_utils.py',
                    'process_data/equirect_utils.py', 'process_data/process_data_utils.py', 'utils/rich_utils.py']
    save_json(output/'baseline.json', dict(
        schema_version=1, python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
        nerfstudio=importlib.metadata.version('nerfstudio'), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        generator_sha256=digest(Path(__file__)),
        sources={name: digest(source_root/name) for name in source_paths},
        files={path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob('*')) if path.is_file()}))
    print(f'CPU baseline written: {output}')


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument('--output', type=Path, required=True)
    generate(argument_parser.parse_args().output)

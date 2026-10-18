"""Real pixel resizing with immutable training-only COLMAP provenance."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np

from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write, validate_package


def resize_package(source: Path, output: Path, max_size: int):
    from gsstudio.pipeline.editor.data import read_model, read_image, write_image, _write_training_colmap
    if type(max_size) is not int or max_size < 1:
        raise ValueError('Positive maximum image side required')
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Derived package must be separate from its immutable source')
    meta = validate_package(source)
    identity = dict(package_sha256=sha256_file(source/'dataset.json'), max_image_size=max_size,
                    algorithm='area-nearest-training-provenance-v1')
    if output.exists():
        current = validate_package(output)
        if current.get('resize') != identity:
            raise RuntimeError('Derived package identity differs')
        return current
    staging = output.with_name(f'.{output.name}.building-{uuid.uuid4().hex[:12]}')
    staging.mkdir(parents=True)
    # Failures remain reviewable; no source or existing package is overwritten.
    _, images, points = read_model(source/'colmap')
    rows = copy.deepcopy(meta['images'])
    by_name = {Path(r['image']).name: (i, r) for i, r in enumerate(rows)}
    colors, counts, observed, records, scales, masks = {}, {}, {}, [], {}, {}
    for index, (original, row) in enumerate(zip(meta['images'], rows)):
        w0, h0 = original['width'], original['height']
        factor = min(1., max_size/max(w0, h0))
        w, h = max(1, round(w0*factor)), max(1, round(h0*factor))
        sx, sy = w/w0, h/h0
        scales[index] = (sx, sy)
        pixels, keep = read_image(source/row['image']), read_image(source/row['mask'], True)
        if pixels.shape[:2] != (h0, w0) or keep.shape != (h0, w0):
            raise RuntimeError('Resize source pixels differ from camera dimensions')
        if (w, h) != (w0, h0):
            pixels = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_AREA)
            keep = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
        # Lossless output; colors below sample exactly these pixels.
        row['image'] = f'images/{index:08d}.png'
        row['mask'] = f'masks/{index:08d}.png'
        K = np.array(original['K'], dtype=float)
        K[0] *= sx
        K[1] *= sy
        row.update(width=w, height=h, K=K.tolist())
        write_image(staging/row['image'], pixels)
        write_image(staging/row['mask'], keep)
        masks[index] = keep
    for iid, image in images.items():
        index, row = by_name[image['name']]
        if row['split'] != 'train':
            raise RuntimeError('Heldout image in source training provenance')
        sx, sy = scales[index]
        pixels = read_image(staging/row['image'])
        obs = []
        for x, y, pid in image['obs']:
            x, y = x*sx, y*sy
            valid = np.isfinite([x, y]).all()
            px, py = (int(round(x)), int(round(y))) if valid else (-1, -1)
            valid = (valid and pid in points and 0 <= px < row['width'] and
                     0 <= py < row['height'] and masks[index][py, px] == 255)
            obs.append((float(x) if np.isfinite(x) else 0., float(y) if np.isfinite(y) else 0., pid if valid else -1))
            if valid:
                colors[pid] = colors.get(pid, np.zeros(3)) + pixels[py, px, ::-1]
                counts[pid] = counts.get(pid, 0) + 1
                observed.setdefault(pid, set()).add(iid)
        records.append((iid, row, image, obs))
    ids = sorted(pid for pid in counts if counts[pid] >= 2 and len(observed[pid]) >= 2)
    if len(ids) < 4:
        raise RuntimeError('Resizing leaves fewer than four valid training-only initialization points')
    _write_training_colmap(staging, records, points, colors, counts, ids)
    np.savez(staging/'points.npz', xyz=np.array([points[pid][0] for pid in ids], np.float32),
             rgb=np.array([np.rint(colors[pid]/counts[pid]) for pid in ids], np.uint8))
    depth_count = 0
    if 'sparse_depth.npz' in meta['files']:
        with np.load(source/'sparse_depth.npz', allow_pickle=False) as archive:
            depth = {k: archive[k].copy() for k in archive.files}
        keep = np.ones(len(depth['depth']), dtype=bool)
        accepted = set(ids)
        for j, (ri, u, v, pid) in enumerate(zip(*(depth[k] for k in ('row_index','u','v','point_id')))):
            ri = int(ri)
            sx, sy = scales[ri]
            u, v = int(round(float(u)*sx)), int(round(float(v)*sy))
            row = rows[ri]
            keep[j] = (pid in accepted and row['split'] == 'train' and 0 <= u < row['width']
                       and 0 <= v < row['height'] and masks[ri][v, u] == 255)
            depth['u'][j], depth['v'][j] = u, v
        depth = {k: v[keep] for k, v in depth.items()}
        depth_count = len(depth['depth'])
        np.savez(staging/'sparse_depth.npz', **depth)
    changed = {'points.npz', 'sparse_depth.npz'} | {r[k] for r in meta['images'] for k in ('image','mask')}
    for name in meta['files']:
        if name in changed or name.startswith('colmap/'):
            continue
        destination = staging/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, destination)
    result = {k: v for k, v in meta.items() if not k.startswith('_')}
    result.update(images=rows, initial_points=len(ids), sparse_depth_observations=depth_count,
                  resize=identity, files={p.relative_to(staging).as_posix():sha256_file(p)
                                          for p in sorted(staging.rglob('*')) if p.is_file()})
    json_write(staging/'dataset.json', result)
    validate_package(staging)
    # The core validator checks new tracks, colors, masks, cameras and anchors.
    from gsstudio.pipeline.training.presets import depth_availability
    depth_availability(staging)
    staging.rename(output)
    return validate_package(output)

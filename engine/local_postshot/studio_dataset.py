"""Immutable GS schema1 package -> native RUB world/RDF camera project.

Public API: prepare(package, output, settings=None) -> saved project dict.
No source files are linked for writing or replaced. Validation caches are keyed
by every file's OS change signature, including ctime (not just mtime/size).
"""
from collections import Counter, OrderedDict
import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
import struct
import uuid

import numpy as np
from PIL import Image

from . import project
from .geometry import CV_TO_WORLD, quat_matrix

_cache = OrderedDict()


class _Masks:
    def __init__(self,root,rows):
        self.root,self.rows,self.cache = root,rows,OrderedDict()

    def __getitem__(self,index):
        if index not in self.cache:
            with Image.open(self.root/self.rows[index]['mask']) as im:
                self.cache[index] = np.asarray(im)==255
            while len(self.cache)>4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def safe_path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
            PureWindowsPath(name).is_absolute() or Path(name).is_absolute() or
            any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError(f'Unsafe package path: {name}')
    path = (root / name).resolve(strict=True)
    if root not in path.parents or not path.is_file():
        raise ValueError(f'Package path escapes inventory: {name}')
    return path


def signature(path):
    s = Path(path).stat()
    import os
    change = s.st_ctime_ns
    if os.name == 'nt':
        # Windows st_ctime is file CREATION time, not the NTFS change time.
        # Query ChangeTime so restoring mtime/size cannot hide source writes.
        import ctypes
        from ctypes import wintypes
        import msvcrt
        class BasicInfo(ctypes.Structure):
            _fields_ = [(n,ctypes.c_longlong) for n in ('creation','access','write','change')]+[('attributes',wintypes.DWORD)]
        info = BasicInfo()
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        query = kernel.GetFileInformationByHandleEx
        query.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        query.restype = wintypes.BOOL
        with Path(path).open('rb') as f:
            if not query(msvcrt.get_osfhandle(f.fileno()),0,ctypes.byref(info),ctypes.sizeof(info)):
                # No trustworthy change signature: force full content validation.
                import time
                change = time.monotonic_ns()
            else:
                change = info.change
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, change)


def _read(f, fmt):
    size = struct.calcsize('<' + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError('Truncated COLMAP provenance')
    return struct.unpack('<' + fmt, data)


def _provenance(root, meta, masks, xyz, rgb):
    """Verify poses and training-only point tracks/colors from packaged COLMAP.

    The prepared points are reservoir-ordered; COLMAP points are ID-ordered.
    Compare multisets after the actual float32 conversion, never assume order.
    """
    required = [f'colmap/{n}.bin' for n in ('cameras', 'images', 'points3D')]
    if any(n not in meta['files'] for n in required):
        raise ValueError('Initialization provenance requires hashed training COLMAP tracks')
    cameras = {}
    with (root / required[0]).open('rb') as f:
        for _ in range(_read(f, 'Q')[0]):
            cid, model, w, h = _read(f, 'iiQQ')
            if model != 1:
                raise ValueError('Prepared COLMAP provenance must be PINHOLE')
            fx, fy, cx, cy = _read(f, '4d')
            cameras[cid] = (w, h, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
        if f.read(1):
            raise ValueError('Trailing COLMAP camera data')
    rows = {Path(r['image']).name: (i, r) for i, r in enumerate(meta['images'])}
    observations, sums, counts, tracks = {}, {}, {}, {}
    seen = set()
    with (root / required[1]).open('rb') as f:
        for _ in range(_read(f, 'Q')[0]):
            values = _read(f, 'i7di')
            iid, cid = values[0], values[-1]
            name = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Truncated COLMAP image name')
                if c == b'\0':
                    break
                name.extend(c)
            name = name.decode('utf-8')
            if name not in rows or name in seen or iid in observations:
                raise ValueError('COLMAP image inventory differs from package')
            seen.add(name)
            ri, row = rows[name]
            if row['split'] != 'train':
                raise ValueError('Heldout image leaks into initialization provenance')
            pose = np.eye(4)
            pose[:3, :3] = quat_matrix(np.array(values[1:5]))
            pose[:3, 3] = values[5:8]
            w, h, K = cameras[cid]
            if ((w, h) != (row['width'], row['height']) or
                    not np.allclose(K, row['K'], atol=1e-6) or
                    not np.allclose(pose, row['world_to_camera'], atol=1e-6)):
                raise ValueError('COLMAP/package camera pose mismatch')
            count = _read(f, 'Q')[0]
            raw = f.read(24 * count)
            if len(raw) != 24 * count:
                raise ValueError('Truncated COLMAP observations')
            obs = np.frombuffer(raw, dtype=[('x','<f8'), ('y','<f8'), ('pid','<i8')])
            observations[iid] = obs
            with Image.open(root / row['image']) as im:
                pixels = np.asarray(im.convert('RGB'))
            for j in np.flatnonzero(obs['pid'] >= 0):
                x, y, pid = obs[j].tolist()
                if not np.isfinite([x, y]).all():
                    raise ValueError('Nonfinite initialization observation')
                x, y, pid = int(round(x)), int(round(y)), int(pid)
                if not (0 <= x < w and 0 <= y < h and masks[ri][y, x]):
                    raise ValueError('Masked initialization observation')
                sums[pid] = sums.get(pid, np.zeros(3)) + pixels[y, x]
                counts[pid] = counts.get(pid, 0) + 1
                tracks.setdefault(pid, set()).add((iid, int(j)))
        if f.read(1):
            raise ValueError('Trailing COLMAP image data')
    if seen != {Path(r['image']).name for r in meta['images'] if r['split'] == 'train'}:
        raise ValueError('Incomplete training provenance image inventory')
    points, records = {}, Counter()
    with (root / required[2]).open('rb') as f:
        for _ in range(_read(f, 'Q')[0]):
            v = _read(f, 'Q3d3BdQ')
            pid, p, color, error, n = v[0], np.array(v[1:4]), np.array(v[4:7]), v[7], v[8]
            pairs = [_read(f, 'ii') for _ in range(n)]
            if (pid in points or not np.isfinite(p).all() or not np.isfinite(error) or
                    len(pairs) != len(set(pairs)) or set(pairs) != tracks.get(pid) or
                    len({i for i, _ in pairs}) < 2 or counts.get(pid, 0) < 2):
                raise ValueError('Invalid training-only initialization track')
            if not np.array_equal(color, np.rint(sums[pid] / counts[pid]).astype(np.uint8)):
                raise ValueError('Initialization colors differ from unmasked training observations')
            points[pid] = p
            records[(p.astype(np.float32).tobytes(), color.astype(np.uint8).tobytes())] += 1
        if f.read(1):
            raise ValueError('Trailing COLMAP point data')
    if set(points) != set(tracks) or records != Counter((p.tobytes(), c.tobytes()) for p,c in zip(xyz, rgb)):
        raise ValueError('points.npz differs from initialization provenance')
    return points


def validate_package(package):
    root = Path(package).resolve(strict=True)
    manifest = root / 'dataset.json'
    before = signature(manifest)
    manifest_bytes = manifest.read_bytes()
    meta = json.loads(manifest_bytes.decode('utf-8'))
    if (meta.get('schema_version') != 1 or meta.get('validation') != 'passed' or
            meta.get('coordinates') != 'COLMAP' or meta.get('mask_polarity') != 'white_keep' or
            meta.get('initialization_colors') != 'unmasked_training_observations_only'):
        raise ValueError('Unsupported/incomplete GS dataset or initialization provenance')
    files = meta.get('files')
    if not isinstance(files, dict) or 'points.npz' not in files:
        raise ValueError('Missing package inventory')
    paths = {n: safe_path(root, n) for n in files}
    if len({str(p).casefold() for p in paths.values()}) != len(paths):
        raise ValueError('Aliased package inventory')
    if 'dataset.json' in paths:
        raise ValueError('Manifest cannot hash itself')
    stamps = (before, tuple((n, signature(p)) for n,p in paths.items()))
    cached = _cache.get(str(root))
    if cached and cached[0] == stamps:
        _cache.move_to_end(str(root))
        return copy.deepcopy(cached[1])
    for name, path in paths.items():
        expected = files[name]
        if (not isinstance(expected, str) or len(expected) != 64 or
                project.digest(path) != expected):
            raise ValueError(f'Package hash changed: {name}')
    rows, has_masks, groups, names = meta.get('images'), [], {}, set()
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty package images')
    for row in rows:
        if row['image'] in names or any(row[k] not in paths for k in ('image', 'mask')):
            raise ValueError('Duplicate or unhashed image/mask')
        names.add(row['image'])
        w, h, frame, t = row['width'], row['height'], row['frame'], row['timestamp_seconds']
        if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (w, h)) or
                not isinstance(frame, int) or isinstance(frame, bool) or not np.isfinite(t)):
            raise ValueError('Invalid group/time/image dimensions')
        split = row['split']
        if split not in ('train', 'validation'):
            raise ValueError('Unknown group split')
        if frame in groups and groups[frame] != (split, t):
            raise ValueError('Group split leakage or inconsistent group time')
        groups[frame] = (split, t)
        pose, K = np.array(row['world_to_camera'], float), np.array(row['K'], float)
        if (pose.shape != (4,4) or K.shape != (3,3) or not np.isfinite(pose).all() or
                not np.isfinite(K).all() or not np.allclose(pose[3], [0,0,0,1], atol=1e-7) or
                not np.allclose(pose[:3,:3] @ pose[:3,:3].T, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(pose[:3,:3]), 1, atol=1e-5) or
                not np.allclose(K[2], [0,0,1]) or K[0,0] <= 0 or K[1,1] <= 0 or
                K[0,1] != 0 or K[1,0] != 0):
            raise ValueError('Invalid rigid pose or PINHOLE intrinsics')
        with Image.open(paths[row['image']]) as im:
            if im.size != (w,h):
                raise ValueError('Image/camera dimensions differ')
            im.load()
        with Image.open(paths[row['mask']]) as im:
            mask = np.asarray(im)
        if mask.shape != (h,w) or not np.isin(mask, [0,255]).all() or not (mask == 255).any():
            raise ValueError('Mask must be binary white-keep with valid observations')
        has_masks.append(not bool((mask==255).all()))
    if {s for s,t in groups.values()} != {'train', 'validation'}:
        raise ValueError('Package needs independent train and validation groups')
    if len({Path(r['image']).name.casefold() for r in rows}) != len(rows):
        raise ValueError('Ambiguous image basenames')
    with np.load(paths['points.npz'], allow_pickle=False) as data:
        if set(data.files) != {'xyz', 'rgb'}:
            raise ValueError('Expected points.npz xyz/rgb arrays')
        xyz, rgb = data['xyz'], data['rgb']
    if xyz.ndim==2 and len(xyz)<4:
        raise ValueError('Frozen native initialization requires at least 4 training-observed points (kNN k=4)')
    if (xyz.dtype != np.float32 or xyz.ndim != 2 or xyz.shape[1] != 3 or
            rgb.dtype != np.uint8 or rgb.shape != xyz.shape or not np.isfinite(xyz).all() or
            len(xyz) < 4 or meta.get('initial_points') != len(xyz)):
        raise ValueError('Invalid initialization xyz/rgb arrays/count (need four points)')
    masks = _Masks(root,rows)
    points = _provenance(root, meta, masks, xyz, rgb)
    depth_count = 0
    if 'sparse_depth.npz' in paths:
        with np.load(paths['sparse_depth.npz'], allow_pickle=False) as data:
            if not {'row_index','u','v','depth','point_id'} <= set(data.files):
                raise ValueError('Sparse depth missing geometry provenance')
            anchors = {k:data[k] for k in data.files}
        n = len(anchors['depth'])
        if any(a.shape != (n,) or not np.isfinite(a).all() for a in anchors.values()):
            raise ValueError('Invalid sparse depth arrays')
        for ri,u,v,z,pid in zip(*(anchors[k] for k in ('row_index','u','v','depth','point_id'))):
            if int(ri) != ri or not 0 <= ri < len(rows) or int(pid) != pid or pid not in points:
                raise ValueError('Invalid sparse depth indices')
            row = rows[int(ri)]
            if (row['split'] != 'train' or u != round(float(u)) or v != round(float(v)) or
                    not 0 <= u < row['width'] or not 0 <= v < row['height'] or
                    not masks[int(ri)][int(v),int(u)] or z <= 0):
                raise ValueError('Sparse depth leaks heldout/masked/outside observation')
            camera_z = (np.array(row['world_to_camera']) @ np.r_[points[pid],1])[2]
            if not np.isclose(z, camera_z, rtol=2e-5, atol=1e-6):
                raise ValueError('Sparse depth coordinate/scale mismatch')
        if meta.get('sparse_depth_observations', n) != n:
            raise ValueError('Sparse depth count mismatch')
        depth_count = n
    # Reject concurrent source replacement while checking the immutable package.
    after = (signature(manifest), tuple((n, signature(p)) for n,p in paths.items()))
    if after != stamps:
        raise ValueError('Package changed during validation')
    meta['_package_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
    meta['_has_masks'] = has_masks
    meta['_sparse_depth_observations'] = depth_count
    _cache[str(root)] = (stamps, copy.deepcopy(meta))
    while len(_cache) > 4:
        _cache.popitem(last=False)
    return meta


def package_signature(package):
    """Cheap immutable inventory guard for long-running observations/caches."""
    root = Path(package).resolve()
    manifest = root/'dataset.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    return (signature(manifest),tuple((n,signature(safe_path(root,n))) for n in meta['files']))


def sparse_depth_availability(package):
    """Public, verified availability for pipeline/UI; requested != effective."""
    meta = validate_package(package)
    count = meta['_sparse_depth_observations']
    return {'available':count>0,'observations':count,
            'reason':None if count else 'no_valid_training_anchors',
            'package_sha256':meta['_package_sha256']}


def prepare(package, output, settings=None):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output == package or package in output.parents or output in package.parents:
        raise ValueError('Output must be separate from immutable package')
    meta = validate_package(package)
    cfg = {**project.DEFAULTS, 'steps':max(30000,30*sum(r['split']=='train' for r in meta['images'])),
           'max_image_size':max(max(r['width'],r['height']) for r in meta['images']),
           **(settings or {})}
    largest = max(max(r['width'],r['height']) for r in meta['images'])
    if type(cfg['max_image_size']) is not int or cfg['max_image_size']<largest:
        raise ValueError('max_image_size would require resizing: prepare a coherently resized immutable package (images, K, masks and depth anchors) first')
    # Optional normalization applies one similarity to points/camera translations
    # AND all depth anchors. Identity by default; no implicit recenter/rescale.
    norm = cfg.pop('scene_normalization', {'scale':1., 'center':[0.,0.,0.]})
    scale, center = float(norm['scale']), np.asarray(norm['center'], float)
    if scale <= 0 or not np.isfinite(scale) or center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('Invalid scene normalization')
    norm = {'scale':scale, 'center':center.tolist(), 'center_coordinates':'RUB'}
    bridge = {'schema_version':1, 'package':str(package), 'package_sha256':meta['_package_sha256'],
              'normalization':norm, 'train_indices':[i for i,r in enumerate(meta['images']) if r['split']=='train'],
              'test_indices':[i for i,r in enumerate(meta['images']) if r['split']=='validation']}
    bridge['identity'] = project.identity(bridge)
    if (output/'project.json').exists():
        p = project.load(output)
        if p.get('studio_dataset') != bridge or p['settings'] != cfg:
            raise ValueError('Existing prepared project identity/settings differ')
        validate_project(p)
        return p
    output.mkdir(parents=True, exist_ok=True)
    for d in ('logs','models','renders','checkpoints','prepared'):
        (output/d).mkdir(exist_ok=True)
    cameras = []
    for i,row in enumerate(meta['images']):
        w2c = np.array(row['world_to_camera']) @ CV_TO_WORLD
        w2c[:3,3] = scale * (w2c[:3,3] + w2c[:3,:3] @ center)
        cameras.append({**row, 'name':row['image'], 'w2c':w2c.tolist(),
                        'has_mask':meta['_has_masks'][i], 'row_index':i})
    with np.load(package/'points.npz', allow_pickle=False) as data:
        points = (data['xyz'] @ CV_TO_WORLD[:3,:3].T - center) * scale
        colors = data['rgb'].astype(np.float32)/255
    sparse = output/'prepared'/'sparse_points.npz'
    np.savez(sparse, points=points.astype(np.float32), colors=colors)
    artifacts = {'sparse_points.npz':project.digest(sparse)}
    if 'sparse_depth.npz' in meta['files']:
        with np.load(package/'sparse_depth.npz', allow_pickle=False) as data:
            anchors = {k:data[k] for k in data.files}
        anchors['depth'] = anchors['depth'] * scale
        np.savez(output/'prepared'/'sparse_depth.npz', **anchors)
        artifacts['sparse_depth.npz'] = project.digest(output/'prepared'/'sparse_depth.npz')
    return project.save({'id':str(uuid.uuid4()), 'name':output.name, 'path':str(output),
                         'created_at':project.now(), 'status':'reconstructed', 'settings':cfg,
                         'models':[], 'assets':{'images':[], 'videos':[]}, 'studio_dataset':bridge,
                         'pipeline':{'reconstruction':{'status':'complete','key':bridge['identity']}},
                         'reconstruction':{'key':bridge['identity'], 'path':str(output/'prepared'),
                                           'image_path':str(package), 'cameras':cameras, 'artifacts':artifacts,
                                           'coordinate_system':'RUB world; RDF camera'}})


def validate_project(p):
    bridge = p['studio_dataset']
    meta = validate_package(bridge['package'])
    if meta['_package_sha256'] != bridge['package_sha256']:
        raise ValueError('Prepared package identity changed')
    expected = copy.deepcopy(bridge)
    key = expected.pop('identity')
    if project.identity(expected) != key or key != p['reconstruction']['key']:
        raise ValueError('Prepared project identity changed')
    expected_artifacts = {'sparse_points.npz'} | ({'sparse_depth.npz'} if 'sparse_depth.npz' in meta['files'] else set())
    if set(p['reconstruction']['artifacts'])!=expected_artifacts:
        raise ValueError('Prepared artifact inventory differs from verified package')
    if 'sparse_depth.npz' not in expected_artifacts and (Path(p['reconstruction']['path'])/'sparse_depth.npz').exists():
        raise ValueError('Undeclared prepared depth anchors')
    for name, sha in p['reconstruction']['artifacts'].items():
        if project.digest(safe_path(Path(p['reconstruction']['path']).resolve(), name)) != sha:
            raise ValueError('Prepared artifact changed')
    # The editable project record cannot bless different initialization/depth
    # merely by rewriting its artifact hash. Verify the actual conversion too.
    package = Path(bridge['package'])
    prepared = Path(p['reconstruction']['path'])
    norm = bridge['normalization']
    with np.load(package/'points.npz',allow_pickle=False) as source,np.load(prepared/'sparse_points.npz',allow_pickle=False) as converted:
        expected = ((source['xyz']@CV_TO_WORLD[:3,:3].T-np.asarray(norm['center']))*norm['scale']).astype(np.float32)
        if (set(converted.files)!={'points','colors'} or not np.array_equal(converted['points'],expected) or
                not np.array_equal(converted['colors'],source['rgb'].astype(np.float32)/255)):
            raise ValueError('Prepared initialization conversion/provenance changed')
    if 'sparse_depth.npz' in meta['files']:
        with np.load(package/'sparse_depth.npz',allow_pickle=False) as source,np.load(prepared/'sparse_depth.npz',allow_pickle=False) as converted:
            if set(source.files)!=set(converted.files) or any(not np.array_equal(converted[k],source[k]*norm['scale'] if k=='depth' else source[k]) for k in source.files):
                raise ValueError('Prepared depth normalization/provenance changed')
    # Cameras/order/splits are immutable too, not editable project settings.
    for i,(camera,row) in enumerate(zip(p['reconstruction']['cameras'], meta['images'])):
        norm = bridge['normalization']
        expected = np.asarray(row['world_to_camera']) @ CV_TO_WORLD
        expected[:3,3] = norm['scale']*(expected[:3,3]+expected[:3,:3]@np.asarray(norm['center']))
        if (any(camera.get(k)!=v for k,v in row.items()) or
                camera['name'] != row['image'] or camera['row_index'] != i or
                camera['K'] != row['K'] or not np.array_equal(camera['w2c'], expected) or
                camera['split'] != row['split'] or camera['frame'] != row['frame'] or
                camera['has_mask'] != meta['_has_masks'][i]):
            raise ValueError('Prepared camera/split identity changed')
    if len(p['reconstruction']['cameras']) != len(meta['images']):
        raise ValueError('Prepared camera inventory changed')
    return meta

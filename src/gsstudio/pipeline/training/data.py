"""Immutable segment packages shared by native gsplat and Postshot ADC."""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import uuid
from pathlib import Path

import cv2
from gsstudio.infrastructure.adapters.image_io import read_cv_image, write_cv_image
import numpy as np

from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.masks.masking import validate_mask_filter
from gsstudio.pipeline.masks.finalize import validate_mask_finalization
from gsstudio.domain.models import SegmentQAConfig
from gsstudio.pipeline.reconstruction.core import _write_colmap_binary_model
from gsstudio.pipeline.training.sparse_depth import export_sparse_depth_anchors, write_sparse_depth
from gsstudio.pipeline.quality.segments import frame_id, image_name, validate_segments
from gsstudio.infrastructure.persistence.storage import link_or_copy


def json_write(path: Path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def grouped_split(frame_ids: list[int], every=8) -> dict[int,str]:
    return {f:('validation' if i % every == every-1 else 'train') for i,f in enumerate(sorted(set(frame_ids)))}


def validate_package(path: Path) -> dict:
    meta = json.loads((path/'dataset.json').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1 or meta.get('validation') != 'passed':
        raise RuntimeError('Training package incomplete or unsupported')
    with np.load(path/'points.npz',allow_pickle=False) as initial:
        xyz,rgb=initial['xyz'],initial['rgb']
    if (xyz.dtype!=np.float32 or rgb.dtype!=np.uint8 or xyz.ndim!=2 or xyz.shape[1]!=3
            or xyz.shape!=rgb.shape or len(xyz)<4 or not np.isfinite(xyz).all()
            or meta.get('initial_points')!=len(xyz)):
        raise RuntimeError('Independent engine requires at least four finite verified initialization points')
    for name,digest in meta['files'].items():
        relative = Path(name)
        if relative.is_absolute() or ':' in name or '..' in relative.parts:
            raise RuntimeError('Unsafe package path')
        if sha256_file(path/relative) != digest:
            raise RuntimeError(f'Training package changed: {name}')
    if not any(r['split']=='train' for r in meta['images']) or not any(r['split']=='validation' for r in meta['images']):
        raise RuntimeError('Package needs separate temporal training/validation groups')
    groups = {}
    names=set()
    for r in meta['images']:
        if r['image'] in names or any(r[key] not in meta['files'] for key in ('image','mask')):
            raise RuntimeError('Duplicate or unverified image/mask inventory')
        names.add(r['image'])
        if r['split'] not in ('train','validation'):
            raise RuntimeError('Unknown dataset split')
        camera=np.asarray(r['world_to_camera'],dtype=float)
        intrinsic=np.asarray(r['K'],dtype=float)
        if camera.shape!=(4,4) or intrinsic.shape!=(3,3) or not np.isfinite(camera).all() or not np.isfinite(intrinsic).all():
            raise RuntimeError('Invalid package camera')
        if not np.isfinite(r['timestamp_seconds']) or r['width']<=0 or r['height']<=0:
            raise RuntimeError('Invalid package timestamp/dimensions')
        f = r['frame']
        if f in groups and groups[f] != r['split']:
            raise RuntimeError('Temporal group leaks across training and validation')
        groups[f] = r['split']
    return meta


def prepare_segment(dataset: Path, records: list[dict], attempt, settings: SegmentQAConfig,
                    segment_id: str, output: Path, report_path: Path | None = None,
                    max_initial_points: int = 1_000_000, duration_seconds: float | None = None, reuse_files: bool = False,
                    source_kind: str | None = None) -> dict:
    if output.exists():
        return _prepare_segment(dataset,records,attempt,settings,segment_id,output,report_path,max_initial_points,duration_seconds,reuse_files,source_kind)
    staging=output.with_name(f'.{output.name}.building-{uuid.uuid4().hex[:12]}')
    try:
        _prepare_segment(dataset,records,attempt,settings,segment_id,staging,report_path,max_initial_points,duration_seconds,reuse_files,source_kind)
        staging.rename(output)
        return validate_package(output)
    except Exception as error:
        if staging.exists():
            json_write(staging/'preparation-failure.json',dict(error=str(error),target=str(output.resolve())))
        raise


def select_training_window(segment, report, records, included, settings, duration):
    """Earliest duration window entirely inside an already validated segment."""
    if not math.isfinite(duration) or duration<=0:
        raise ValueError('Window duration must be finite and positive')
    times={i:float(r['timestamp_seconds']) for i,r in enumerate(records,1)}
    centers={s['frame']:np.array(s['center']) for s in report['samples']}
    for first in segment['frames']:
        start,end=times[first],times[first]+duration
        if end>segment['end_seconds']:
            break
        frames=[f for f in segment['frames'] if start<=times[f]<=end]
        frame_set=set(frames)
        images=[n for n in segment['images'] if frame_id(n) in frame_set]
        expected=sum(start<=times[frame_id(n)]<=end for n in included)
        ratio=len(images)/expected if expected else 0.
        positions=[centers[f] for f in frames]
        epsilon=max(float(np.linalg.norm(np.ptp(positions,axis=0)))*1e-6,1e-9)
        distinct=[]
        for p in positions:
            if all(np.linalg.norm(p-q)>epsilon for q in distinct):
                distinct.append(p)
        if len(frames)<settings.minimum_samples or len(distinct)<settings.minimum_positions or ratio<settings.registration_threshold:
            continue
        return {**segment,'frames':frames,'images':images,'start_seconds':start,'end_seconds':end,
            'input_images':expected,'registration_ratio':ratio,'distinct_positions':len(distinct),
            'requested_duration_seconds':duration,'source_segment':segment['id']}
    raise RuntimeError(f'No complete {duration:g}-second training window in {segment["id"]}; use a longer independent capture')


def _prepare_segment(dataset: Path, records: list[dict], attempt, settings: SegmentQAConfig,
                    segment_id: str, output: Path, report_path: Path | None = None,
                    max_initial_points: int = 1_000_000, duration_seconds: float | None = None, reuse_files: bool = False,
                    source_kind: str | None = None) -> dict:
    from gsstudio._vendor.nerfstudio.colmap_parsing_utils import read_cameras_binary, read_images_binary, qvec2rotmat, Point3D
    filtered = validate_mask_filter(dataset,verify_hashes=True)
    expected = {r['image'] for r in filtered['accepted']}
    final = validate_mask_finalization(dataset,expected)
    included = expected-set(final.get('excluded_images',[]))
    report = validate_segments(dataset,records,included,attempt,settings,report_path)
    matches = [s for s in report['segments'] if s['id']==segment_id and s['status']=='passed']
    if len(matches)!=1:
        raise RuntimeError(f'Segment is absent or failed: {segment_id}')
    segment = matches[0]
    identity_fields={'report':report,'segment':segment_id,'max_initial_points':max_initial_points}
    if duration_seconds is not None:
        segment=select_training_window(segment,report,records,included,settings,duration_seconds)
        identity_fields['window']=segment
    identity = canonical_hash(identity_fields)
    if output.exists():
        meta = validate_package(output)
        if meta['source_identity'] != identity:
            raise RuntimeError('Existing training package has different inputs')
        if source_kind is not None and meta.get('source_kind', source_kind) != source_kind:
            raise RuntimeError('Training package source kind differs from Capture')
        return meta
    output.mkdir(parents=True)
    (output/'images').mkdir()
    (output/'masks').mkdir()
    cameras = read_cameras_binary(dataset/'colmap'/'cameras.bin')
    images = read_images_binary(dataset/'colmap'/'images.bin')
    transforms=json.loads((dataset/'transforms.json').read_text(encoding='utf-8'))
    poses={image_name(f['file_path']):f for f in transforms['frames']}
    applied=np.eye(4)
    if 'applied_transform' in transforms:
        applied[:3]=transforms['applied_transform']
    wanted = set(segment['images'])
    selected = {i:im for i,im in images.items() if image_name(im.name) in wanted}
    if {image_name(im.name) for im in selected.values()} != wanted:
        raise RuntimeError('Segment image/COLMAP inventory mismatch')
    split = grouped_split(segment['frames'])
    rows, train_images, valid_observations = [], {}, {}
    from gsstudio.pipeline.reconstruction.colmap import media_root
    training_media = media_root(dataset)
    image_id_to_row_index = {}
    storage = []
    for image_id,im in sorted(selected.items()):
        name = image_name(im.name)
        f = frame_id(name)
        camera = cameras[im.camera_id]
        if camera.model != 'PINHOLE':
            raise RuntimeError(f'Unsupported training camera: {camera.model}')
        output_name = f'image_{image_id:08d}{Path(name).suffix}'
        mask = read_cv_image(str(training_media/'masks'/f'{name}.png'),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (camera.height,camera.width):
            raise RuntimeError(f'Mask/camera size mismatch: {name}')
        if reuse_files:
            storage.append(link_or_copy(training_media/'images'/name,output/'images'/output_name))
            storage.append(link_or_copy(training_media/'masks'/f'{name}.png',output/'masks'/f'{output_name}.png'))
        else:
            shutil.copy2(training_media/'images'/name,output/'images'/output_name)
            if not write_cv_image(str(output/'masks'/f'{output_name}.png'),mask):
                raise RuntimeError('Mask write failed')
        world_to_camera = np.eye(4)
        world_to_camera[:3,:3] = qvec2rotmat(im.qvec)
        world_to_camera[:3,3] = im.tvec
        fx,fy,cx,cy = camera.params
        expected_pose=applied@np.linalg.inv(world_to_camera)@np.diag([1.,-1.,-1.,1.])
        if not np.allclose(expected_pose,poses[name]['transform_matrix'],atol=1e-5):
            raise RuntimeError(f'QA pose/COLMAP mismatch: {name}')
        for key,value in zip(('fl_x','fl_y','cx','cy','w','h'),(fx,fy,cx,cy,camera.width,camera.height)):
            actual=poses[name].get(key,transforms.get(key))
            if actual is None or not np.isclose(actual,value,atol=1e-5):
                raise RuntimeError(f'QA camera/COLMAP mismatch: {name}: {key}')
        rows.append(dict(image=f'images/{output_name}',mask=f'masks/{output_name}.png',source_image=name,
            frame=f,timestamp_seconds=records[f-1]['timestamp_seconds'],split=split[f],width=camera.width,height=camera.height,
            K=[[float(fx),0,float(cx)],[0,float(fy),float(cy)],[0,0,1]],world_to_camera=world_to_camera.tolist()))
        image_id_to_row_index[image_id] = len(rows) - 1
        if split[f]=='train':
            train_images[image_id] = im._replace(name=output_name)
            xy = np.rint(im.xys).astype(np.int64)
            valid = (xy[:,0]>=0)&(xy[:,0]<camera.width)&(xy[:,1]>=0)&(xy[:,1]<camera.height)
            indices = np.flatnonzero(valid)
            indices = indices[mask[xy[indices,1],xy[indices,0]]==255]
            valid_observations[image_id] = set(indices.tolist())
    # Stream the multi-million-point source model. Reservoir sampling bounds RAM
    # and gives both trainers the exact same deterministic initial point set.
    rng = np.random.default_rng(20260912)
    reservoir, count = [], 0
    with (dataset/'colmap'/'points3D.bin').open('rb') as stream:
        total = struct.unpack('<Q',stream.read(8))[0]
        for _ in range(total):
            header = stream.read(51)
            if len(header)!=51:
                raise RuntimeError('Truncated COLMAP point model')
            values = struct.unpack('<Q3d3BdQ',header)
            length = values[-1]
            raw = stream.read(8*length)
            if len(raw)!=8*length:
                raise RuntimeError('Truncated point track')
            track = np.frombuffer(raw,dtype='<i4').reshape(-1,2)
            track = np.array([(int(i),int(j)) for i,j in track if int(j) in valid_observations.get(int(i),())],dtype=np.int32).reshape(-1,2)
            if len(track)<2 or not np.isfinite(values[1:4]).all():
                continue
            point = Point3D(id=values[0],xyz=np.array(values[1:4]),rgb=np.array(values[4:7]),error=values[7],image_ids=track[:,0],point2D_idxs=track[:,1])
            count += 1
            if len(reservoir)<max_initial_points:
                reservoir.append(point)
            else:
                index = int(rng.integers(count))
                if index<max_initial_points:
                    reservoir[index]=point
    points = {p.id:p for p in reservoir}
    if len(points)<4:
        raise RuntimeError('Independent engine requires four unmasked training-only initialization points')
    for i,im in train_images.items():
        ids = np.array([p if p in points and j in valid_observations[i] else -1 for j,p in enumerate(im.point3D_ids)])
        train_images[i]=im._replace(point3D_ids=ids)
    # Source COLMAP colors may average validation observations. Recompute them
    # solely from the retained, unmasked training tracks for both backends.
    point_index={p.id:i for i,p in enumerate(reservoir)}
    colors=np.zeros((len(reservoir),3),dtype=np.float64)
    color_counts=np.zeros(len(reservoir),dtype=np.int64)
    for im in train_images.values():
        pixels=read_cv_image(str(output/'images'/im.name),cv2.IMREAD_COLOR)
        positions=np.flatnonzero(im.point3D_ids>=0)
        indices=np.array([point_index[int(im.point3D_ids[j])] for j in positions],dtype=np.int64)
        xy=np.rint(im.xys[positions]).astype(np.int64)
        np.add.at(colors,indices,pixels[xy[:,1],xy[:,0]][:,::-1])
        np.add.at(color_counts,indices,1)
    if np.any(color_counts<2):
        raise RuntimeError('Initialization track/color inventory mismatch')
    reservoir=[p._replace(rgb=np.rint(colors[i]/color_counts[i]).astype(np.uint8)) for i,p in enumerate(reservoir)]
    points={p.id:p for p in reservoir}
    used_cameras = {im.camera_id for im in train_images.values()}
    _write_colmap_binary_model(output/'colmap',{i:cameras[i] for i in used_cameras},train_images,points)
    np.savez(output/'points.npz',xyz=np.array([p.xyz for p in reservoir],dtype=np.float32),rgb=np.array([p.rgb for p in reservoir],dtype=np.uint8))
    sparse_anchors = export_sparse_depth_anchors(train_images, rows, points, image_id_to_row_index)
    write_sparse_depth(output/'sparse_depth.npz', sparse_anchors)
    json_write(output/'segments.json',report)
    json_write(output/'selection.json',segment)
    if reuse_files:
        json_write(output/'storage.json',dict(files=storage,duplicate_bytes=sum(r['duplicate_bytes'] for r in storage)))
    meta = dict(schema_version=1,validation='passed',source_identity=identity,source_dataset=str(dataset.resolve()),
                segment=segment_id,coverage_status=report['coverage_status'],mask_polarity='white_keep',
                selection=segment,
                coordinates='COLMAP',images=rows,initial_points=len(points),eligible_initial_points=count,
                initialization_colors='unmasked_training_observations_only',
                sparse_depth_observations=int(len(sparse_anchors['depth'])),
                files={p.relative_to(output).as_posix():sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()})
    if source_kind is not None:
        meta['source_kind'] = source_kind
    json_write(output/'dataset.json',meta)
    return validate_package(output)


def postshot_adapter(package: Path, output: Path) -> Path:
    meta = validate_package(package)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for name in ['images','masks']:
        (output/name).mkdir()
    shutil.copytree(package/'colmap',output/'colmap')
    storage = []
    for row in meta['images']:
        if row['split']!='train':
            continue
        name = Path(row['image']).name
        storage.append(link_or_copy(package/row['image'],output/'images'/name))
        mask = cv2.imdecode(np.fromfile(package/row['mask'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Postshot source mask cannot be decoded: {row["mask"]}')
        ok, encoded = cv2.imencode('.png',255-mask)
        if not ok:
            raise RuntimeError('Postshot mask conversion failed')
        encoded.tofile(output/'masks'/f'{Path(name).stem}.png')
    json_write(output/'storage.json',dict(files=storage,duplicate_bytes=sum(r['duplicate_bytes'] for r in storage)))
    json_write(output/'adapter.json',{'package_sha256':sha256_file(package/'dataset.json'),'mask_polarity':'white_ignore','split':'train'})
    validate_postshot_adapter(package,output,meta)
    return output


def validate_postshot_adapter(package: Path, output: Path, meta: dict | None = None):
    """Recheck prepared GUI inputs before CLI training, including mask polarity."""
    meta = validate_package(package) if meta is None else meta
    adapter = json.loads((output/'adapter.json').read_text(encoding='utf-8'))
    if adapter != dict(package_sha256=sha256_file(package/'dataset.json'),mask_polarity='white_ignore',split='train'):
        raise RuntimeError('Postshot adapter identity changed')
    expected = {'adapter.json'}
    if (output/'storage.json').is_file():
        expected.add('storage.json')
    for source in (package/'colmap').rglob('*'):
        if source.is_file():
            name = source.relative_to(package).as_posix()
            expected.add(name)
            if sha256_file(output/name) != sha256_file(source):
                raise RuntimeError(f'Postshot camera/point data changed: {name}')
    for row in meta['images']:
        if row['split'] != 'train':
            continue
        name = Path(row['image']).name
        image = f'images/{name}'
        mask = f'masks/{Path(name).stem}.png'
        expected.update((image,mask))
        if sha256_file(output/image) != meta['files'][row['image']]:
            raise RuntimeError(f'Postshot image changed: {image}')
        original = cv2.imdecode(np.fromfile(package/row['mask'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        converted = cv2.imdecode(np.fromfile(output/mask,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if original is None or converted is None or not np.array_equal(converted,255-original):
            raise RuntimeError(f'Postshot mask changed or has wrong polarity: {mask}')
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != expected:
        raise RuntimeError('Postshot adapter file inventory changed')

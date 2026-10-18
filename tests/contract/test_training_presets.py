"""Actual CPU pixels/provenance and pretraining API, without a CUDA call."""
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gsstudio.pipeline.editor.data import Dataset, read_image, write_image, read_model
from gsstudio.pipeline.training.data import json_write, validate_package
from gsstudio.pipeline.training.presets import prepare_training, resolve_settings, depth_availability
from gsstudio.pipeline.training.runner import train_package, _verify_identity
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.application.external_imports import prepare_external_import
from gsstudio.interfaces.worker.main import execute
from gsstudio.interfaces.cli.app import app
from tests.unit.test_external_imports import _source


@pytest.fixture
def large_package(tmp_path):
    images, model = _source(tmp_path)
    w, h = 3201, 1603
    (model/'cameras.txt').write_text(f'1 PINHOLE {w} {h} 2400 2300 1599.5 801\n', encoding='utf-8')
    lines = []
    coords = [(200.3, 220.2), (450.7, 500.1), (1001.4, 800.8), (2000.2, 1000.4), (3200.4, 1400)]
    for index, path in enumerate(sorted(images.glob('*.png')), 1):
        pixels = np.empty((h, w, 3), np.uint8)
        pixels[:,:,0] = (np.arange(w, dtype=np.uint32)[None,:]*19 % 256)
        pixels[:,:,1] = (np.arange(h, dtype=np.uint32)[:,None]*13 % 256)
        pixels[:,:,2] = index*30 if index < 3 else 250
        write_image(path, pixels)
        lines += [f'{index} 1 0 0 0 {index} 0 0 1 {path.name}',
                  ' '.join(f'{x} {y} {pid}' for pid, (x,y) in enumerate(coords, 1))]
    (model/'images.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (model/'points3D.txt').write_text(''.join(f'{pid} {pid%2} {pid//2} 2 255 255 255 0\n' for pid in range(1,6)), encoding='utf-8')
    package = tmp_path/'package'
    masks=tmp_path/'raw-masks';masks.mkdir()
    for path in sorted(images.glob('*.png')):
        mask=np.full((h,w),255,np.uint8);mask[100:150,100:150]=0
        write_image(masks/f'{path.name}.png',mask)
    Dataset(model, images=images, model=model, masks=masks, independent_images=True).prepare(package)
    meta = validate_package(package)
    row_index, u, v, pid = [], [], [], []
    for ri, row in enumerate(meta['images']):
        if row['split'] != 'train':
            continue
        for point, (x,y) in enumerate(coords, 1):
            row_index.append(ri); u.append(round(x)); v.append(round(y)); pid.append(point)
    np.savez(package/'sparse_depth.npz', row_index=np.array(row_index,np.int32),
             u=np.array(u,np.float32), v=np.array(v,np.float32),
             depth=np.full(len(pid),2,np.float32), point_id=np.array(pid,np.int64))
    meta.update(source_kind='perspective_images', sparse_depth_observations=len(pid))
    meta['files']['sparse_depth.npz'] = sha256_file(package/'sparse_depth.npz')
    json_write(package/'dataset.json',meta)
    return package


def test_actual_photo_resize_pixels_K_tracks_colors_and_depth(large_package):
    package = large_package
    before = {p:sha256_file(p) for p in package.rglob('*') if p.is_file()}
    original = validate_package(package)
    derived, meta, settings = prepare_training(package)
    assert derived != package and settings['steps'] == 10000
    assert settings['training_preset'] == 'photo' and settings['max_image_size'] == 1600
    assert (settings['seed'], settings['max_splats'], settings['sh_degree']) == (42,1000000,3)
    assert not any(settings[k] for k in ('photo_comp','use_bilateral_grid','use_sparse_depth','sparse_depth_requested'))
    assert settings['sparse_depth_available'] and settings['sparse_depth_observations'] == 8
    assert [r['split'] for r in original['images']] == [r['split'] for r in meta['images']]
    for old, new in zip(original['images'],meta['images']):
        assert (new['width'],new['height']) == (1600,801)
        assert read_image(derived/new['image']).shape == (801,1600,3)
        assert read_image(derived/new['mask'], True).shape == (801,1600)
        import cv2
        np.testing.assert_array_equal(read_image(derived/new['mask'],True),
            cv2.resize(read_image(package/old['mask'],True),(1600,801),interpolation=cv2.INTER_NEAREST))
        assert set(np.unique(read_image(derived/new['mask'],True))) == {0,255}
        expected = np.array(old['K'],float)
        expected[0] *= 1600/3201; expected[1] *= 801/1603
        np.testing.assert_array_equal(new['K'],expected)
        assert new['frame'] == old['frame'] and new['timestamp_seconds'] == old['timestamp_seconds']
        np.testing.assert_array_equal(new['world_to_camera'],old['world_to_camera'])
    cameras, images, points = read_model(derived/'colmap')
    assert len(points) == 4  # The rounded right-edge observation no longer fits.
    sums, counts = {}, {}
    for image in images.values():
        row = next(r for r in meta['images'] if Path(r['image']).name == image['name'])
        assert row['split'] == 'train'
        assert cameras[image['camera']][1:3] == (1600,801)
        pixels, mask = read_image(derived/row['image']), read_image(derived/row['mask'],True)
        for x,y,pid in image['obs']:
            if pid < 0: continue
            px,py = round(x),round(y)
            assert mask[py,px] == 255
            sums[pid] = sums.get(pid,np.zeros(3))+pixels[py,px,::-1]
            counts[pid] = counts.get(pid,0)+1
    for pid, (_,rgb) in points.items():
        np.testing.assert_array_equal(rgb,np.rint(sums[pid]/counts[pid]))
        assert rgb[0] == 45  # Train RGB red=30,60; heldout red=250 excluded.
    with np.load(derived/'sparse_depth.npz') as anchors:
        assert len(anchors['depth']) == 8
        assert set(anchors['point_id']) == {1,2,3,4}
        np.testing.assert_array_equal(anchors['u'][:4],np.rint(np.array([200,451,1001,2000])*1600/3201))
        np.testing.assert_array_equal(anchors['v'][:4],np.rint(np.array([220,500,801,1000])*801/1603))
        np.testing.assert_array_equal(anchors['depth'],np.full(8,2))
    assert depth_availability(derived)['observations'] == 8
    from local_postshot.studio_dataset import prepare
    project=prepare(derived,package.parent/'actual-core-photo',settings={
        'steps':settings['steps'],'training_recipe':settings['training_recipe']})
    assert {k:project['settings'][k] for k in ('steps','max_splats','seed','sh_degree','max_image_size','training_recipe')} == {
        'steps':10000,'max_splats':1000000,'seed':42,'sh_degree':3,'max_image_size':1600,'training_recipe':'quality_v2'}
    assert all(sha256_file(p) == digest for p,digest in before.items())
    assert prepare_training(package)[0] == derived
    gs, _, options = prepare_training(package, training_preset='gs', training_recipe='observed_v5')
    assert gs == package and options['steps'] == 30000 and options['use_sparse_depth']
    assert options['training_recipe'] == 'observed_v5'


@pytest.mark.parametrize('source,expected',[('perspective_images','photo'),('colmap','photo'),
    ('insta360_insv','gs'),('equirect_video','gs'),('equirect_sequence','gs'),
    ('perspective_video','gs'),('shared_package','gs')])
def test_source_defaults_recipe_independent(source,expected):
    meta = {'images':[{'split':'train'}]*1100}
    result = resolve_settings(meta,source_kind=source,training_recipe='observed_v6')
    assert result['training_preset'] == expected and result['training_recipe'] == 'observed_v6'
    assert result['steps'] == (10000 if expected == 'photo' else 33000)
    overridden = resolve_settings(meta,source_kind=source,training_preset='photo',steps=17,
                                  photo_comp=True,use_bilateral_grid=True,use_sparse_depth=True)
    assert overridden['steps'] == 17 and overridden['sparse_depth_requested']


def test_explicit_bilateral_requires_photometric_master():
    meta={'images':[{'split':'train'}]}
    for master in (None,False):
        with pytest.raises(ValueError,match='master enable'):
            resolve_settings(meta,source_kind='perspective_images',photo_comp=master,use_bilateral_grid=True)
    result=resolve_settings(meta,source_kind='perspective_images',photo_comp=True,use_bilateral_grid=True)
    assert result['photo_comp'] and result['use_bilateral_grid']


def test_real_external_options_rpc_cli_and_requested_effective_identity(tmp_path,monkeypatch):
    images,model = _source(tmp_path)
    root=tmp_path/'catalog'
    prepared = prepare_external_import(root,'photos',images=images,model=model,independent_images=True)
    package=Path(prepared['package_path'])
    assert not prepared['sparse_depth_available']
    request=dict(root=str(root),action='training_options',import_id='photos')
    options=execute(request)
    assert options['training_preset'] == 'photo' and options['steps'] == 10000
    assert options['sparse_depth_reason'] == 'no_valid_training_anchors'
    from local_postshot.studio_bridge import call
    rpc=call('studio.operation',request,request_id='actual-training-options')
    assert rpc == options  # Actual JSONL subprocess, same defaults/package identity.
    requested=execute(dict(request,use_sparse_depth=True,training_recipe='observed_v5'))
    assert requested['sparse_depth_requested'] and not requested['use_sparse_depth']
    assert requested['sparse_depth_status'] == 'unavailable'
    output=tmp_path/'experiment'
    record=train_package(package,output,dry_run=True,use_sparse_depth=True)
    assert record['sparse_depth_requested'] and not record['use_sparse_depth']
    assert '--no-use-sparse-depth' not in record['command']
    # Contract verification uses actual package anchors and synthetic CPU records.
    cfg={k:record[k] for k in ('package_sha256','photo_comp','use_bilateral_grid','use_sparse_depth',
        'sparse_depth_requested','sparse_depth_available','sparse_depth_observations','sparse_depth_status','training_recipe')}
    cfg.update(steps=10000,checkpoint=str(output/'checkpoints/latest.pt'),engine={
        'name':'local-postshot-independent-gsstudio','schema_version':1,
        'sources':{'training.py':'0'*64},'versions':{'torch':'2.8.0+cu129'}})
    json_write(output/'config.json',cfg)
    summary=dict(cfg,checkpoint=None)
    _verify_identity(output,record,summary,require_checkpoint=False)
    summary['use_sparse_depth']=True
    with pytest.raises(RuntimeError,match='use_sparse_depth'):
        _verify_identity(output,record,summary,require_checkpoint=False)
    monkeypatch.setenv('GSSTUDIO_DATA_ROOT',str(root))
    result=CliRunner().invoke(app,['external','options','photos'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['steps'] == 10000
    result=CliRunner().invoke(app,['external','train','photos','--dry-run','--training-preset','gs',
                                  '--no-photo-comp','--training-recipe','observed_v5'])
    assert result.exit_code == 0, result.output
    dispatch=json.loads(result.output)
    assert dispatch['training_preset'] == 'gs' and dispatch['requested_steps'] == 30000
    assert dispatch['training_recipe'] == 'observed_v5' and not dispatch['photo_comp']
    wire=call('studio.operation',dict(root=str(root),action='train_external_import',
        import_id='photos',dry_run=True,training_preset='gs',training_recipe='observed_v5',
        use_sparse_depth=True),request_id='actual-dry-dispatch')
    assert wire['source_kind'] == 'colmap' and wire['training_preset'] == 'gs'
    assert wire['training_recipe'] == 'observed_v5' and wire['sparse_depth_requested']
    assert wire['sparse_depth_status'] == 'unavailable' and not wire['use_sparse_depth']


@pytest.mark.parametrize('empty',[False,True])
def test_old_valid_shared_package_and_empty_depth_are_truthful(tmp_path,empty):
    images,model=_source(tmp_path)
    package=tmp_path/'shared'
    Dataset(model,images=images,model=model,independent_images=True).prepare(package)
    meta=validate_package(package)
    meta.pop('source_kind',None)  # Existing valid shared schema1 input remains usable.
    if empty:
        np.savez(package/'sparse_depth.npz',row_index=np.array([],np.int32),
            u=np.array([],np.float32),v=np.array([],np.float32),depth=np.array([],np.float32),
            point_id=np.array([],np.int64))
        meta['files']['sparse_depth.npz']=sha256_file(package/'sparse_depth.npz')
        meta['sparse_depth_observations']=0
    else:
        meta['sparse_depth_observations']=999  # A label cannot create actual anchors.
    json_write(package/'dataset.json',meta)
    before=sha256_file(package/'dataset.json')
    actual,_,settings=prepare_training(package)
    assert actual == package and settings['training_preset'] == 'gs' and settings['steps'] == 30000
    assert settings['sparse_depth_requested'] and not settings['use_sparse_depth']
    assert settings['sparse_depth_status'] == 'unavailable' and settings['sparse_depth_observations'] == 0
    assert sha256_file(package/'dataset.json') == before

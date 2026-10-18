"""Photo and review gates operate on temporary data only; no commercial tools."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from gsstudio.application import operations as ops
from gsstudio.application.studio import execute_additional,catalog
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.persistence.run_repository import load_run,save_run
from gsstudio.pipeline.masks.masking import filter_masked_images


@pytest.fixture
def photos(tmp_path):
    root=tmp_path/'catalog';root.mkdir()
    ops.create_location(root,'loc','地点');ops.create_scene(root,'loc','scene','场景')
    raw=tmp_path/'originals';raw.mkdir()
    files=[]
    for i,size in enumerate([(80,60),(60,90),(100,60)]):
        path=raw/f'photo-{i}.jpg'
        Image.new('RGB',size,(100+i,90,70)).save(path,quality=95)
        files.append(path)
    capture=ops.create_capture(root,'loc','scene','photos','perspective_images',files)
    return root,files,capture


def _request(root,action,run_id=None,**args):
    return dict(root=str(root),action=action,location_id='loc',scene_id='scene',
                capture_id='photos',**({'run_id':run_id} if run_id else {}),**args)


def test_native_photos_use_canonical_run_and_keep_immutable_originals(photos):
    root,files,capture=photos
    before=[sha256_file(x) for x in files]
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True)
    scene=root/'loc'/'scene';run=load_run(scene,run_id)
    assert run.config.input.source_kind=='perspective_images'
    assert run.config.input.source_sha256==before
    assert run.config.reconstruction.primary.panorama is None
    assert run.config.reconstruction.backend=='realityscan'
    assert run.config.reconstruction.fix_intrinsics is False
    assert run.stages['preprocess'].status.value=='succeeded'
    assert run.stages['reconstruct'].status.value=='pending'
    payload=json.loads((scene/run_id/run.config.input_relative_path/'dataset.json').read_text(encoding='utf-8'))
    assert [(x['width'],x['height']) for x in payload['frames']]==[(80,60),(60,90),(100,60)]
    assert [x['sha256'] for x in payload['frames']]==before
    assert [sha256_file(x) for x in files]==before
    assert catalog(root)['captures'][0]['source_kind']=='perspective_images'
    with pytest.raises(RuntimeError,match='QA must be accepted'):
        ops.train_segment(root,'loc','scene',run_id,'segment-001')


def test_photo_source_relink_hash_and_normalize_consumption_gates(photos,tmp_path):
    root,files,_=photos
    _,cfg=execute_additional(_request(root,'capture_config'))
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True,
                                     reconstruction_backend='colmap')
    assert load_run(root/'loc'/'scene',run_id).config.reconstruction.backend=='colmap'
    assert load_run(root/'loc'/'scene',run_id).config.reconstruction.fix_intrinsics is False
    with pytest.raises(RuntimeError,match='consumed'):
        execute_additional(_request(root,'normalize_capture',expected_sha256=cfg['manifest_sha256'],horizontal_fov=80))
    with pytest.raises(RuntimeError,match='hash'):
        ops.relink_capture_sources(root,'loc','scene','photos',list(reversed(files)))
    files[0].write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='identity changed'):
        ops.create_and_preprocess(root,'loc','scene','photos')


def _mask_run(photos):
    root,files,_=photos
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True)
    scene=root/'loc'/'scene';run=load_run(scene,run_id)
    dataset=scene/run_id/'reconstruction-primary'
    (dataset/'images').mkdir(parents=True);(dataset/'masks').mkdir()
    for i,file in enumerate(files,1):
        name=f'frame_{i:06d}.jpg'
        (dataset/'images'/name).write_bytes(file.read_bytes())
        Image.new('L',Image.open(file).size,255).save(dataset/'masks'/f'{name}.png')
    records=[{'image':f'frame_{i:06d}.jpg','mask':f'frame_{i:06d}.jpg.png','masked_fraction':0.}
             for i in range(1,len(files)+1)]
    filter_masked_images(dataset,records,run.config.masking.mask_discard_threshold)
    from gsstudio.domain.common import StageStatus
    run.stages['mask'].status=StageStatus.SUCCEEDED;save_run(scene,run)
    return root,run_id,dataset


def test_mask_edits_bind_source_and_mask_hash_then_require_explicit_review(photos):
    root,run_id,dataset=_mask_run(photos)
    _,page=execute_additional(_request(root,'masks.review',run_id,limit=1))
    assert page['total']==3 and len(page['items'])==1
    assert page['summary']=={'unreviewed':3}
    with pytest.raises(RuntimeError,match='explicit'):
        ops.finalize_masks(root,'loc','scene',run_id)
    _,media=execute_additional(_request(root,'masks.read',run_id,image_id=1))
    edit={k:media[k] for k in ('image_id','image_sha256','mask_sha256')}
    edit.update(status='exclude',polygons=[{'points':[[0,0],[20,0],[20,20]],'ignore':True}])
    _,result=execute_additional(_request(root,'masks.commit',run_id,
                                         dataset_identity=page['dataset_identity'],edits=[edit]))
    assert result['updated'][0]['mask_sha256']!=edit['mask_sha256']
    with pytest.raises(RuntimeError,match='identity changed'):
        ops.finalize_masks(root,'loc','scene',run_id,expected_identity=page['dataset_identity'])
    with pytest.raises(RuntimeError,match='identity changed'):
        execute_additional(_request(root,'masks.commit',run_id,dataset_identity=page['dataset_identity'],edits=[edit]))
    for image_id in (2,3):
        _,page=execute_additional(_request(root,'masks.review',run_id,limit=1))
        _,media=execute_additional(_request(root,'masks.read',run_id,image_id=image_id))
        edit={k:media[k] for k in ('image_id','image_sha256','mask_sha256')}
        edit['status']='ok'
        execute_additional(_request(root,'masks.commit',run_id,dataset_identity=page['dataset_identity'],edits=[edit]))
    result=ops.finalize_masks(root,'loc','scene',run_id)
    assert result['counts']['unreviewed']==0 and result['excluded_images']==['frame_000001.jpg']
    _,page=execute_additional(_request(root,'masks.review',run_id,limit=1))
    with pytest.raises(RuntimeError,match='immutable'):
        execute_additional(_request(root,'masks.commit',run_id,dataset_identity=page['dataset_identity'],edits=[edit]))


def test_mask_review_page_bounds_and_source_change_reject(photos):
    root,run_id,dataset=_mask_run(photos)
    with pytest.raises(ValueError,match='limit'):
        execute_additional(_request(root,'masks.review',run_id,limit=1000))
    Image.new('RGB',(80,60),(10,20,30)).save(dataset/'images'/'frame_000001.jpg')
    with pytest.raises(RuntimeError,match='hash changed'):
        execute_additional(_request(root,'masks.read',run_id,image_id=1))


def test_photo_collection_repair_is_bounded_by_available_native_photos(photos):
    root,_,_=photos
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True)
    run=load_run(root/'loc'/'scene',run_id)
    from gsstudio.pipeline.reconstruction.stage import repair_plan
    records=[{'timestamp_seconds':float(i),'source_frame_index':i} for i in range(3)]
    plan=repair_plan(run,records,records,{'missing_frames':[1]})
    assert plan['attempt']==1 and plan['candidate_fps']==1 and plan['frames']==[]
    assert run.config.reconstruction.repair_attempts==1


def test_qa_acceptance_requires_boolean_and_unchanged_source(photos):
    root,files,_=photos
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True)
    with pytest.raises(ValueError,match='explicit boolean'):
        ops.review_qa(root,'loc','scene',run_id,'false','reviewed')
    files[0].write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='identity changed'):
        ops.review_qa(root,'loc','scene',run_id,True,'reviewed')


def test_catalog_build_and_status_share_authoritative_run(photos):
    root,_,_=photos
    run_id=ops.create_and_preprocess(root,'loc','scene','photos',keep_intermediates=True)
    _,result=execute_additional(_request(root,'catalog.build'))
    assert result['counts']=={'locations':1,'scenes':1,'captures':1,'runs':1}
    assert Path(result['path']).is_file()
    _,detail=execute_additional(_request(root,'run_status',run_id))
    assert detail['entry']['id']==run_id
    assert detail['entry']['stages']['reconstruct']=='pending'


def test_rpc_forwards_run_options_and_disallows_external_upload(monkeypatch,tmp_path):
    from gsstudio.interfaces.worker import main
    recorded={}
    def create(root,location_id,scene_id,capture_id,*,reconstruction_backend='realityscan',
               projection_size=1746,mask_review_gate=False,sink=None,vision_qa=False):
        recorded.update(reconstruction_backend=reconstruction_backend,projection_size=projection_size,
                        mask_review_gate=mask_review_gate);return 'run'
    monkeypatch.setattr(main.ops,'create_and_preprocess',create)
    result=main.execute(_request(tmp_path,'create_run',advance=False,reconstruction_backend='colmap',projection_size=1024))
    assert result['run_id']=='run' and recorded==dict(reconstruction_backend='colmap',projection_size=1024,mask_review_gate=True)
    with pytest.raises(ValueError,match='authorization'):
        main.execute(_request(tmp_path,'create_run',vision_qa=True))

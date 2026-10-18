"""Optional donor-style COLMAP reconstruction within the canonical Run gates.

Original reviewed images/masks stay immutable. Undistorted pairs have their own
hash manifest; packages consume those pairs with the exported pinhole cameras.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import uuid
import cv2
import numpy as np
from gsstudio.infrastructure.adapters.image_io import read_cv_image
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.runtime.configuration import runtime_config
from gsstudio.infrastructure.runtime.processes import run_logged
from gsstudio.pipeline.masks.masking import atomic_imwrite
from gsstudio.pipeline.reconstruction.core import _write_colmap_binary_model, attach_frame_masks
from gsstudio.pipeline.training.data import json_write


def media_root(dataset: Path) -> Path:
    audit = dataset/'colmap-undistortion.json'
    if not audit.is_file():
        return dataset
    record=json.loads(audit.read_text(encoding='utf-8'))
    if record.get('schema_version')!=1 or record.get('status')!='complete':
        raise RuntimeError('COLMAP undistortion is incomplete')
    root=(dataset/record['media_root']).resolve()
    if not root.is_relative_to(dataset.resolve()):
        raise RuntimeError('Unsafe undistorted media root')
    for item in record['files']:
        path=(dataset/item['path']).resolve()
        if not path.is_relative_to(dataset.resolve()) or sha256_file(path)!=item['sha256']:
            raise RuntimeError('COLMAP undistortion source/output identity changed')
    return root


def _undistort_mask(mask, old, new):
    model,p=old.model,np.asarray(old.params,dtype=float)
    if model in {'SIMPLE_PINHOLE','SIMPLE_RADIAL','RADIAL'}:
        f,cx,cy=p[:3]; K=np.array([[f,0,cx],[0,f,cy],[0,0,1.]])
        distortion=np.zeros(5)
        if model=='SIMPLE_RADIAL': distortion[0]=p[3]
        if model=='RADIAL': distortion[:2]=p[3:5]
    elif model in {'PINHOLE','OPENCV','FULL_OPENCV'}:
        fx,fy,cx,cy=p[:4]; K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
        distortion=p[4:] if model!='PINHOLE' else np.zeros(5)
    else:
        raise ValueError(f'Unsupported COLMAP mask undistortion camera: {model}')
    fx,fy,cx,cy=new.params
    target=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    a,b=cv2.initUndistortRectifyMap(K,distortion,None,target,(new.width,new.height),cv2.CV_32FC1)
    return cv2.remap(mask,a,b,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def run_colmap_alignment(dataset, attempt, log_dir, settings, *, included_images,
                         projected_image_count=None):
    from gsstudio._vendor.nerfstudio.colmap_parsing_utils import read_cameras_binary,read_images_binary,read_points3D_binary
    from gsstudio._vendor.nerfstudio.colmap_utils import colmap_to_json
    from gsstudio.pipeline.reconstruction.realityscan import realityscan_reconstruction_metrics
    import shutil
    colmap=runtime_config().get('colmap') or shutil.which('colmap')
    if not colmap or not Path(colmap).is_file():
        raise RuntimeError('Configure the isolated COLMAP binary in GSSTUDIO_RUNTIME')
    names=sorted(included_images)
    if len(names)<2:
        raise ValueError('COLMAP requires at least two reviewed images')
    audit=dataset/'alignment-masks.json'
    records=[{'image':name,'image_sha256':sha256_file(dataset/'images'/name),
              'mask_sha256':sha256_file(dataset/'masks'/f'{name}.png')} for name in names]
    identity=canonical_hash({'records':records,'settings':settings.model_dump(mode='json')})
    if (dataset/'colmap-undistortion.json').is_file():
        previous=json.loads(audit.read_text(encoding='utf-8'))
        if previous.get('input_identity')!=identity:
            raise RuntimeError('COLMAP alignment identity changed')
        media_root(dataset)
        return realityscan_reconstruction_metrics(dataset,attempt,settings,included_images=set(names),
                                                  projected_image_count=projected_image_count)
    # Each failed invocation gets fresh databases/model paths, never stale output success.
    work=dataset/'colmap-work'/uuid.uuid4().hex
    sparse=work/'sparse'; sparse.mkdir(parents=True)
    log_dir.mkdir(parents=True,exist_ok=True)
    image_list=work/'input-images.txt'
    image_list.write_text('\n'.join(names)+'\n',encoding='utf-8')
    database=work/'database.db'
    gpu='1' if settings.use_gpu_sift else '0'
    # These are the current donor COLMAP option names, supplied by its tool binary.
    features=[colmap,'feature_extractor','--database_path',database,'--image_path',dataset/'images',
              '--image_list_path',image_list,'--ImageReader.camera_model',
              'PINHOLE' if attempt.panorama else 'SIMPLE_RADIAL',
              '--ImageReader.mask_path',dataset/'masks','--FeatureExtraction.use_gpu',gpu]
    if attempt.panorama:
        size=attempt.panorama.size
        f=size/(2*math.tan(math.radians(attempt.projection_fov_degrees)/2))
        features+=['--ImageReader.single_camera_per_folder','1','--ImageReader.camera_params',
                   f'{f},{f},{size/2},{size/2}']
    stages=[('features',features),('matching',[colmap,'exhaustive_matcher','--database_path',database,
                                             '--FeatureMatching.use_gpu',gpu]),
            ('mapping',[colmap,'mapper','--database_path',database,'--image_path',dataset/'images',
                        '--output_path',sparse])]
    if settings.fix_intrinsics:
        stages[-1][1].extend(['--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_principal_point','0'])
    json_write(audit,{'completed':False,'backend':'colmap','input_identity':identity,
                     'polarity':'white_keep','records':records,'commands':[[str(x) for x in c] for _,c in stages]})
    for name,command in stages:
        run_logged(command,log_dir/f'colmap-{name}.log')
    models=[(len(read_images_binary(f)),f.parent) for f in sparse.glob('*/images.bin')]
    if not models:
        raise RuntimeError('COLMAP produced no registered model; inspect overlap and logs')
    _,selected=max(models,key=lambda x:(x[0],str(x[1])))
    undistorted=work/'undistorted'
    run_logged([colmap,'image_undistorter','--image_path',dataset/'images','--input_path',selected,
                '--output_path',undistorted,'--output_type','COLMAP'],log_dir/'colmap-undistort.log')
    old_cameras=read_cameras_binary(selected/'cameras.bin')
    old_images=read_images_binary(selected/'images.bin')
    cameras=read_cameras_binary(undistorted/'sparse'/'cameras.bin')
    images=read_images_binary(undistorted/'sparse'/'images.bin')
    points=read_points3D_binary(undistorted/'sparse'/'points3D.bin')
    for cid,camera in list(cameras.items()):
        if camera.model=='SIMPLE_PINHOLE':
            f,cx,cy=camera.params
            cameras[cid]=camera._replace(model='PINHOLE',params=np.array([f,f,cx,cy]))
        elif camera.model!='PINHOLE':
            raise RuntimeError('COLMAP undistorter did not produce pinhole cameras')
    if not images or any(im.name not in included_images for im in images.values()):
        raise RuntimeError('COLMAP registered inventory exceeds frozen mask inclusion')
    files=[]
    for iid,image in images.items():
        old_image=old_images[iid]
        source_mask=dataset/'masks'/f'{image.name}.png'
        mask=read_cv_image(str(source_mask),cv2.IMREAD_GRAYSCALE)
        old_camera=old_cameras[old_image.camera_id]
        if mask is None or mask.shape!=(old_camera.height,old_camera.width):
            raise RuntimeError('Source mask and COLMAP camera dimensions differ')
        target=undistorted/'masks'/f'{image.name}.png'
        target.parent.mkdir(parents=True,exist_ok=True)
        atomic_imwrite(target,_undistort_mask(mask,old_camera,cameras[image.camera_id]))
        for path in (dataset/'images'/image.name,source_mask,undistorted/'images'/image.name,target):
            files.append({'path':path.relative_to(dataset).as_posix(),'sha256':sha256_file(path)})
    colmap_root=dataset/'colmap'
    colmap_root.mkdir(exist_ok=True)
    _write_colmap_binary_model(colmap_root,cameras,images,points)
    for name in ('cameras.bin','images.bin','points3D.bin'):
        p=colmap_root/name
        files.append({'path':p.relative_to(dataset).as_posix(),'sha256':sha256_file(p)})
    colmap_to_json(colmap_root,dataset,use_single_camera_mode=False)
    attach_frame_masks(dataset/'transforms.json',dataset/'masks')
    json_write(colmap_root/'selected-attempt.json',{'model':'.','rig':{'enabled':False,'backend':'colmap'},
               'component_sizes':[x[0] for x in models]})
    json_write(dataset/'colmap-undistortion.json',{'schema_version':1,'status':'complete',
               'input_identity':identity,'media_root':undistorted.relative_to(dataset).as_posix(),'files':files})
    payload=json.loads(audit.read_text(encoding='utf-8'));payload['completed']=True
    payload['verification']='paired_binary_masks_and_image_list_supplied_to_feature_extractor'
    json_write(audit,payload)
    metrics=realityscan_reconstruction_metrics(dataset,attempt,settings,included_images=set(names),
                                              projected_image_count=projected_image_count)
    metrics.update(backend='colmap',component_count=len(models),component_sizes=[x[0] for x in models])
    return metrics

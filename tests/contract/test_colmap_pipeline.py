"""CPU simulated tool boundary; real model conversion, masks and hash QA."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import struct
import numpy as np
from PIL import Image
import pytest
from gsstudio.domain.models import ReconstructionConfig
from gsstudio.pipeline.reconstruction import colmap
from gsstudio.pipeline.reconstruction.core import _write_colmap_binary_model


def test_colmap_path_masks_undistortion_and_reuse_identity(tmp_path,monkeypatch):
    from gsstudio._vendor.nerfstudio.colmap_parsing_utils import Camera,Image as CameraImage,Point3D
    dataset=tmp_path/'dataset';dataset.mkdir()
    (dataset/'images').mkdir();(dataset/'masks').mkdir()
    names={f'frame_{i:06d}.jpg' for i in (1,2,3)}
    for name in names:
        Image.new('RGB',(16,12),(100,100,100)).save(dataset/'images'/name)
        Image.new('L',(16,12),255).save(dataset/'masks'/f'{name}.png')
    old=Camera(id=1,model='SIMPLE_RADIAL',width=16,height=12,params=np.array([10.,8.,6.,.01]))
    new=Camera(id=1,model='PINHOLE',width=16,height=12,params=np.array([10.,10.,8.,6.]))
    images={i:CameraImage(id=i,qvec=np.array([1.,0,0,0]),tvec=np.array([i*.1,0,0]),camera_id=1,
                    name=f'frame_{i:06d}.jpg',xys=np.array([[2.,2.],[3.,3.],[4.,4.]]),
                    point3D_ids=np.array([1,2,3])) for i in (1,2,3)}
    points={i:Point3D(id=i,xyz=np.array([i*.1,0.,3.]),rgb=np.array([100,100,100],np.uint8),
                    error=0.,image_ids=np.array([1,2,3]),point2D_idxs=np.array([i-1]*3)) for i in (1,2,3)}
    executable=tmp_path/'colmap.exe';executable.touch()
    monkeypatch.setattr(colmap,'runtime_config',lambda:{'colmap':str(executable)})
    calls=[]
    def tool(command,log):
        calls.append([str(x) for x in command]);kind=command[1]
        if kind=='mapper':
            output=Path(command[command.index('--output_path')+1])/'0'
            _write_colmap_binary_model(output,{1:new},images,points)
            (output/'cameras.bin').write_bytes(struct.pack('<QiiQQ4d',1,old.id,2,old.width,old.height,*old.params))
        if kind=='image_undistorter':
            output=Path(command[command.index('--output_path')+1]);output.mkdir()
            shutil.copytree(dataset/'images',output/'images')
            _write_colmap_binary_model(output/'sparse',{1:new},images,points)
        return {}
    monkeypatch.setattr(colmap,'run_logged',tool)
    config=ReconstructionConfig(backend='colmap',primary={'temporal_rank_limit':1,
                                'projection_fov_degrees':84.,'panorama':None})
    before={p:colmap.sha256_file(p) for folder in ('images','masks') for p in (dataset/folder).iterdir()}
    result=colmap.run_colmap_alignment(dataset,config.primary,tmp_path/'logs',config,included_images=names)
    assert result['registered_images']==3 and result['backend']=='colmap'
    assert [x[1] for x in calls]==['feature_extractor','exhaustive_matcher','mapper','image_undistorter']
    assert '--ImageReader.mask_path' in calls[0]
    assert '--image_list_path' in calls[0]
    media=colmap.media_root(dataset)
    assert media!=dataset and (media/'masks'/f'{sorted(names)[0]}.png').is_file()
    assert all(colmap.sha256_file(p)==digest for p,digest in before.items())
    colmap.run_colmap_alignment(dataset,config.primary,tmp_path/'logs',config,included_images=names)
    assert len(calls)==4
    (media/'images'/sorted(names)[0]).write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='identity changed'): colmap.media_root(dataset)


def test_undistorted_mask_remains_binary_and_invalid_border_ignored():
    old=SimpleNamespace(model='SIMPLE_RADIAL',params=[20,10,8,.1])
    new=SimpleNamespace(width=20,height=16,params=[20,20,10,8])
    mask=np.full((16,20),255,np.uint8);mask[7:9,9:11]=0
    result=colmap._undistort_mask(mask,old,new)
    assert set(np.unique(result))<={0,255} and (result==0).any()

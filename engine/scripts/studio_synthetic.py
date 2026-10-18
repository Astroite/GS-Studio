"""CPU-only synthetic immutable GS package fixture, also used by GPU acceptance."""
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
from local_postshot.project import digest,atomic_json


def rehash(root,meta=None):
    root = Path(root)
    if meta is None:
        meta = json.loads((root/'dataset.json').read_text())
    meta['files'] = {p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob('*'))
                     if p.is_file() and p.name!='dataset.json'}
    atomic_json(root/'dataset.json',meta)
    return meta


def make_package(root,size=32):
    root = Path(root)
    for name in ('images','masks','colmap'):
        (root/name).mkdir(parents=True,exist_ok=True)
    xyz = np.array([[x,y,z] for z in (3.,3.5) for x,y in ((-.3,-.2),(.3,-.2),(.3,.2),(-.3,.2))],np.float32)
    rgb = np.tile(np.array([80,100,140],np.uint8),(len(xyz),1))
    K = np.array([[size*.8,0,size/2],[0,size*.8,size/2],[0,0,1.]])
    rows = []
    observations = []
    for i in range(6):
        pose = np.eye(4)
        pose[0,3] = -(i//2)*.05
        image,mask = f'images/{i:08d}.png',f'masks/{i:08d}.png'
        Image.new('RGB',(size,size),tuple(int(c) for c in rgb[0])).save(root/image)
        pixels = np.full((size,size),255,np.uint8)
        pixels[:4,:4] = 0
        Image.fromarray(pixels).save(root/mask)
        rows.append({'image':image,'mask':mask,'source_image':f'frame{i//2}-view{i%2}.png',
                     'frame':i//2,'timestamp_seconds':float(i//2),'split':'train' if i<4 else 'validation',
                     'width':size,'height':size,'K':K.tolist(),'world_to_camera':pose.tolist()})
        cam_xyz = xyz@pose[:3,:3].T+pose[:3,3]
        xy = (cam_xyz@K.T)[:,:2]/cam_xyz[:,2:3]
        observations.append(xy)
    def write(f,fmt,*values):
        f.write(struct.pack('<'+fmt,*values))
    with (root/'colmap/cameras.bin').open('wb') as f:
        write(f,'Q',1)
        write(f,'iiQQ4d',1,1,size,size,K[0,0],K[1,1],K[0,2],K[1,2])
    with (root/'colmap/images.bin').open('wb') as f:
        write(f,'Q',4)
        for i in range(4):
            t = np.array(rows[i]['world_to_camera'])[:3,3]
            write(f,'i7di',i+1,1.,0.,0.,0.,*t,1)
            f.write((Path(rows[i]['image']).name+'\0').encode())
            write(f,'Q',len(xyz))
            for j,(x,y) in enumerate(observations[i]):
                write(f,'ddq',x,y,j+1)
    with (root/'colmap/points3D.bin').open('wb') as f:
        write(f,'Q',len(xyz))
        for j,(p,c) in enumerate(zip(xyz,rgb)):
            write(f,'Q3d3BdQ',j+1,*p,*c,0.,4)
            for i in range(4):
                write(f,'ii',i+1,j)
    # Deliberately opposite to COLMAP ID order.
    np.savez(root/'points.npz',xyz=xyz[::-1].copy(),rgb=rgb[::-1].copy())
    row_index = np.repeat(np.arange(4,dtype=np.int32),len(xyz))
    xy = np.rint(np.concatenate(observations[:4])).astype(np.float32)
    depth = np.tile(xyz[:,2],4)
    np.savez(root/'sparse_depth.npz',row_index=row_index,u=xy[:,0],v=xy[:,1],depth=depth,
             point_id=np.tile(np.arange(1,len(xyz)+1,dtype=np.int64),4))
    meta = {'schema_version':1,'validation':'passed','coordinates':'COLMAP','mask_polarity':'white_keep',
            'initialization_colors':'unmasked_training_observations_only','initial_points':len(xyz),
            'sparse_depth_observations':len(depth),'images':rows}
    rehash(root,meta)
    return root

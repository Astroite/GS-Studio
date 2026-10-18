"""Synthetic native CUDA integration check; never accesses a capture or Run."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
import torch

from gsstudio.pipeline.training import native
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.pipeline.training.data import json_write

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
native.configure_windows_cuda()
root=args.output.resolve()
package=root/'synthetic-data'
package.mkdir(parents=True,exist_ok=False)
(package/'images').mkdir()
(package/'masks').mkdir()
rng=np.random.default_rng(7)
xyz=rng.uniform([-1,-1,2],[1,1,4],(100,3)).astype(np.float32)
rgb=rng.uniform(0.1,0.9,(100,3)).astype(np.float32)
np.savez(package/'points.npz',xyz=xyz,rgb=(rgb*255).astype(np.uint8))
params=native.initialize(package)
rows=[]
for i in range(16):
    matrix=np.eye(4)
    matrix[0,3]=(i-8)*0.03
    row=dict(image=f'images/{i}.png',mask=f'masks/{i}.png',width=64,height=64,
        K=[[50.,0,32],[0,50.,32],[0,0,1]],world_to_camera=matrix.tolist(),frame=i,
        timestamp_seconds=i*0.5,split='validation' if i%8==7 else 'train')
    pixels,_,_=native.render(params,row)
    assert pixels.is_cuda and torch.isfinite(pixels).all()
    pixels.sum().backward()
    assert params['means'].grad is not None and torch.isfinite(params['means'].grad).all()
    for p in params.values():
        p.grad=None
    cv2.imwrite(str(package/row['image']),(pixels[0].detach().cpu().numpy()[...,::-1].clip(0,1)*255).astype(np.uint8))
    cv2.imwrite(str(package/row['mask']),np.full((64,64),255,dtype=np.uint8))
    rows.append(row)
del params,pixels
torch.cuda.empty_cache()
json_write(package/'dataset.json',dict(schema_version=1,validation='passed',synthetic=True,images=rows,
    files={p.relative_to(package).as_posix():sha256_file(p) for p in package.rglob('*') if p.is_file()}))
native.train(package,root/'training',steps=4,checkpoint_every=2)
before=sha256_file(root/'training'/'model.ply')
native.train(package,root/'training',steps=4,resume=True,checkpoint_every=2)
assert sha256_file(root/'training'/'model.ply')==before
original_render=native.render
calls=0
def interrupted_render(*args,**kwargs):
    global calls
    calls+=1
    if calls==4:
        raise RuntimeError('intentional smoke interruption')
    return original_render(*args,**kwargs)
native.render=interrupted_render
try:
    native.train(package,root/'interrupted',steps=4,checkpoint_every=2)
except RuntimeError as error:
    assert str(error)=='intentional smoke interruption'
finally:
    native.render=original_render
assert torch.load(root/'interrupted'/'checkpoint.pt')['step']==2
native.train(package,root/'interrupted',steps=4,resume=True,checkpoint_every=2)
reference=native.load_ply(root/'training'/'model.ply')
resumed=native.load_ply(root/'interrupted'/'model.ply')
with torch.no_grad():
    render_a,_,_=native.render(reference,rows[0])
    render_b,_,_=native.render(resumed,rows[0])
    resume_render_error=float((render_a-render_b).abs().max())
assert resume_render_error<1e-4
for name in reference:
    assert torch.allclose(reference[name],resumed[name],atol=1e-5,rtol=1e-4),name
json_write(root/'result.json',dict(status='passed',cuda_forward_backward=True,checkpoint_resume=True,ply_roundtrip=True,
    interrupted_resume_matches_uninterrupted=True,resume_render_max_error=resume_render_error,
    torch=torch.__version__,device=torch.cuda.get_device_name(),synthetic_only=True))
print((root/'result.json').read_text(),flush=True)

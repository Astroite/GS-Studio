"""Measure native checkpoint/PLY invariants and unclipped render roundtrip error."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from gsstudio.pipeline.training.native import configure_windows_cuda,load_ply,render
from gsstudio.pipeline.training.data import json_write

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',type=Path,required=True)
parser.add_argument('--dataset',type=Path,required=True)
args=parser.parse_args()
configure_windows_cuda()
saved=torch.load(args.experiment/'checkpoint.pt',map_location='cuda')['params']
loaded=load_ply(args.experiment/'model.ply')
meta=json.loads((args.dataset/'dataset.json').read_text(encoding='utf-8'))
row=next(r for r in meta['images'] if r['split']=='train')
with torch.no_grad():
    a,_,_=render(saved,row)
    b,_,_=render(loaded,row)
    diff=(a-b).abs()
    flat=diff.flatten()
    # torch.quantile rejects tensors above 2^24 elements and a 4K render is 24.9M.
    # NumPy interpolates identically and has no such limit; a strided sample would
    # alias against the three channels.
    p999_error=float(np.quantile(diff.detach().cpu().numpy(),0.999))
    result=dict(max_error=float(diff.max()),mean_error=float(diff.mean()),rms_error=float(diff.square().mean().sqrt()),
        p999_error=p999_error,fraction_over_quarter_8bit_level=float((diff>1/(4*255)).float().mean()),
        pixels_over_quarter_8bit_level=int((diff.amax(dim=2)>1/(4*255)).sum()),
        pixels_over_half_8bit_level=int((diff.amax(dim=2)>0.5/255).sum()),
        pixels_over_one_8bit_level=int((diff.amax(dim=2)>1/255).sum()),
        channels_over_one_8bit_level=int((flat>1/255).sum()),
        normalized_quaternion_max_error=float((F.normalize(saved['quats'],dim=1)-F.normalize(loaded['quats'],dim=1)).abs().max()),
        exact_parameter_equality={k:torch.equal(saved[k],loaded[k]) for k in saved if k!='quats'})
json_write(args.experiment/'export-diagnostic.json',result)
print(json.dumps(result,indent=2),flush=True)

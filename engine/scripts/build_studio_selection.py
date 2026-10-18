"""Compile editor-only helper offline; explicit architecture avoids GPU queries."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture',default='8.9')
    args=parser.parse_args()
    os.environ['TORCH_CUDA_ARCH_LIST']=args.architecture
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import torch
    from local_postshot.studio_selection import build_location
    from torch.utils.cpp_extension import load
    source,output=build_location()
    output.mkdir(parents=True,exist_ok=True)
    load(name='studio_selection_cuda',sources=[str(source/'selection.cpp'),str(source/'selection.cu')],
         build_directory=str(output),extra_cflags=['/O2'] if os.name=='nt' else ['-O3'],
         extra_cuda_cflags=['-O3'],verbose=True)
    assert not torch.cuda.is_initialized(),'Offline compilation must not initialize CUDA'
    proof={'output':str(output),'architecture':args.architecture,'cuda_initialized':False}
    (output/'build-proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof))


if __name__=='__main__':main()

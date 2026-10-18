"""Literal scalar CPU interpretation of bounded AA PTX arithmetic blocks.

This is not execution of the product kernel: loads, stores, branches, GPU state
and all unlisted instructions are rejected. Float32 uses the system fmaf for
one correctly rounded fused instruction, plus explicit FTZ for .ftz opcodes.
"""
from pathlib import Path
import ctypes
import hashlib
import math
import re
import struct
import numpy as np
import os
import pytest

PTX=Path(os.environ.get('LOCAL_POSTSHOT_AA_PTX',str(Path(__file__).parent/'init-loss-30127-gpu.ptx')))
if not PTX.is_file():
    pytest.skip('Frozen AA PTX audit evidence not bundled; set LOCAL_POSTSHOT_AA_PTX to read-only evidence',allow_module_level=True)
PTX_SHA='4e82a0083b23fb54643172c760e7dc2fd941b690ee8853c2712322bdf9d209c2'
EPS=float(np.float32(1e-6))
BLUR=float(np.float32(.3))
LINES=PTX.read_text(encoding='utf-8').splitlines()
assert hashlib.sha256(PTX.read_bytes()).hexdigest()==PTX_SHA
FMA=ctypes.CDLL('ucrtbase').fmaf
FMA.argtypes=[ctypes.c_float]*3
FMA.restype=ctypes.c_float


def scalar_block(first,last,registers,*,binary32=False):
    r=dict(registers)
    cast=np.float32 if binary32 else float
    def value(x):
        if x.startswith('%'):return cast(r[x])
        if x.startswith('0f'):return cast(struct.unpack('<f',bytes.fromhex(x[2:])[::-1])[0])
        raise AssertionError('Unsupported operand '+x)
    def ftz(x):
        return math.copysign(0.,x) if 0<abs(x)<np.finfo(np.float32).tiny else x
    for lineno in range(first,last+1):
        line=LINES[lineno-1].strip()
        match=re.fullmatch(r'(\S+)\s+(%f\d+),\s*(.*);',line)
        assert match, (lineno,line)
        opcode,dest,operands=match.groups()
        args=[value(x.strip()) for x in operands.split(',')]
        if binary32 and '.ftz' in opcode:args=[cast(ftz(x)) for x in args]
        name=opcode.split('.')[0]
        if name=='fma':out=FMA(*args) if binary32 else args[0]*args[1]+args[2]
        elif name=='mul':out=args[0]*args[1]
        elif name=='add':out=args[0]+args[1]
        elif name=='div':out=args[0]/args[1]
        elif name=='max':out=max(args)
        elif name=='sqrt':out=math.sqrt(args[0])
        elif name=='neg':out=-args[0]
        elif name=='mov':out=args[0]
        else:raise AssertionError((lineno,opcode))
        out=cast(out)
        if binary32 and '.ftz' in opcode:out=cast(ftz(out))
        r[dest]=out
    return r


def forward(raw,*,binary32=False):
    a,b,c=raw
    r=scalar_block(35712,35714,{'%f67':a,'%f68':b,'%f69':c},binary32=binary32)
    r=scalar_block(35718,35726,r,binary32=binary32)
    return dict(compensation=float(r['%f760']),numerator=float(r['%f470']),denominator=float(r['%f471']))


def vjp(raw,upstream=1.,*,binary32=False):
    a,b,c=raw
    cast=np.float32 if binary32 else float
    # i_Od9 repeats the forward operations with different register IDs.
    r=scalar_block(139629,139631,{'%f74':a,'%f75':b,'%f76':c},binary32=binary32)
    r=scalar_block(139635,139644,r,binary32=binary32)
    r.update({'%f92':cast(upstream),'%f990':cast(0),'%f991':cast(0),'%f992':cast(0)})
    r=scalar_block(139702,139716,r,binary32=binary32)
    return np.array([r['%f990'],r['%f991'],r['%f992']],dtype=np.float32 if binary32 else np.float64)


def analytic(raw,upstream=1.):
    a,b,c=raw
    f=forward(raw)
    n,d,comp=f['numerator'],f['denominator'],f['compensation']
    gn=upstream/(2*comp*d)
    gd=-upstream*n/(2*comp*d*d)
    return np.array([gn*c+gd*(c+BLUR),-2*b*(gn+gd),gn*a+gd*(a+BLUR)])


def public_vjp(raw,upstream=1.):
    """Scalar shared-b mapping of installed gsplat Utils.cuh:413..422."""
    a,b,c=raw
    d=(a+BLUR)*(c+BLUR)-b*b
    comp=math.sqrt(max(0.,(a*c-b*b)/d))
    q=np.array([[c+BLUR,-b],[-b,a+BLUR]])/d
    factor=upstream*.5/(comp+1e-6)
    v=factor*((1-comp*comp)*q-BLUR*np.linalg.det(q)*np.eye(2))
    return comp,np.array([v[0,0],v[0,1]+v[1,0],v[1,1]])

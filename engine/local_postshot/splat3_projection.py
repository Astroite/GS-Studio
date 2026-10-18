"""Optional private native-floor AA on public gsplat 1.5.3 geometry.

Single pinhole camera, unpacked CUDA float32, normalized public wxyz quaternion
convention. Only AA compensation is recovered; geometry/bbox/VJP remain public.
No CPU fallback. The classic path must continue to use its frozen old module.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import importlib
import importlib.metadata
import torch


@lru_cache(maxsize=1)
def _extension():
    try:
        module=importlib.import_module('local_splat3_projection')
    except (ImportError,OSError) as error:
        raise RuntimeError('Private native-floor projection unavailable; no public fallback is permitted') from error
    if getattr(module,'implementation_version',None)!='native_floor_v1':
        raise RuntimeError('Private projection differs from native_floor_v1')
    return module


def projection_identity():
    """Read provenance without importing the native module or initializing CUDA."""
    try:
        dist=importlib.metadata.distribution('local-splat3-projection')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('Private native-floor projection distribution is not installed') from error
    binaries=[Path(dist.locate_file(f)) for f in dist.files if str(f).endswith('.pyd')]
    if len(binaries)!=1:
        raise RuntimeError('Private projection must contain exactly one native binary')
    return {'distribution':'local-splat3-projection','version':dist.version,
            'binary_sha256':hashlib.sha256(binaries[0].read_bytes()).hexdigest()}


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means,quats,scales,opacities,viewmats,Ks,width,height,eps2d,near,far,radius_clip):
        radii,means2d,depths,conics,comp=_extension().forward(means,quats,scales,opacities,viewmats,Ks,
                                                          width,height,eps2d,near,far,radius_clip)
        ctx.save_for_backward(means,quats,scales,viewmats,Ks,radii,conics,comp)
        ctx.width,ctx.height,ctx.eps2d=width,height,eps2d
        ctx.mark_non_differentiable(radii)
        ctx.set_materialize_grads(False)
        return radii,means2d,depths,conics,comp

    @staticmethod
    def backward(ctx,_radii,gm,gz,gc,ga):
        means,quats,scales,viewmats,Ks,radii,conics,comp=ctx.saved_tensors
        n=len(means)
        if gm is None:gm=means.new_zeros((1,n,2))
        if gz is None:gz=means.new_zeros((1,n))
        if gc is None:gc=torch.zeros_like(conics)
        if ga is None:ga=torch.zeros_like(comp)
        gradients=_extension().backward(means,quats,scales,viewmats,Ks,radii,conics,comp,
            gm.contiguous(),gz.contiguous(),gc.contiguous(),ga.contiguous(),ctx.width,ctx.height,ctx.eps2d)
        return gradients[0],gradients[1],gradients[2],None,None,None,None,None,None,None,None,None


def fully_fused_projection(means, covars, quats, scales, viewmats, Ks, width, height,
                           eps2d=.3, near_plane=.01, far_plane=1e8, radius_clip=0.,
                           packed=False, sparse_grad=False, calc_compensations=True,
                           camera_model='pinhole', opacities=None):
    """Checked subset of the gsplat API; opacities affect support, not geometry.

    Caller multiplies returned compensation into activated opacity, preserving
    its ordinary gradient. Bbox/culling are non-differentiable as in gsplat.
    """
    if covars is not None or packed or sparse_grad or camera_model!='pinhole' or not calc_compensations:
        raise ValueError('Private AA projection requires quats/scales, unpacked pinhole, dense gradients and AA')
    if opacities is None:
        raise ValueError('Explicit activated opacity is required for support')
    if viewmats.requires_grad or Ks.requires_grad:
        raise ValueError('Private AA projection supports fixed camera calibration only')
    if not means.is_cuda or means.dtype!=torch.float32:
        raise ValueError('Private AA projection requires CUDA float32')
    tensors=[v.contiguous() for v in (means,quats,scales,opacities,viewmats,Ks)]
    return _Projection.apply(*tensors,int(width),int(height),float(eps2d),float(near_plane),float(far_plane),float(radius_clip))

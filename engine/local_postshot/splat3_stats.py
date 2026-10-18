"""Independent contribution statistics using public gsplat raster semantics.

The native backend is explicitly versioned and never falls back to CPU/Python.
Inputs are detached forward-pass means/conics/activated opacity and its original
depth-sorted tile metadata, BEFORE any optimizer/topology change. Caller must
provide finite parameters, valid tile lists, and finite nonnegative pixel error.
Public alpha cap.999 differs from the original observed.998046875. This is an
independent public rasterizer replacement, not a bitwise original reconstruction.
CUDA floating-point atomic sums are nondeterministic; max_weight is an exact max
of the computed float32 contributions. No autograd is defined for these outputs.
"""
from __future__ import annotations
from typing import NamedTuple
import importlib
import math


def native_stats_identity():
    """Portable binary identity; absolute install paths are not checkpoint data."""
    import hashlib
    from importlib.metadata import version
    from pathlib import Path
    # Register Torch's bundled DLL directories before importing its extension.
    # Importing Torch does not initialize a CUDA context.
    import torch
    module=importlib.import_module("local_splat3_stats")
    return {"distribution":"local-splat3-stats","version":version("local-splat3-stats"),
            "binary_sha256":hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}


def contribution_identity(recipe):
    if recipe.get("contribution_backend")=="native_renderer_v1":
        from .splat3_renderer import renderer_identity
        return renderer_identity()
    return native_stats_identity()


class ContributionStats(NamedTuple):
    weighted_error_sum: object
    max_weight: object
    sum_weight: object


def _validate(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size):
    import torch
    if (isinstance(width,bool) or isinstance(height,bool) or isinstance(tile_size,bool)
            or int(width)!=width or int(height)!=height or int(tile_size)!=tile_size
            or width<=0 or height<=0 or not 1<=tile_size<=32):
        raise ValueError("Positive integer width/height and tile_size1..32 required")
    tensors=(means2d,conics,opacities,isect_offsets,flatten_ids,error_map)
    if not all(isinstance(t,torch.Tensor) for t in tensors):
        raise TypeError("All raster metadata and error_map inputs must be Torch tensors")
    if means2d.ndim!=3 or means2d.shape[0]!=1 or means2d.shape[-1]!=2:
        raise ValueError("means2d must have shape[1,N,2]")
    n=means2d.shape[1]
    if conics.shape!=(1,n,3) or opacities.shape not in ((n,),(1,n)):
        raise ValueError("conics[1,N,3] and opacities[N] or[1,N] required")
    if (isect_offsets.shape!=(1,(height+tile_size-1)//tile_size,(width+tile_size-1)//tile_size)
            or flatten_ids.ndim!=1 or error_map.shape!=(height,width)):
        raise ValueError("Invalid tile offsets, flatten_ids or error_map shape")
    if any(t.device!=means2d.device or not t.is_contiguous() for t in tensors):
        raise ValueError("All inputs must be contiguous and on the same device")
    if any(t.dtype!=torch.float32 for t in (means2d,conics,opacities,error_map)):
        raise ValueError("Float inputs must be float32")
    if any(t.dtype!=torch.int32 for t in (isect_offsets,flatten_ids)):
        raise ValueError("Tile offsets and flatten_ids must be int32")


def reference_statistics(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size):
    """Explicit CPU float64 oracle. Small scenes only; never a production fallback."""
    import numpy as np
    import torch
    _validate(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size)
    if means2d.device.type!="cpu":
        raise ValueError("Reference requires explicit CPU input copies")
    xy=means2d.detach().numpy().astype(np.float64)[0]
    co=conics.detach().numpy().astype(np.float64)[0]
    opacity=opacities.detach().numpy().astype(np.float64).reshape(-1)
    offsets=isect_offsets.numpy().reshape(-1)
    ids=flatten_ids.numpy()
    error=error_map.detach().numpy().astype(np.float64)
    n=len(xy);sumerr=np.zeros(n,np.float64);maxw=sumerr.copy();sumw=sumerr.copy()
    tile_w=(width+tile_size-1)//tile_size
    for y in range(height):
        for x in range(width):
            tile=(y//tile_size)*tile_w+x//tile_size
            start=int(offsets[tile]);end=int(offsets[tile+1]) if tile+1<len(offsets) else len(ids)
            if not 0<=start<=end<=len(ids):
                continue
            transmittance=1.
            for index in range(start,end):
                g=int(ids[index])
                if not 0<=g<n:
                    continue
                dx,dy=xy[g]-[x+.5,y+.5]
                sigma=.5*(co[g,0]*dx*dx+co[g,2]*dy*dy)+co[g,1]*dx*dy
                if sigma<0:
                    continue
                alpha=min(.999,opacity[g]*math.exp(-sigma))
                if alpha<1/255:
                    continue
                next_t=transmittance*(1-alpha)
                if next_t<=1e-4:
                    break
                weight=transmittance*alpha
                sumerr[g]+=weight*error[y,x];maxw[g]=max(maxw[g],weight);sumw[g]+=weight
                transmittance=next_t
    return ContributionStats(*(torch.from_numpy(v) for v in (sumerr,maxw,sumw)))


def contribution_stats(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,
                       width,height,tile_size,*,backend="auto"):
    """Return three[N] statistics, with caller-selected frozen backend.

    native_cuda_v1 requires CUDA and the installed private wheel. native_cpu_v1
    requires CPU. auto resolves only between those native device paths. reference
    is an explicit CPU double oracle and must not be selected in production.
    No value-dependent CPU/GPU synchronization is used for input validation;
    valid depth order/list values and nonnegative finite error remain the caller's
    responsibility. Invalid list bounds are safely skipped, not repaired.
    """
    _validate(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size)
    if backend=="reference":
        return reference_statistics(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size)
    if backend=="auto":
        backend="native_cuda_v1" if means2d.is_cuda else "native_cpu_v1"
    if backend not in ("native_cuda_v1","native_cpu_v1","native"):
        raise ValueError("Unknown contribution statistics backend")
    if backend=="native_cuda_v1" and not means2d.is_cuda:
        raise ValueError("native_cuda_v1 requires CUDA inputs")
    if backend=="native_cpu_v1" and means2d.device.type!="cpu":
        raise ValueError("native_cpu_v1 requires CPU inputs")
    try:
        extension=importlib.import_module("local_splat3_stats")
    except (ImportError,OSError) as exc:
        raise RuntimeError("Private local_splat3_stats extension unavailable; install the pinned wheel. No CPU fallback is allowed.") from exc
    if getattr(extension,"implementation_version",None)!="native_cuda_v1":
        raise RuntimeError("Private statistics implementation version differs from the frozen native_cuda_v1 contract")
    values=extension.contribution_stats(*(v.detach() for v in (means2d,conics,opacities,isect_offsets,flatten_ids,error_map)),
                                        int(width),int(height),int(tile_size))
    return ContributionStats(*values)

"""Private compositor with frozen, explicitly selected projection/tile policies.

Compositor v1: alpha cap511/512, pixel centers+.5, alpha cutoff1/255, process
the current Gaussian whenever OLD T>=float32(1e-4), cap straight-through VJP.
Background is an explicit tensor; this module never consumes random numbers.
Old recipes retain gsplat1.5.3 projection/AA/bounding support. New policies can
select conservative/native-algebra tiles and native-floor AA over public
geometry. They do not establish bitwise equivalence to original PTX.
No original product binary, plugin or PTX is imported or executed.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from typing import NamedTuple

import torch

ALPHA_CAP=511/512
T_THRESHOLD=float(torch.tensor(1e-4,dtype=torch.float32))


class ContributionStats(NamedTuple):
    weighted_error_sum: object
    max_weight: object
    sum_weight: object


@lru_cache(maxsize=1)
def _extension():
    try:
        module=importlib.import_module('local_splat3_render')
    except (ImportError,OSError) as error:
        raise RuntimeError('Private local_splat3_render extension unavailable; no public/CPU fallback is permitted') from error
    if getattr(module,'implementation_version',None)!='native_renderer_v1':
        raise RuntimeError('Private renderer implementation differs from frozen native_renderer_v1')
    return module


def renderer_identity():
    """Artifact identity for frozen checkpoints; imports no CUDA extension."""
    try:
        dist=importlib.metadata.distribution('local-splat3-render')
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError('Private renderer distribution is not installed') from error
    binaries=[Path(dist.locate_file(f)) for f in dist.files if str(f).endswith('.pyd')]
    if len(binaries)!=1:
        raise RuntimeError('Private renderer must contain exactly one native binary')
    return {'distribution':'local-splat3-render','version':dist.version,
            'binary_sha256':hashlib.sha256(binaries[0].read_bytes()).hexdigest()}


class _CompositeRGB(torch.autograd.Function):
    @staticmethod
    def forward(ctx,means2d,conics,colors,opacities,background,offsets,ids,width,height,tile_size):
        image,alpha,transmittance,last=_extension().forward(means2d,conics,colors,opacities,background,offsets,ids,
                                                          int(width),int(height),int(tile_size))
        ctx.save_for_backward(means2d,conics,colors,opacities,background,offsets,ids,transmittance,last)
        ctx.width,ctx.height,ctx.tile_size=int(width),int(height),int(tile_size)
        ctx.set_materialize_grads(False)
        return image,alpha

    @staticmethod
    def backward(ctx,grad_image,grad_alpha):
        means,conics,colors,opacities,background,offsets,ids,trans,last=ctx.saved_tensors
        if grad_image is None:
            grad_image=torch.zeros((1,ctx.height,ctx.width,3),device=means.device,dtype=means.dtype)
        if grad_alpha is None:
            grad_alpha=torch.zeros_like(trans)
        gm,gc,grgb,go,gb=_extension().backward(means,conics,colors,opacities,background,offsets,ids,trans,last,
            grad_image.contiguous(),grad_alpha.contiguous(),ctx.width,ctx.height,ctx.tile_size)
        return gm,gc,grgb,go,gb,None,None,None,None,None


def rasterize_projected(means2d,conics,colors,opacities,isect_offsets,flatten_ids,
                        width,height,tile_size=16,*,background=None):
    """CUDA-only differentiable RGB3 compositor over one view's metadata.

    Float32 contiguous same-device inputs and valid sorted tile indices required.
    The native boundary validates metadata shapes/devices without GPU scalar
    synchronization. Background may be[3] or[1,3], defaults to black.
    """
    if not means2d.is_cuda:
        raise ValueError('Native RGB compositor requires CUDA; use explicit reference_projected for CPU audit')
    if background is None:
        background=torch.zeros((1,3),device=means2d.device,dtype=means2d.dtype)
    else:
        if background.shape not in ((3,),(1,3)):
            raise ValueError('Background must have shape[3] or[1,3]')
        background=background.reshape(1,3).contiguous()
    return _CompositeRGB.apply(means2d.contiguous(),conics.contiguous(),colors.contiguous(),opacities.contiguous(),
                               background,isect_offsets.contiguous(),flatten_ids.contiguous(),int(width),int(height),int(tile_size))


def contribution_stats(means2d,conics,opacities,isect_offsets,flatten_ids,error_map,width,height,tile_size):
    """No-grad same-cap/same-T statistics; never uses the old stats module.

    Caller guarantees finite nonnegative error_map and uses this exact forward's
    means/conics/compensated opacity/list before optimizer or topology mutation.
    Floating-point atomic/warp sums are not deterministic bitwise reductions.
    """
    values=_extension().contribution_stats(*(v.detach().contiguous() for v in
        (means2d,conics,opacities,isect_offsets,flatten_ids,error_map)),int(width),int(height),int(tile_size))
    return ContributionStats(*values)


def rasterize_parameters(params,camera,degree,*,camera_tensors=None,policy=None,training=False,background=None):
    """Required production API: one pinhole camera, RGB3, unpacked, SHdegree0..3.

    Existing projection_kwargs owns near/far/eps2d/classic-vs-AA policy. The
    compositor and statistics are private; geometry/SH use public gsplat math.
    Optional tile and AA implementations are selected by the frozen policy.
    """
    from gsplat.cuda._wrapper import fully_fused_projection,spherical_harmonics,isect_tiles,isect_offset_encode
    from .projection_policy import projection_kwargs
    if degree not in range(4):
        raise ValueError('Private compositor supports SHdegree0..3')
    means=params['means']
    if means.ndim!=2 or means.shape[1]!=3 or means.dtype!=torch.float32 or not means.is_cuda:
        raise ValueError('Private renderer requires CUDAfloat32 Nx3 means')
    n=len(means);width,height=int(camera['width']),int(camera['height']);tile=16
    view=(camera_tensors['w2c'] if camera_tensors is not None else torch.tensor(camera['w2c'],dtype=means.dtype,device=means.device)[None])
    K=(camera_tensors['K'] if camera_tensors is not None else torch.tensor(camera['K'],dtype=means.dtype,device=means.device)[None])
    if view.shape!=(1,4,4) or K.shape!=(1,3,3):
        raise ValueError('Private renderer supports exactly one pinhole view')
    settings=projection_kwargs(camera,policy,training=training)
    aa=settings.pop('rasterize_mode','classic')
    if aa not in ('classic','antialiased'):
        raise ValueError('Unsupported projection AA mode')
    opacity=params['opacities'].sigmoid()
    tile_culling=(policy or {}).get('tile_culling','public')
    aa_compensation=(policy or {}).get('aa_compensation','public')
    if aa=='antialiased' and aa_compensation=='native_floor_v1':
        from .splat3_projection import fully_fused_projection
    if n:
        radii,means2d,depths,conics,compensation=fully_fused_projection(means=means,covars=None,quats=params['quats'],
            scales=params['scales'].exp(),viewmats=view,Ks=K,width=width,height=height,
            packed=False,calc_compensations=(aa=='antialiased'),camera_model='pinhole',opacities=opacity,**settings)
        projected_opacity=opacity[None] if compensation is None else opacity[None]*compensation
        center=camera_tensors.get('center') if camera_tensors is not None else None
        if center is None:center=torch.inverse(view)[...,:3,3]
        directions=means[None]-center[:,None]
        coefficients=torch.cat([params['sh0'],params['shN']],dim=1)[None]
        colors=spherical_harmonics(degree,directions,coefficients,masks=(radii>0).all(-1))
        colors=(colors+.5).clamp_min(0)
        tile_w,tile_h=(width+tile-1)//tile,(height+tile-1)//tile
        if tile_culling=='public':
            tiles_per_gauss,isect_ids,flat_ids=isect_tiles(means2d,radii,depths,tile,tile_w,tile_h,packed=False)
            offsets=isect_offset_encode(isect_ids,1,tile_w,tile_h).reshape(1,tile_h,tile_w)
        else:
            from .splat3_tiles import intersect_tiles
            tiles_per_gauss,isect_ids,flat_ids,offsets=intersect_tiles(
                means2d,radii,depths,conics,projected_opacity,width,height,tile,mode=tile_culling)
    else:
        # Avoid launching a zero-grid public projection kernel.
        means2d=means[None,:,:2]
        conics=torch.empty((1,0,3),device=means.device,dtype=means.dtype)
        depths=torch.empty((1,0),device=means.device,dtype=means.dtype)
        radii=torch.empty((1,0,2),device=means.device,dtype=torch.int32)
        colors=params['sh0'][:,0][None]
        projected_opacity=opacity[None]
        tile_w,tile_h=(width+tile-1)//tile,(height+tile-1)//tile
        offsets=torch.zeros((1,tile_h,tile_w),device=means.device,dtype=torch.int32)
        flat_ids=torch.empty(0,device=means.device,dtype=torch.int32)
        isect_ids=torch.empty(0,device=means.device,dtype=torch.int64)
        tiles_per_gauss=torch.empty((1,0),device=means.device,dtype=torch.int32)
    image,alpha=rasterize_projected(means2d,conics,colors,projected_opacity,offsets,flat_ids,width,height,tile,background=background)
    info={'batch_ids':None,'camera_ids':None,'gaussian_ids':None,'radii':radii,'means2d':means2d,
        'depths':depths,'conics':conics,'opacities':projected_opacity,'tiles_per_gauss':tiles_per_gauss,
        'isect_ids':isect_ids,'flatten_ids':flat_ids,'isect_offsets':offsets,'tile_width':tile_w,'tile_height':tile_h,
        'width':width,'height':height,'tile_size':tile,'n_batches':1,'n_cameras':1,
        'compositor':'native_renderer_v1',
        'projection':'public_gsplat1.5.3_geometry_native_floor_AA' if aa=='antialiased' and aa_compensation=='native_floor_v1' else 'public_gsplat1.5.3',
        'projection_aa_floor':aa_compensation if aa=='antialiased' else 'disabled_classic',
        'tile_culling':tile_culling}
    return image,alpha,info


def reference_projected(means2d,conics,colors,opacities,isect_offsets,flatten_ids,width,height,tile_size=16,
                        *,background=None,error_map=None):
    """Explicit small CPU autograd oracle, including cap straight-through.

    Supports float32 or float64 CPU tensors. Decisions use detached values,
    matching nondifferentiable selection; this is not a production fallback.
    """
    if means2d.device.type!='cpu' or min(width,height,tile_size)<=0:
        raise ValueError('CPU oracle requires CPU tensors and positive dimensions')
    n=means2d.shape[1]
    opacity=opacities.reshape(-1)
    bg=torch.zeros(3,dtype=means2d.dtype) if background is None else background.reshape(3)
    error=torch.ones((height,width),dtype=means2d.dtype) if error_map is None else error_map
    sums=[means2d.new_zeros(()) for _ in range(n)]
    maxima=[means2d.new_zeros(()) for _ in range(n)]
    errors=[means2d.new_zeros(()) for _ in range(n)]
    tw=(width+tile_size-1)//tile_size;offsets=isect_offsets.reshape(-1)
    pixels=[];transmittances=[]
    for y in range(height):
        for x in range(width):
            tile=(y//tile_size)*tw+x//tile_size
            begin=int(offsets[tile]);end=int(offsets[tile+1]) if tile+1<len(offsets) else len(flatten_ids)
            T=means2d.new_ones(());pixel=means2d.new_zeros(3)
            for index in range(begin,end):
                if float(T.detach())<T_THRESHOLD:break
                g=int(flatten_ids[index]);dx=means2d[0,g,0]-(x+.5);dy=means2d[0,g,1]-(y+.5)
                sigma=.5*(conics[0,g,0]*dx.square()+conics[0,g,2]*dy.square())+conics[0,g,1]*dx*dy
                if float(sigma.detach())<0:continue
                raw=opacity[g]*(-sigma).exp()
                alpha=raw+(raw.clamp_max(ALPHA_CAP)-raw).detach()
                if float(alpha.detach())<1/255:continue
                weight=T*alpha;pixel=pixel+weight*colors[0,g];T=T*(1-alpha)
                sums[g]=sums[g]+weight.detach();errors[g]=errors[g]+weight.detach()*error[y,x].detach()
                maxima[g]=torch.maximum(maxima[g],weight.detach())
            pixels.append(pixel+T*bg);transmittances.append(T)
    image=torch.stack(pixels).reshape(1,height,width,3)
    trans=torch.stack(transmittances).reshape(1,height,width,1)
    stack=lambda values:torch.stack(values) if values else means2d.new_empty(0)
    return {'image':image,'alpha':1-trans,'transmittance':trans,
            'stats':ContributionStats(stack(errors),stack(maxima),stack(sums))}

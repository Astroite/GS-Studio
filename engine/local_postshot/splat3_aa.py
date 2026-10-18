"""CPU/Torch oracle for the recovered Splat3 2D opacity compensation.

This is a numerical reference, not the training hot path. The native renderer
uses straight-through derivatives at all three determinant/ratio floors. This
VJP is intentionally not the mathematical derivative of a clamped function.
Source: runtime-matched PTX i_jy8:35712..35731, i_Od9:139629..139716.
Float32 instruction/FMA parity is tested by a separate scalar PTX witness;
these vectorized Torch operations promise formula parity only.
"""
from __future__ import annotations
import struct
import torch

AA_EPSILON = struct.unpack('<f', struct.pack('<I', 0x358637BD))[0]
NATIVE_BLUR = struct.unpack('<f', struct.pack('<I', 0x3E99999A))[0]


def _parts(raw, eps2d):
    a, b, c = raw.unbind(-1)
    numerator = (a*c-b*b).clamp_min(AA_EPSILON)
    denominator = ((a+eps2d)*(c+eps2d)-b*b).clamp_min(AA_EPSILON)
    compensation = (numerator/denominator).clamp_min(AA_EPSILON).sqrt()
    return a, b, c, numerator, denominator, compensation


class _Compensation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, eps2d):
        a, b, c, n, d, comp = _parts(raw, eps2d)
        ctx.save_for_backward(a, b, c, n, d, comp)
        ctx.eps2d = eps2d
        return comp

    @staticmethod
    def backward(ctx, upstream):
        a, b, c, n, d, comp = ctx.saved_tensors
        # i_Od9 has no predicate for any floor and no '+epsilon' on comp.
        factor = upstream/(comp+comp)
        gn = factor/d
        gd = -(factor*n)/(d*d)
        result = torch.stack((gn*c+gd*(c+ctx.eps2d),
                              -2*b*(gn+gd),
                              gn*a+gd*(a+ctx.eps2d)), -1)
        return result, None


def compensation(raw: torch.Tensor, eps2d: float = NATIVE_BLUR) -> torch.Tensor:
    """Return compensation for symmetric raw covariance (..., a,b,c).

    Ordinary float32/float64 tensors only. Non-PSD inputs are admitted solely
    for testing floor semantics; actual projected covariance must be PSD.
    """
    if raw.ndim < 1 or raw.shape[-1] != 3 or raw.dtype not in (torch.float32, torch.float64):
        raise ValueError('raw covariance must be float32/float64 (...,3)')
    if not isinstance(eps2d, (float, int)) or not 0 < eps2d < float('inf'):
        raise ValueError('eps2d must be finite and positive')
    return _Compensation.apply(raw, float(eps2d))

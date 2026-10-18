"""Private tile support, independent of frozen renderer/statistics binaries.

conservative_v1 only prunes the public bounding rectangle and preserves stable
depth ordering. native_exact_v1 implements recovered native support algebra,
but still consumes public projected conics/visibility; it is not bitwise native
projection parity. Its C log/FMA rounding is not the original PTX polynomial.
"""
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import struct

MODES = {"conservative_v1": 0, "native_exact_v1": 1}
ALPHA_CUTOFF = struct.unpack("f", struct.pack("I", 0x3B808081))[0]
NATIVE_RECIPROCAL = struct.unpack("f", struct.pack("I", 0x437EFFFF))[0]


@lru_cache(maxsize=1)
def _extension():
    try:
        module = importlib.import_module("local_splat3_tiles")
    except (ImportError, OSError) as error:
        raise RuntimeError("Private local_splat3_tiles is unavailable; no silent fallback") from error
    if getattr(module, "implementation_version", None) != "tile_support_v1":
        raise RuntimeError("Private tile implementation differs from tile_support_v1")
    return module


def tiles_identity():
    """No CUDA initialization, including from an uninitialized worker process."""
    dist = importlib.metadata.distribution("local-splat3-tiles")
    binaries = [Path(dist.locate_file(f)) for f in dist.files if str(f).endswith(".pyd")]
    if len(binaries) != 1:
        raise RuntimeError("Private tiles distribution requires exactly one binary")
    return {"distribution": "local-splat3-tiles", "version": dist.version,
            "binary_sha256": hashlib.sha256(binaries[0].read_bytes()).hexdigest()}


def intersect_tiles(means2d, radii, depths, conics, opacities, width, height,
                    tile_size=16, *, mode="conservative_v1"):
    """One CUDA float32 view; returns counts, sorted keys, IDs, [1,TH,TW] offsets.

    Selection is nondifferentiable. Invalid conics/opacities are conservatively
    retained in the public bbox; projected zero radii remain invisible. Device
    allocation has one intersection-count synchronization, as in public gsplat.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown tile support mode: {mode}")
    return tuple(_extension().intersect(
        *(v.detach().contiguous() for v in (means2d, radii, depths, conics, opacities)),
        int(width), int(height), int(tile_size), MODES[mode]))


def minimum_quadratic_on_rectangle(mean, conic, rect):
    """Float64 convex reference, all four edges; no GPU/public implementation."""
    mx, my = mean
    a, b, c = conic
    x0, y0, x1, y1 = rect
    if x0 <= mx <= x1 and y0 <= my <= y1:
        return 0.0
    candidates = []
    for x in (x0-mx, x1-mx):
        y = min(max(-b*x/c, y0-my), y1-my)
        candidates.append(a*x*x + 2*b*x*y + c*y*y)
    for y in (y0-my, y1-my):
        x = min(max(-b*y/a, x0-mx), x1-mx)
        candidates.append(a*x*x + 2*b*x*y + c*y*y)
    return min(candidates)


def native_edge_intersection(mean, conic, rect, threshold):
    """Recovered i_xy8 36482..36556 two-near-edge discriminant algebra.

    CPU float64 interpretation, inclusive tangencies and full tile boundaries;
    not a claim about FTZ, polynomial logarithm, or float32 FMA boundary parity.
    """
    mx, my = mean
    a, b, c = conic
    x0, y0, x1, y1 = rect
    if x0 <= mx <= x1 and y0 <= my <= y1:
        return True
    y = (y0 if y0+y1 > 2*my else y1) - my
    B = 2*b*y
    D = B*B + 4*a*(threshold-c*y*y)
    edge = max(2*a*(x0-mx)+B, -(2*a*(x1-mx)+B))
    if D >= 0 and (edge <= 0 or D >= edge*edge):
        return True
    x = (x0 if x0+x1 > 2*mx else x1) - mx
    B = 2*b*x
    D = B*B + 4*c*(threshold-a*x*x)
    edge = max(2*c*(y0-my)+B, -(2*c*(y1-my)+B))
    return D >= 0 and (edge <= 0 or D >= edge*edge)


def native_circle_radius(conic, opacity):
    a, b, c = conic
    det = a*c-b*b
    xx, xy, yy = c/det, -b/det, a/det
    mid = .5*(xx+yy)
    eigen = mid+math.sqrt(max(.01, mid*mid-(xx*yy-xy*xy)))
    threshold = 2*math.log(max(1., NATIVE_RECIPROCAL*opacity))
    return math.sqrt(eigen*threshold), threshold


def reference_tiles(means, radii, depths, conics, opacities, width, height,
                    tile_size=16, *, mode="conservative_v1"):
    """Independent CPU list oracle; stable equal-depth Gaussian order."""
    if mode not in MODES:
        raise ValueError(mode)
    tw, th = (width+tile_size-1)//tile_size, (height+tile_size-1)//tile_size
    pairs, counts = [], []
    for g, ((mx, my), (rx, ry), z, q, opacity) in enumerate(zip(means, radii, depths, conics, opacities)):
        count = 0
        if rx <= 0 or ry <= 0 or not math.isfinite(mx+my):
            counts.append(0)
            continue
        a, b, c = q
        valid = all(math.isfinite(v) for v in (*q, opacity)) and a > 0 and c > 0 and a*c-b*b > 0 and opacity > 0
        native = mode == "native_exact_v1" and valid
        if native:
            radius, threshold = native_circle_radius(q, opacity)
            if radius <= 0:
                counts.append(0)
                continue
            x0, y0 = max(0, math.floor((mx-radius)/tile_size)), max(0, math.floor((my-radius)/tile_size))
            x1, y1 = min(tw-1, math.floor((mx+radius)/tile_size)), min(th-1, math.floor((my+radius)/tile_size))
        else:
            x0, y0 = max(0, min(tw, math.floor(mx/tile_size-rx/tile_size))), max(0, min(th, math.floor(my/tile_size-ry/tile_size)))
            x1, y1 = max(0, min(tw, math.ceil(mx/tile_size+rx/tile_size)))-1, max(0, min(th, math.ceil(my/tile_size+ry/tile_size)))-1
            threshold = 2*math.log(max(1., opacity/ALPHA_CUTOFF)) if valid else math.inf
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                rect = (x*tile_size, y*tile_size, (x+1)*tile_size, (y+1)*tile_size)
                keep = not valid or (not native and x0 == x1 and y0 == y1)
                if not keep:
                    keep = native_edge_intersection((mx, my), q, rect, threshold) if native else minimum_quadratic_on_rectangle((mx, my), q, rect) <= threshold+2e-4*(1+threshold)
                if keep:
                    key = ((y*tw+x) << 32) | struct.unpack("I", struct.pack("f", z))[0]
                    pairs.append((key, g))
                    count += 1
        counts.append(count)
    pairs.sort(key=lambda item: item[0])
    offsets, cursor = [], 0
    for tile in range(tw*th):
        while cursor < len(pairs) and pairs[cursor][0] >> 32 < tile:
            cursor += 1
        offsets.append(cursor)
    return counts, [k for k, _ in pairs], [g for _, g in pairs], offsets

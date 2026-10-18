"""Editor-only bounded ROI maxima over unmodified full-frame metadata.

Build with engine/scripts/build_studio_selection.ps1. No JIT build or fallback
in a user operation; packaged installations ship this separate helper under engine/extensions.
"""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys


def build_location():
    import torch
    from .studio_bridge import config
    source = Path(__file__).resolve().parents[1]/'editor_native'
    key = hashlib.sha256((torch.__version__+sys.version).encode())
    for name in ('selection.cpp','selection.cu'):
        key.update((source/name).read_bytes())
    cfg = config()
    # Required release binaries are always package-owned, regardless of user caches.
    extensions = source.parent/'extensions' if cfg.get('release') else Path(cfg['torch_extensions_dir'])
    root = extensions/'studio-selection'/key.hexdigest()[:20]
    return source,root


@lru_cache(maxsize=1)
def extension():
    _,root = build_location()
    candidates = list(root.glob('studio_selection_cuda*.pyd'))+list(root.glob('studio_selection_cuda*.so'))
    if len(candidates)!=1:
        raise RuntimeError('Editor ROI precompiled helper missing; repair the release package or build it offline before packaging')
    spec = importlib.util.spec_from_file_location('studio_selection_cuda',candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def roi_max(meta,bounds):
    return extension().roi_max(*(meta[k].detach().contiguous() for k in
        ('means2d','conics','opacities','isect_offsets','flatten_ids')),
        meta['width'],meta['height'],meta['tile_size'],*bounds,
        meta.get('compositor')=='native_renderer_v1')

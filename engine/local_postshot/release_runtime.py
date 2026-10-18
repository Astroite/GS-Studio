"""Release import boundary. No kernels or training calculations are changed."""
import importlib
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import PathFinder, EXTENSION_SUFFIXES
import sys

_INSTALLED = False
REQUIRED_BINARIES = ('gsplat.csrc', 'fused_ssim_cuda', 'local_splat3_stats',
                     'local_splat3_render', 'local_splat3_tiles',
                     'local_splat3_projection', 'spz.spz')


def validate_binary_files():
    # Traverse specs without executing package initializers, importing torch or CUDA.
    for name in REQUIRED_BINARIES:
        search = None
        spec = None
        for index, part in enumerate(name.split('.')):
            prefix = '.'.join(name.split('.')[:index + 1])
            spec = PathFinder.find_spec(prefix, search)
            if spec is None:
                break
            search = spec.submodule_search_locations
        if spec is None or not any(str(spec.origin).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            raise RuntimeError(f'GS Studio required precompiled binary {name} is missing; repair the release package. Runtime JIT/download is disabled.')



def _unavailable(*args, **kwargs):
    raise RuntimeError('GS Studio release requires precompiled native binaries and offline model weights; runtime JIT/download is disabled. Repair the package.')


def require_binary(name):
    """Fail before gsplat can catch ImportError and enter its JIT branch."""
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f'GS Studio required native binary {name} is missing or cannot load: {exc}. Repair the release package; JIT fallback is disabled.') from exc


class _ReleaseLoader(Loader):
    def __init__(self, name, original):
        self.name, self.original = name, original

    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, 'create_module') else None

    def exec_module(self, module):
        if self.name == 'gsplat.cuda._backend':
            require_binary('gsplat.csrc')
        self.original.exec_module(module)
        if self.name == 'torch.utils.cpp_extension':
            for name in ('load', 'load_inline', '_jit_compile'):
                setattr(module, name, _unavailable)
        if self.name == 'torch.hub':
            module.download_url_to_file = _unavailable


class _ReleaseFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ('gsplat.cuda._backend', 'torch.utils.cpp_extension', 'torch.hub'):
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = _ReleaseLoader(fullname, spec.loader)
        return spec


def install_if_packaged():
    global _INSTALLED
    if _INSTALLED:
        return
    from .runtime import config
    if not config().get('release'):
        return
    validate_binary_files()
    sys.meta_path.insert(0, _ReleaseFinder())
    _INSTALLED = True
    # Protect processes whose controlled bootstrap imported torch first too.
    for name in ('torch.utils.cpp_extension', 'torch.hub'):
        module = sys.modules.get(name)
        if module is not None:
            for attr in (('load', 'load_inline', '_jit_compile') if name.endswith('cpp_extension') else ('download_url_to_file',)):
                setattr(module, attr, _unavailable)

"""Gaussian-splat scene database workflow."""

__version__ = "0.3.0rc1"

# Enforce the same release import boundary for direct CLI and pipeline children.
def _configure_release():
    from .infrastructure.runtime.configuration import runtime_config
    cfg = runtime_config()
    if cfg.get('release'):
        import sys
        from pathlib import Path
        root = Path(cfg.get('engine_root') or '')
        if not root.is_absolute() or not (root / 'local_postshot/release_runtime.py').is_file():
            raise RuntimeError('GS Studio release is missing its native runtime guard; repair the package')
        sys.path.insert(0, str(root))
        from local_postshot.release_runtime import install_if_packaged
        install_if_packaged()

_configure_release()
del _configure_release

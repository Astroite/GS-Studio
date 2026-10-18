import importlib.metadata
import importlib
import hashlib
import json
import os
from pathlib import Path
import sys


def _adjacent_runtime():
    return Path(__file__).resolve().parents[2] / 'runtime' / 'runtime.json'


# Keep this pure-stdlib resolver identical in the engine and pipeline; parity is tested.
PATH_KEYS = ('python', 'preprocess_python', 'data_root', 'user_data_root', 'engine_root',
             'pipeline_root', 'colmap', 'ffmpeg', 'ffprobe', 'media_helper', 'mediasdk_root',
             'realityscan', 'realityscan_export_params', 'torch_extensions_dir', 'cuda_path',
             'gpu_lock_path', 'torch_home', 'cache_root', 'log_root')
ENV_PATHS = {'data_root': 'GSSTUDIO_DATA_ROOT', 'gpu_lock_path': 'GSSTUDIO_GPU_LOCK_PATH',
             'cache_root': 'GSSTUDIO_CACHE_ROOT', 'torch_home': 'TORCH_HOME',
             'torch_extensions_dir': 'TORCH_EXTENSIONS_DIR', 'cuda_path': 'CUDA_PATH',
             'media_helper': 'GSSTUDIO_MEDIA_HELPER', 'mediasdk_root': 'INSTA360_MEDIA_SDK_ROOT',
             'realityscan': 'GSSTUDIO_REALITYSCAN_CLI',
             'realityscan_export_params': 'GSSTUDIO_REALITYSCAN_EXPORT_PARAMS'}


def _local_app_data():
    configured = os.environ.get('LOCALAPPDATA')
    if configured:
        return Path(configured)
    if os.name == 'nt':
        import ctypes
        buffer = ctypes.create_unicode_buffer(32768)
        # CSIDL_LOCAL_APPDATA also works for isolated children without USERPROFILE.
        if ctypes.windll.shell32.SHGetFolderPathW(None, 28, None, 0, buffer) == 0:
            return Path(buffer.value)
    return Path.home() / 'AppData' / 'Local'


def runtime_config() -> dict:
    """Environment > saved project choice > override JSON > runtime JSON > portable defaults.

    GSSTUDIO_DATA_ROOT changes projects only: all entry points still share one
    GPU lock under LOCALAPPDATA/GSStudio unless explicitly overridden.
    """
    local = _local_app_data() / 'GSStudio'
    selected = os.environ.get('GSSTUDIO_RUNTIME') or os.environ.get('LOCAL_POSTSHOT_RUNTIME')
    path = Path(selected).expanduser().resolve() if selected else _adjacent_runtime()
    user_root = local.resolve()
    cfg = {'runtime_path': str(path)}
    path_bases = {}
    override = Path(os.environ.get('GSSTUDIO_RUNTIME_OVERRIDE') or local / 'runtime.override.json').expanduser().resolve()
    for document in (path, override):
        if not document.is_file():
            if document == path and selected or document == override and os.environ.get('GSSTUDIO_RUNTIME_OVERRIDE'):
                raise FileNotFoundError(f'Configured runtime JSON is missing: {document}')
            continue
        fields = json.loads(document.read_text(encoding='utf-8-sig'))
        if not isinstance(fields, dict):
            raise ValueError(f'Runtime JSON must be an object: {document}')
        if fields.get('user_data_root'):
            value = str(fields['user_data_root']).replace('${user_data_root}', str(user_root))
            user_root = Path(value).expanduser()
            if not user_root.is_absolute():
                user_root = document.parent / user_root
            user_root = user_root.resolve()
        for key in PATH_KEYS:
            if fields.get(key):
                path_bases[key] = document.parent
        cfg.update(fields)
    for key in PATH_KEYS:
        if cfg.get(key):
            value = str(cfg[key]).replace('${user_data_root}', str(user_root))
            resolved = Path(value).expanduser()
            cfg[key] = str((resolved if resolved.is_absolute() else path_bases[key] / resolved).resolve())
    cfg['runtime_path'] = str(path)
    cfg['user_data_root'] = str(user_root)
    cfg.setdefault('data_root', str(user_root / 'projects'))
    cfg.setdefault('cache_root', str(user_root / 'cache'))
    cfg.setdefault('log_root', str(user_root / 'logs'))
    cfg.setdefault('gpu_lock_path', str(user_root / 'cache' / 'gpu.lock'))
    if cfg.get('studio_source') and not cfg.get('pipeline_root'):
        cfg['pipeline_root'] = str((path.parent / cfg['studio_source']).resolve())
    settings_dir = Path(os.environ.get('GSSTUDIO_SETTINGS_DIR') or local).expanduser()
    settings = settings_dir / 'settings.json'
    if settings.is_file():
        saved = json.loads(settings.read_text(encoding='utf-8-sig'))
        if saved.get('schema_version') != 1 or not isinstance(saved.get('data_root'), str):
            raise RuntimeError(f'Invalid GS Studio settings: {settings}')
        cfg['data_root'] = str(Path(saved['data_root']).expanduser().resolve())
    for key, name in ENV_PATHS.items():
        if os.environ.get(name, '').strip():
            value = Path(os.environ[name]).expanduser()
            if not value.is_absolute():
                raise ValueError(f'{name} requires an absolute path')
            cfg[key] = str(value.resolve())
    if cfg.get('release') and cfg.get('python') != cfg.get('preprocess_python'):
        raise ValueError('GS Studio release python and preprocess_python must name the same interpreter')
    return cfg


def config():
    return runtime_config()

def executable(name):
    result = config().get(name)
    if not result or not Path(result).is_file():
        raise RuntimeError(f"Missing {name}; install the isolated runtime first")
    return str(result)


def health():
    cfg = config()
    paths = {name: cfg.get(name) for name in ("python", "colmap", "ffmpeg")}
    versions, missing = {}, []
    for name in ("torch", "gsplat", "numpy", "scipy", "Pillow", "plyfile", "spz"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "spz":
                try:
                    import spz
                    versions[name] = getattr(spz, "__version__", "official binding")
                    continue
                except ImportError:
                    pass
            missing.append(name)
    for name in ("colmap", "ffmpeg"):
        try:
            paths[name] = executable(name)
        except RuntimeError:
            missing.append(name)
    for module in ("gsplat.csrc","spz","PIL.Image","plyfile"):
        try:
            importlib.import_module(module)
        except (ImportError,OSError) as exc:
            missing.append(f"{module} import: {exc}")
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                   "memory_mb": torch.cuda.get_device_properties(0).total_memory // 1048576}
        else:
            missing.append("CUDA GPU")
    except (ImportError, OSError) as exc:
        missing.append(f"torch runtime: {exc}")
    optional_backends = {}
    for module, distribution in (("fused_ssim", "fused-ssim"), ("local_splat3_stats", "local-splat3-stats"),
                                 ("local_splat3_render", "local-splat3-render"),
                                 ("local_splat3_tiles", "local-splat3-tiles"),
                                 ("local_splat3_projection", "local-splat3-projection")):
        try:
            importlib.import_module(module)
            backend = {"ready": True, "distribution": distribution,
                       "version": importlib.metadata.version(distribution)}
            if module in ("local_splat3_render", "local_splat3_tiles", "local_splat3_projection"):
                dist = importlib.metadata.distribution(distribution)
                binaries = [Path(dist.locate_file(p)) for p in (dist.files or ()) if str(p).endswith(".pyd")]
                if len(binaries) != 1:
                    raise RuntimeError("Private renderer must contain exactly one native binary")
                backend["binary_sha256"] = hashlib.sha256(binaries[0].read_bytes()).hexdigest()
                backend["implementation"] = getattr(importlib.import_module(module), "implementation_version", None)
            optional_backends[module] = backend
        except (ImportError, OSError, RuntimeError, importlib.metadata.PackageNotFoundError) as exc:
            optional_backends[module] = {"ready": False, "error": str(exc)}
    return {"ready": not missing, "versions": versions, "paths": paths,
            "optional_backends": optional_backends,
            "gpu": gpu, "missing": missing, "python": sys.executable,
            "data_root": cfg["data_root"], "engine_version": "0.3.0rc1",
            "algorithm": "Public gsplat projection with public or independent compositor; public ADC or observed Splat3 policy"}

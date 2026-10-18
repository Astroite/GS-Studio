"""Pure Python runtime configuration; importing it never imports a torch ABI."""
from __future__ import annotations
import json
import os
from pathlib import Path


def _adjacent_runtime():
    return Path(__file__).resolve().parents[4] / 'runtime' / 'runtime.json'


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

MODULE_BOOTSTRAP = (
    'import json,runpy,sys; sys.dont_write_bytecode=True; '
    '[s.reconfigure(encoding="utf-8") for s in (sys.stdin,sys.stdout,sys.stderr) '
    'if s is not None and hasattr(s,"reconfigure")]; '
    'roots=json.loads(sys.argv[1]); module=sys.argv[2]; '
    'sys.path[:0]=roots; del sys.argv[1:3]; sys.argv[0]=module; '
    'runpy.run_module(module,run_name="__main__",alter_sys=True)'
)


def isolated_module_command(cfg: dict, module: str, *args: str) -> list[str]:
    """Build a private interpreter command without cwd/PYTHONPATH import lookup."""
    python = Path(cfg.get('python') or '')
    if not cfg.get('python') or not python.is_absolute() or not python.is_file():
        raise RuntimeError('Configure explicit private python in GSSTUDIO_RUNTIME')
    roots = []
    for key in ('engine_root', 'pipeline_root'):
        root = Path(cfg.get(key) or '')
        if not cfg.get(key) or not root.is_absolute() or not root.is_dir():
            raise RuntimeError(f'Configure explicit {key} in GSSTUDIO_RUNTIME')
        roots.append(str(root.resolve()))
    return [str(python), '-I', '-B', '-u', '-c', MODULE_BOOTSTRAP,
            json.dumps(roots), module, *map(str, args)]


def child_environment(cfg: dict | None = None, *, core: bool = False) -> dict[str, str]:
    cfg = cfg or runtime_config()
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', GSSTUDIO_RUNTIME=cfg['runtime_path'])
    env.pop('PYTHONHOME', None)
    roots = [cfg[k] for k in ('engine_root', 'pipeline_root') if cfg.get(k)]
    env['PYTHONPATH'] = os.pathsep.join(roots)
    for key, name in ENV_PATHS.items():
        if cfg.get(key):
            env.setdefault(name, cfg[key])
    if cfg.get('cuda_path'):
        env.setdefault('CUDA_HOME', cfg['cuda_path'])
    if cfg.get('cache_root'):
        env.setdefault('HF_HOME', str(Path(cfg['cache_root']) / 'huggingface'))
    tool_dirs = [str(Path(cfg[k]).parent) for k in ('ffmpeg', 'ffprobe', 'colmap') if cfg.get(k)]
    env['PATH'] = os.pathsep.join(tool_dirs + [env.get('PATH', '')])
    return env

"""Source-aware experiment defaults, independent of native recipe math."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from gsstudio.infrastructure.runtime.configuration import child_environment, runtime_config

SOURCE_KINDS = {'perspective_images', 'insta360_insv', 'equirect_video',
                'equirect_sequence', 'perspective_video', 'colmap', 'shared_package'}


def resolve_settings(meta, *, source_kind=None, training_preset='auto', steps=None,
                     training_recipe='quality_v2', photo_comp=None,
                     use_bilateral_grid=None, use_sparse_depth=None, sh_degree=3):
    source_kind = source_kind or meta.get('source_kind') or 'shared_package'
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f'Unknown training source kind: {source_kind}')
    if training_preset not in {'auto', 'photo', 'gs'}:
        raise ValueError('training_preset must be auto, photo or gs')
    preset = ('photo' if source_kind in {'perspective_images', 'colmap'} else 'gs') if training_preset == 'auto' else training_preset
    recipe = 'quality_v2' if training_recipe == 'gs_studio' else training_recipe
    if recipe not in {'quality_v2', 'observed_v5', 'observed_v6'} or sh_degree != 3:
        raise ValueError('Choose quality_v2, observed_v5 (released) or observed_v6 (experimental), with SH3')
    count = sum(r['split'] == 'train' for r in meta['images'])
    budget = (10000 if preset == 'photo' else max(30000, 30 * count)) if steps is None else steps
    if type(budget) is not int or budget < 1:
        raise ValueError('steps must be a positive integer')
    flags = {}
    for name, value in (('photo_comp', photo_comp), ('use_bilateral_grid', use_bilateral_grid),
                        ('sparse_depth_requested', use_sparse_depth)):
        if value is not None and type(value) is not bool:
            raise ValueError(f'{name} must be a boolean or omitted')
        flags[name] = preset == 'gs' if value is None else value
    if use_bilateral_grid is True and not flags['photo_comp']:
        raise ValueError('Bilateral Grid requires photo_comp=true (CLI: --photo-comp --use-bilateral-grid); photo_comp is the master enable')
    return dict(source_kind=source_kind, requested_training_preset=training_preset,
                training_preset=preset, steps=budget, training_recipe=recipe,
                sh_degree=3, max_splats=1_000_000, seed=42,
                max_image_size=1600 if preset == 'photo' else 0, **flags)


def depth_availability(package: Path) -> dict:
    """Use the core's real CPU validator without loading its ABI in preprocessing."""
    from gsstudio.infrastructure.adapters.gsplat import trainer_python
    cfg = runtime_config()
    engine = cfg.get('engine_root') or str(Path(__file__).resolve().parents[4] / 'engine')
    script = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
              'from local_postshot.studio_dataset import sparse_depth_availability;'
              'print(json.dumps(sparse_depth_availability(sys.argv[2]),allow_nan=False))')
    proc = subprocess.run([str(trainer_python()), '-I', '-B', '-c', script, str(engine), str(package)],
                          env=child_environment(core=True), capture_output=True, text=True,
                          encoding='utf-8', timeout=300,
                          creationflags=subprocess.CREATE_NO_WINDOW if __import__('os').name == 'nt' else 0)
    if proc.returncode:
        raise RuntimeError(f'Core package validation failed: {proc.stderr.strip()}')
    result = json.loads(proc.stdout)
    from gsstudio.infrastructure.adapters.media import sha256_file
    if (type(result.get('available')) is not bool or type(result.get('observations')) is not int
            or result['observations'] < 0 or result['available'] != (result['observations'] > 0)
            or result.get('package_sha256') != sha256_file(package / 'dataset.json')):
        raise RuntimeError('Invalid core sparse-depth availability response')
    return result


def effective_settings(settings, availability):
    requested = settings['sparse_depth_requested']
    return dict(settings, use_sparse_depth=requested and availability['available'],
                sparse_depth_available=availability['available'],
                sparse_depth_observations=availability['observations'],
                sparse_depth_status='disabled' if not requested else 'enabled' if availability['available'] else 'unavailable',
                sparse_depth_reason=availability['reason'])


def prepare_training(package: Path, **options):
    from gsstudio.pipeline.training.data import validate_package
    from gsstudio.pipeline.training.resize import resize_package
    from gsstudio.infrastructure.adapters.media import sha256_file
    package = Path(package).resolve()
    meta = validate_package(package)
    settings = resolve_settings(meta, **options)
    # Validate original provenance before deriving any pixels or tracks.
    availability = depth_availability(package)
    source_digest = availability['package_sha256']
    limit = settings['max_image_size']
    if limit and any(max(r['width'], r['height']) > limit for r in meta['images']):
        target = package.with_name(f'{package.name}-side{limit}-{source_digest[:16]}')
        resize_package(package, target, limit)
        package = target
        meta = validate_package(package)
        availability = depth_availability(package)
    settings = effective_settings(settings, availability)
    return package, meta, dict(settings, source_package_sha256=source_digest,
                               package_sha256=sha256_file(package / 'dataset.json'))

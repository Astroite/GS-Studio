"""Explicit, independent COLMAP imports outside the Run/QA namespace."""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gsstudio.application._shared import _verify_model
from gsstudio.application.contracts import EventSink, _emit
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.persistence.manifests import canonical_hash
from gsstudio.infrastructure.runtime.run_lock import run_session
from gsstudio.pipeline.editor.data import Dataset, discover_models
from gsstudio.pipeline.training.data import json_write, validate_package
from gsstudio.pipeline.training.runner import train_package, verify_evaluation


IMPORT_ROOT = '.external-imports'
_ID = re.compile(r'[a-z][a-z0-9-]{0,63}\Z')


class ModelSelectionRequired(ValueError):
    def __init__(self, candidates: list[str]):
        self.candidates = candidates
        super().__init__('请选择一个完整 COLMAP 模型：' + ', '.join(candidates))


def _project(root: Path, import_id: str) -> Path:
    if not _ID.fullmatch(import_id):
        raise ValueError('External import ID must be a lowercase slug')
    return Path(root).resolve() / IMPORT_ROOT / import_id


def _groups(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(value, dict) and 'groups' in value:
        value = value['groups']
    if not isinstance(value, dict) or not all(isinstance(k, str) and
            isinstance(v, (str, int)) for k, v in value.items()):
        raise ValueError('Grouping JSON must map each COLMAP image name to a group ID')
    return {key: str(item) for key, item in value.items()}


def _dataset(*, source: Path | None = None, images: Path | None = None,
             model: Path | None = None, masks: Path | None = None,
             group_file: Path | None = None, independent_images: bool = False,
             white_ignore: bool = False) -> tuple[Dataset, list[str]]:
    if source is not None:
        if any(value is not None for value in (images, model, masks, group_file)) or independent_images:
            raise ValueError('A verified shared package cannot be combined with raw COLMAP options')
        root = Path(source).resolve()
        if not (root / 'dataset.json').is_file():
            raise ValueError('Shared source needs a validated dataset.json package')
        return Dataset(root), []
    if images is None:
        raise ValueError('Choose an images directory')
    images = Path(images).resolve()
    if not images.is_dir():
        raise FileNotFoundError('Images directory is missing')
    selected = Path(model).resolve() if model is not None else None
    search = selected if selected is not None else images.parent
    discovered = discover_models(search)
    if selected is not None:
        if str(selected) not in discovered:
            raise ModelSelectionRequired(discovered)
        model = selected
    elif len(discovered) == 1:
        model = Path(discovered[0])
    else:
        raise ModelSelectionRequired(discovered)
    if not model.is_dir():
        raise FileNotFoundError('Images or COLMAP model directory is missing')
    try:
        common = Path(os.path.commonpath((images, model)))
    except ValueError:  # Windows source images and model may be on different drives.
        common = model.parent
    # The source may live under an ordinary location, but a Run reconstruction
    # cannot be passed off as an independently approved external input.
    for parent in (common, images, model):
        if any((parent / name).exists() for name in
               ('segments.json', 'mask-filter.json', 'mask-final.json', 'manifest.yaml')):
            raise ValueError('Run reconstruction output must use its QA-approved training package')
    candidates = sorted(set(discover_models(common) + [str(model)]))
    data = Dataset(common, images=images, model=model, masks=masks,
                   white_ignore=white_ignore, groups=_groups(group_file),
                   independent_images=independent_images)
    if group_file is not None:
        data.files[str(Path(group_file).resolve())] = sha256_file(Path(group_file))
    return data, candidates


def _identity(data: Dataset) -> str:
    return canonical_hash({'files': data.files, 'group_source': data.group_source,
                           'group_labels': data.group_labels,
                           'white_ignore': data.white_ignore})


def inspect_external_import(**options: Any) -> dict[str, Any]:
    """Read source material and return errors plus bounded visual inspection data."""
    try:
        data, candidates = _dataset(**options)
    except ModelSelectionRequired as error:
        return {'source_kind': 'colmap', 'source_root': None,
                'image_root': str(options.get('images') or ''), 'model': None,
                'model_candidates': error.candidates, 'registered_images': 0,
                'camera_models': [],
                'image_inventory': 0, 'cameras': [], 'cameras_truncated': False,
                'initial_points': 0, 'masks': 0, 'group_source': 'required',
                'groups': 0, 'input_identity': None, 'errors': [str(error)],
                'warnings': []}
    cameras = []
    for row in data.rows[:2000]:
        sample = []
        if not data.package:
            observations = data.image_records[row['image_id']]['obs']
            for x, y, point_id in observations:
                if (point_id in data.points and np.isfinite((x, y)).all() and
                        0 <= x < row['width'] and 0 <= y < row['height']):
                    sample.append([round(float(x), 2), round(float(y), 2)])
                    if len(sample) == 200:
                        break
        cameras.append({'image_id': row.get('image_id'), 'source_image': row['source_image'],
                        'source_path': row['source_path'], 'width': row['width'],
                        'height': row['height'], 'group': data.group_labels.get(row['source_image']),
                        'split': row['split'],
                        'camera_model': (data.cameras[data.image_records[row['image_id']]['camera']][0]
                                         if not data.package else 'shared_package'),
                        'sparse_points': sample})
    return {'source_kind': 'shared_package' if data.package else 'colmap',
            'source_root': str(data.root), 'image_root': str(data.image_root),
            'model': str(data.model) if data.model else None,
            'model_candidates': candidates, 'registered_images': len(data.rows),
            'camera_models': sorted({row['camera_model'] for row in cameras}),
            'image_inventory': data.inventory, 'cameras': cameras,
            'cameras_truncated': len(data.rows) > len(cameras),
            'initial_points': len(data.xyz), 'masks': sum(bool(row['mask_path']) for row in data.rows),
            'group_source': data.group_source, 'groups': len(set(data.group_labels.values())),
            'input_identity': _identity(data), 'errors': data.errors,
            'warnings': data.warnings}


def prepare_external_import(root: Path, import_id: str, *, expected_input_identity: str | None = None,
                            sink: EventSink | None = None, **options: Any) -> dict[str, Any]:
    """Copy verified external inputs into a separately identified immutable package."""
    project = _project(root, import_id)
    if project.exists():
        raise FileExistsError(project)
    data, _ = _dataset(**options)
    data.verify()
    identity = _identity(data)
    if expected_input_identity is not None and identity != expected_input_identity:
        raise RuntimeError('External source identity changed since inspection')
    project.mkdir(parents=True)
    record = {'schema_version': 1, 'id': import_id, 'kind': 'external_colmap',
              'status': 'preparing', 'source_kind': 'shared_package' if data.package else 'colmap',
              'source_root': str(data.root), 'input_identity': identity,
              'source_files': data.files,
              'source_paths': {path: path for path in data.files},
              'white_ignore': data.white_ignore,
              'group_source': data.group_source, 'group_labels': data.group_labels,
              'package': 'package', 'qa_status': 'not_run'}
    json_write(project / 'import.json', record)
    _emit(sink, 'started', 'prepare_external_import', '正在复制并校验外部训练包',
          import_id=import_id)
    staging = project / f'.package-building-{uuid.uuid4().hex[:8]}'
    package = project / 'package'
    try:
        if data.package:
            shutil.copytree(data.root, staging)
        else:
            data.prepare(staging)
        data.verify()
        meta = validate_package(staging)
        from gsstudio.pipeline.training.presets import depth_availability
        availability = depth_availability(staging)
        record.update(source_kind=meta.get('source_kind', record['source_kind']),
                      sparse_depth_available=availability['available'],
                      sparse_depth_observations=availability['observations'],
                      sparse_depth_reason=availability['reason'])
        staging.rename(package)
        data.verify()
        validate_package(package)
        record.update(status='prepared', package_sha256=sha256_file(package / 'dataset.json'),
                      image_count=len(meta['images']), initial_points=meta['initial_points'])
        json_write(project / 'import.json', record)
        _emit(sink, 'completed', 'prepare_external_import', '外部训练包已校验',
              import_id=import_id, package=str(package), package_sha256=record['package_sha256'])
        return {**record, 'project': str(project), 'package_path': str(package)}
    except Exception as error:
        for path in (staging, package):
            # Both are created only by this new import ID; never follow a
            # replacement symlink outside its project while removing a failure.
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                absolute = path.resolve()
                if absolute == project.resolve() or not absolute.is_relative_to(project.resolve()):
                    raise RuntimeError(f'Unsafe import cleanup path: {absolute}') from error
                shutil.rmtree(path)
        record.update(status='failed', error=str(error))
        json_write(project / 'import.json', record)
        _emit(sink, 'failed', 'prepare_external_import', str(error), import_id=import_id)
        raise


def _load_package(root: Path, import_id: str) -> tuple[dict[str, Any], Path]:
    project = _project(root, import_id)
    record = json.loads((project / 'import.json').read_text(encoding='utf-8'))
    if (record.get('schema_version') != 1 or record.get('id') != import_id or
            record.get('status') != 'prepared' or record.get('qa_status') != 'not_run'):
        raise RuntimeError('External import record is incomplete or has invalid identity')
    package = project / 'package'
    meta = validate_package(package)
    expected_files = set(meta['files']) | {'dataset.json'}
    actual_files = {item.relative_to(package).as_posix() for item in package.rglob('*')
                    if item.is_file()}
    if actual_files != expected_files:
        raise RuntimeError('External training package file inventory changed')
    if sha256_file(package / 'dataset.json') != record.get('package_sha256'):
        raise RuntimeError('External training package identity changed')
    if canonical_hash({'files': record['source_files'],
                       'group_source': record['group_source'],
                       'group_labels': record['group_labels'],
                       'white_ignore': record['white_ignore']}) != record['input_identity']:
        raise RuntimeError('External source or grouping identity changed')
    return record, project


def verify_external_import(root: Path, import_id: str) -> tuple[dict[str, Any], Path]:
    record, project = _load_package(root, import_id)
    paths = record['source_paths']
    if set(paths) != set(record['source_files']):
        raise RuntimeError('External source path inventory changed')
    for path, digest in record['source_files'].items():
        current = Path(paths[path])
        if sha256_file(current) != digest:
            raise RuntimeError(f'External source identity changed: {path}')
    return record, project


def relink_external_import(root: Path, import_id: str, *, new_root: Path | None = None,
                           mapping_file: Path | None = None) -> dict[str, Any]:
    """Repoint read-only source references only after every original hash matches."""
    if (new_root is None) == (mapping_file is None):
        raise ValueError('Choose exactly one new source root or path mapping JSON')
    project = _project(root, import_id)
    with run_session(project):
        record, project = _load_package(root, import_id)
        old_paths = record['source_paths']
        originals = set(record['source_files'])
        if mapping_file is not None:
            mapping = json.loads(Path(mapping_file).read_text(encoding='utf-8-sig'))
            if not isinstance(mapping, dict) or not all(isinstance(k, str) and
                    isinstance(v, str) for k, v in mapping.items()):
                raise ValueError('Path mapping JSON must map original absolute path to new absolute path')
            if set(mapping) - originals:
                raise ValueError('Path mapping contains files outside the original source inventory')
            replacements = {key: Path(value).resolve() for key, value in mapping.items()}
        else:
            target = Path(new_root).resolve()
            if not target.is_dir():
                raise FileNotFoundError(target)
            source_root = Path(record['source_root']).resolve()
            replacements = {}
            for original in originals:
                try:
                    relative = Path(original).resolve().relative_to(source_root)
                except ValueError:
                    continue
                replacements[original] = target / relative
        updated = {}
        for original, digest in record['source_files'].items():
            current = replacements.get(original, Path(old_paths.get(original, original))).resolve()
            if sha256_file(current) != digest:
                raise RuntimeError(f'Relocated source hash differs: {original} -> {current}')
            updated[original] = str(current)
        record['source_paths'] = updated
        record['source_relocated_at'] = datetime.now(timezone.utc).isoformat()
        json_write(project / 'import.json', record)
        return {'import_id': import_id, 'source_paths': updated,
                'source_relocated_at': record['source_relocated_at']}


def _completed_experiment(output: Path, package: Path, record: dict[str, Any],
                          backend: str, requested_steps: int | None, *, photo_comp=None,
                          use_bilateral_grid=None,use_sparse_depth=None,
                          training_recipe='quality_v2',training_preset='auto',source_kind=None,sh_degree=3) -> dict[str, Any] | None:
    """Adopt a finished dispatch after a crash before the import record was saved."""
    dispatch_path = output / 'dispatch.json'
    if not dispatch_path.is_file():
        return None
    dispatch = json.loads(dispatch_path.read_text(encoding='utf-8'))
    if dispatch.get('status') != 'succeeded':
        return None
    from gsstudio.pipeline.training.runner import verify_dispatch_settings
    meta = verify_dispatch_settings(dispatch, package, steps=requested_steps,
        photo_comp=photo_comp, use_bilateral_grid=use_bilateral_grid,
        use_sparse_depth=use_sparse_depth, training_recipe=training_recipe,
        training_preset=training_preset, source_kind=source_kind, sh_degree=sh_degree)
    digest = dispatch.get('model_sha256')
    if not isinstance(digest, str):
        raise RuntimeError('Completed external training lacks a PLY identity')
    evaluation_digest = dispatch.get('evaluation_sha256')
    if not isinstance(evaluation_digest, str) or verify_evaluation(output, meta) != evaluation_digest:
        raise RuntimeError('Completed external training evaluation identity differs from dispatch')
    from gsstudio.pipeline.training.runner import verify_completed
    verify_completed(output,meta,dispatch)
    _verify_model(output / 'model.ply', digest)
    return dispatch


def train_external_import(root: Path, import_id: str, *, backend: str = 'gsplat',
                          steps: int | None = None, resume: bool = False,
                          experiment_id: str | None = None, dry_run: bool = False,
                          photo_comp: bool | None = None, use_bilateral_grid: bool | None = None,
                          use_sparse_depth: bool | None = None, preview_every: int = 500,
                          training_recipe: str = 'quality_v2', sh_degree: int = 3,
                          training_preset: str = 'auto',
                          restart_from_step_zero: bool = False,
                          sink: EventSink | None = None) -> dict[str, Any]:
    record, project = verify_external_import(root, import_id)
    from gsstudio.pipeline.training.runner import require_backend
    require_backend(backend)
    with run_session(project):
        record, project = verify_external_import(root, import_id)
        if resume:
            if not experiment_id:
                raise ValueError('Select the experiment to resume')
            if not _ID.fullmatch(experiment_id):
                raise ValueError('Invalid experiment ID')
            output = _project(root, import_id) / 'experiments' / experiment_id
            experiment_path = output.with_suffix('.json')
            experiment = json.loads(experiment_path.read_text(encoding='utf-8'))
            if (experiment.get('schema_version') != 1 or experiment.get('id') != experiment_id or
                    experiment.get('import_id') != import_id):
                raise RuntimeError('External experiment record identity differs from its path')
            if experiment.get('backend') != backend or experiment.get('status') not in {'failed', 'preparing', 'stopped', 'paused'}:
                raise ValueError('Experiment cannot be resumed with this backend')
            if steps is not None and experiment['steps'] is not None and steps != experiment['steps']:
                raise RuntimeError('External experiment identity changed: requested_steps')
            steps = experiment['steps'] if steps is None else steps
        else:
            experiment_id = f'{backend}-{uuid.uuid4().hex[:12]}'
            output = project / 'experiments' / experiment_id
            experiment_path = output.with_suffix('.json')
            output.parent.mkdir(parents=True, exist_ok=True)
            experiment = {'schema_version': 1, 'id': experiment_id,
                          'import_id': import_id, 'import_identity': record['input_identity'],
                          'package_sha256': record['package_sha256'], 'backend': backend,
                          'steps': steps, 'status': 'preparing'}
            json_write(experiment_path, experiment)
        if (experiment.get('import_identity') != record['input_identity'] or
                experiment.get('package_sha256') != record['package_sha256']):
            raise RuntimeError('External experiment input identity changed')
        package = project / 'package'
        from gsstudio.pipeline.training.presets import prepare_training
        actual_package, _, effective = prepare_training(package, source_kind=record['source_kind'],
            training_preset=training_preset,steps=steps,training_recipe=training_recipe,
            photo_comp=photo_comp,use_bilateral_grid=use_bilateral_grid,
            use_sparse_depth=use_sparse_depth,sh_degree=sh_degree)
        _emit(sink, 'progress', 'train_external_import', '外部训练日志已就绪',
              import_id=import_id, experiment_id=experiment_id,
              log_path=str(output / 'train.log'), output=str(output), package=str(actual_package),
              **{k:v for k,v in effective.items() if k!='steps'},requested_steps=effective['steps'],backend=backend)
        try:
            completed = (_completed_experiment(output, package, record, backend, steps,
                                               photo_comp=photo_comp,use_bilateral_grid=use_bilateral_grid,
                                               use_sparse_depth=use_sparse_depth,training_recipe=training_recipe,
                                               training_preset=training_preset,source_kind=record['source_kind'],sh_degree=sh_degree)
                         if resume and not dry_run else None)
            if completed is not None:
                experiment.update(status='succeeded', steps=completed['requested_steps'],
                                  model_sha256=completed['model_sha256'])
                experiment.pop('error', None)
                json_write(experiment_path, experiment)
                _emit(sink, 'completed', 'train_external_import', '已恢复完成的外部训练记录',
                      import_id=import_id, experiment_id=experiment_id,
                      output=str(output),model_path=str(output/'model.ply'))
                return {**completed, 'import_id': import_id, 'experiment_id': experiment_id,
                        'output': str(output), 'model_path':str(output/'model.ply')}
            result = train_package(package, output, backend=backend, steps=steps,
                                   resume=resume, dry_run=dry_run,
                                   photo_comp=photo_comp,use_bilateral_grid=use_bilateral_grid,
                                   use_sparse_depth=use_sparse_depth,preview_every=preview_every,
                                   training_recipe=training_recipe,sh_degree=sh_degree,
                                   restart_from_step_zero=restart_from_step_zero,
                                   training_preset=training_preset,source_kind=record['source_kind'])
            experiment.update(status='prepared' if dry_run else result['status'],
                              steps=result['requested_steps'], model_sha256=result.get('model_sha256'))
            experiment.pop('error', None)
            if not dry_run and result.get('status') not in {'stopped','paused'}:
                if result.get('status') != 'succeeded' or not result.get('model_sha256'):
                    raise RuntimeError('External training did not publish a verified PLY')
                _verify_model(output / 'model.ply', result['model_sha256'])
            json_write(experiment_path, experiment)
            _emit(sink, 'completed', 'train_external_import',
                  '训练输入已准备' if dry_run else
                  (('等待 GPU 时已取消；保留已有检查点' if result.get('checkpoint') else
                    '等待 GPU 时已取消；重新启动将从第 0 步开始')
                   if result.get('stop_reason') == 'gpu_wait_cancelled' else
                   '已在安全步边界结束，可从检查点续跑' if result.get('status') in {'stopped','paused'}
                   else '外部 PLY 已校验'),
                  import_id=import_id, experiment_id=experiment_id,
                  output=str(output),model_path=str(output/'model.ply') if result.get('status')=='succeeded' else None)
            return {**result, 'import_id': import_id, 'experiment_id': experiment_id,
                    'output': str(output),
                    'model_path':str(output/'model.ply') if result.get('status')=='succeeded' else None}
        except Exception as error:
            experiment.update(status='failed', error=str(error))
            json_write(experiment_path, experiment)
            _emit(sink, 'failed', 'train_external_import', str(error),
                  import_id=import_id, experiment_id=experiment_id)
            raise


def list_external_imports(root: Path) -> dict[str, Any]:
    base = Path(root).resolve() / IMPORT_ROOT
    items = []
    for path in sorted(base.glob('*/import.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('id') != path.parent.name:
                raise ValueError('Import ID and directory differ')
            experiments = []
            for item in sorted((path.parent / 'experiments').glob('*.json')):
                if item.name == 'dispatch.json':
                    continue
                experiments.append({**json.loads(item.read_text(encoding='utf-8')),
                                    'output':str(item.with_suffix(''))})
            items.append({**record, 'project': str(path.parent), 'experiments': experiments})
        except (OSError, ValueError, TypeError) as error:
            items.append({'id': path.parent.name, 'status': 'invalid',
                          'project': str(path.parent), 'error': str(error), 'experiments': []})
    return {'imports': items}

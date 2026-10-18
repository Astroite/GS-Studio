"""External COLMAP imports require an explicit grouping decision."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsstudio.application.external_imports import (
    inspect_external_import, list_external_imports, prepare_external_import,
    relink_external_import, train_external_import, verify_external_import,
)
from gsstudio.pipeline.training.data import postshot_adapter, validate_package
from gsstudio.pipeline.training.data import json_write
from gsstudio.pipeline.editor.data import write_image
from gsstudio.pipeline.editor.data import read_model
from gsstudio.pipeline.editor.ply import FLOAT_PROPERTIES
from gsstudio.infrastructure.adapters.media import sha256_file


def _source(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / '外部源' / 'images'
    model = tmp_path / '外部源' / 'sparse' / '0'
    images.mkdir(parents=True)
    model.mkdir(parents=True)
    (model / 'cameras.txt').write_text('1 PINHOLE 8 8 6 6 4 4\n', encoding='utf-8')
    lines = []
    for index in range(1, 4):
        name = f'frame_{index:06d}.png'
        write_image(images / name, np.full((8, 8, 3), index * 40, np.uint8))
        lines += [f'{index} 1 0 0 0 {index} 0 0 1 {name}',
                  '2 2 1 3 3 2 4 4 3 5 5 4']
    (model / 'images.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (model / 'points3D.txt').write_text('1 0 0 2 10 20 30 0\n'
                                         '2 1 0 2 40 50 60 0\n'
                                         '3 0 1 2 70 80 90 0\n'
                                         '4 1 1 2 70 80 90 0\n', encoding='utf-8')
    return images, model


def test_colmap_filename_never_proves_grouping(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    result = inspect_external_import(images=images, model=model)
    assert result['group_source'] == 'required'
    assert '缺少可信分组' in '\n'.join(result['errors'])
    assert result['registered_images'] == 3
    assert result['initial_points'] == 4
    assert len(result['cameras'][0]['sparse_points']) == 4
    with pytest.raises(ValueError, match='缺少可信分组'):
        prepare_external_import(tmp_path / 'Data', 'external-one',
                                images=images, model=model)
    assert not (tmp_path / 'Data' / '.external-imports').exists()


def test_explicit_groups_are_preserved_and_source_checked(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    grouping = tmp_path / '外部源' / 'groups.json'
    grouping.write_text(json.dumps({'groups': {
        'frame_000001.png': 'panorama-a',
        'frame_000002.png': 'panorama-a',
        'frame_000003.png': 'panorama-b',
    }}), encoding='utf-8')
    options = {'images': images, 'model': model, 'group_file': grouping}
    inspection = inspect_external_import(**options)
    assert not inspection['errors']
    assert inspection['groups'] == 2
    assert [camera['split'] for camera in inspection['cameras']] == [
        'train', 'train', 'validation']
    root = tmp_path / 'Data'
    result = prepare_external_import(root, 'external-one',
                                     expected_input_identity=inspection['input_identity'],
                                     **options)
    assert result['qa_status'] == 'not_run'
    assert (root / '.external-imports' / 'external-one' / 'import.json').is_file()
    meta = validate_package(Path(result['package_path']))
    assert meta['group_source'] == 'explicit_mapping'
    assert [row['frame'] for row in meta['images']] == [1, 1, 2]
    cameras, registered, points = read_model(Path(result['package_path']) / 'colmap')
    assert len(cameras) == len(registered) == 2
    assert len(points) == 4
    assert {item['name'] for item in registered.values()} == {'00000000.png', '00000001.png'}
    adapter = postshot_adapter(Path(result['package_path']), tmp_path / '适配器')
    assert (adapter / 'colmap' / 'images.bin').is_file()
    assert len(list_external_imports(root)['imports']) == 1
    verify_external_import(root, 'external-one')
    grouping.write_text('{}', encoding='utf-8')
    with pytest.raises(RuntimeError, match='source identity changed'):
        verify_external_import(root, 'external-one')


def test_independent_images_requires_explicit_confirmation(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    inspection = inspect_external_import(images=images, model=model,
                                         independent_images=True)
    assert inspection['group_source'] == 'independent_images_confirmed'
    assert not inspection['errors']
    assert inspection['groups'] == 3
    with pytest.raises(RuntimeError, match='identity changed'):
        prepare_external_import(tmp_path / 'Data', 'external-two',
                                expected_input_identity='0' * 64,
                                images=images, model=model, independent_images=True)


def test_incomplete_group_mapping_is_blocking(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    grouping = tmp_path / 'groups.json'
    grouping.write_text(json.dumps({'frame_000001.png': 'a'}), encoding='utf-8')
    inspected = inspect_external_import(images=images, model=model, group_file=grouping)
    assert '精确覆盖' in '\n'.join(inspected['errors'])
    assert inspected['group_source'] == 'required'


def test_multiple_models_require_selection(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    second = model.parent / '1'
    second.mkdir()
    for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
        (second / name).write_bytes((model / name).read_bytes())
    inspected = inspect_external_import(images=images, model=model.parent)
    assert len(inspected['model_candidates']) == 2
    assert '请选择一个' in inspected['errors'][0]
    selected = inspect_external_import(images=images, model=model,
                                       independent_images=True)
    assert not selected['errors']


def test_explicit_parent_directory_never_silently_selects_child_model(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    inspected = inspect_external_import(images=images, model=model.parent,
                                        independent_images=True)
    assert inspected['model'] is None
    assert inspected['model_candidates'] == [str(model.resolve())]
    assert '请选择一个' in inspected['errors'][0]
    with pytest.raises(ValueError, match='请选择一个'):
        prepare_external_import(tmp_path / 'Data', 'wrong-model', images=images,
                                model=model.parent, independent_images=True)


def test_stopped_external_experiment_can_resume(tmp_path: Path, monkeypatch) -> None:
    images, model = _source(tmp_path)
    root = tmp_path / 'Data'
    prepare_external_import(root, 'external-resume', images=images, model=model,
                            independent_images=True)
    import gsstudio.application.external_imports as imports

    calls = []

    def fake_train(_package, output, **kwargs):
        calls.append(kwargs)
        output.mkdir(parents=True, exist_ok=True)
        return {'status': 'stopped' if len(calls) == 1 else 'succeeded',
                'requested_steps': 30, 'model_sha256': 'a' * 64}

    monkeypatch.setattr(imports, 'train_package', fake_train)
    monkeypatch.setattr(imports, '_verify_model', lambda _path, _digest: None)
    stopped = train_external_import(root, 'external-resume', steps=30)
    assert stopped['status'] == 'stopped'
    assert Path(stopped['output']).is_dir()
    resumed = train_external_import(root, 'external-resume', steps=30, resume=True,
                                    experiment_id=stopped['experiment_id'])
    assert resumed['status'] == 'succeeded'
    assert calls[1]['resume'] is True
    assert list_external_imports(root)['imports'][0]['experiments'][0]['status'] == 'succeeded'


def _finished_unrecorded_experiment(tmp_path: Path, steps: int | None = 30) -> tuple[Path, str, Path, Path]:
    images, model = _source(tmp_path)
    root = tmp_path / 'Data'
    import_id = 'external-completed'
    prepared = prepare_external_import(root, import_id, images=images, model=model,
                                       independent_images=True)
    project = Path(prepared['project'])
    experiment_id = 'gsplat-finished'
    output = project / 'experiments' / experiment_id
    output.mkdir(parents=True)
    experiment_path = output.with_suffix('.json')
    requested_steps = steps if steps is not None else 10000
    json_write(experiment_path, {'schema_version': 1, 'id': experiment_id,
                                'import_id': import_id, 'import_identity': prepared['input_identity'],
                                'package_sha256': prepared['package_sha256'],
                                'backend': 'gsplat', 'steps': steps, 'status': 'preparing'})
    row = np.zeros((1, len(FLOAT_PROPERTIES)), dtype='<f4')
    row[0, FLOAT_PROPERTIES.index('rot_0')] = 1
    header = ('ply\nformat binary_little_endian 1.0\n'
              'element vertex 1\n' +
              ''.join(f'property float {name}\n' for name in FLOAT_PROPERTIES) +
              'end_header\n').encode('ascii')
    model_path = output / 'model.ply'
    model_path.write_bytes(header + row.tobytes())
    digest = sha256_file(model_path)
    evaluation = output / 'evaluation'
    evaluation.mkdir()
    validation_images = [row['image'] for row in validate_package(project / 'package')['images']
                         if row['split'] == 'validation']
    json_write(evaluation / 'metrics.json', {'schema_version': 1,
                                             'images': [{'image': name,'psnr':30.,'ssim':.9} for name in validation_images]})
    evaluation_digest = sha256_file(evaluation / 'metrics.json')
    json_write(output / 'dispatch.json', {'backend': 'gsplat', 'status': 'succeeded',
                                          'engine':'local_postshot.studio_training',
                                          'checkpoint_format':'checkpoints/latest.pt',
                                          'training_recipe':'quality_v2','sh_degree':3,
                                          'package_sha256': prepared['package_sha256'],
                                          'requested_steps': requested_steps, 'photo_comp': True,
                                          'use_bilateral_grid': True, 'use_sparse_depth': True,
                                          'model_sha256': digest,
                                          'evaluation_sha256': evaluation_digest})
    json_write(output / 'config.json', {'package_sha256': prepared['package_sha256'],
                                        'steps': requested_steps, 'photo_comp': True,
                                        'use_bilateral_grid': True, 'use_sparse_depth': True})
    json_write(output / 'training.json', {'status': 'succeeded', 'steps': requested_steps,
                                          'model_sha256': digest,
                                          'evaluation_status': 'completed'})
    checkpoint=output/'checkpoints'/'latest.pt'
    checkpoint.parent.mkdir();checkpoint.write_bytes(b'current independent checkpoint')
    cfg=json.loads((output/'config.json').read_text(encoding='utf-8'))
    cfg.update(engine={'name':'local-postshot-independent-gsstudio','schema_version':1,
                       'sources':{'training.py':'0'*64},'versions':{'torch':'2.8.0+cu129'}},
               checkpoint=str(checkpoint),training_recipe='quality_v2')
    from gsstudio.pipeline.training.presets import prepare_training
    _, _, settings = prepare_training(project/'package', steps=steps, source_kind='colmap')
    dispatch=json.loads((output/'dispatch.json').read_text(encoding='utf-8'))
    dispatch.update({k:v for k,v in settings.items() if k!='steps'},
                    command=['python','--dataset',str(project/'package')])
    json_write(output/'dispatch.json',dispatch)
    cfg.update({k:settings[k] for k in ('photo_comp','use_bilateral_grid','use_sparse_depth',
        'sparse_depth_requested','sparse_depth_available','sparse_depth_observations','sparse_depth_status')})
    json_write(output/'config.json',cfg)
    summary=json.loads((output/'training.json').read_text(encoding='utf-8'))
    json_write(output/'training.json',{**cfg,**summary,'completed_steps':requested_steps,
                                      'checkpoint_sha256':sha256_file(checkpoint)})
    return root, import_id, experiment_path, output


@pytest.mark.parametrize('steps', [30, None])
def test_completed_external_experiment_is_adopted_without_retraining(
        tmp_path: Path, monkeypatch, steps: int | None) -> None:
    root, import_id, experiment_path, output = _finished_unrecorded_experiment(tmp_path, steps)
    import gsstudio.application.external_imports as imports

    monkeypatch.setattr(imports, 'train_package', lambda *_args, **_kwargs:
                        pytest.fail('Completed training must not run again'))
    events = []
    result = train_external_import(root, import_id, backend='gsplat', resume=True,
                                   experiment_id=output.name, sink=events.append)
    assert result['status'] == 'succeeded'
    assert result['output'] == str(output)
    assert result['model_path'] == str(output/'model.ply')
    assert json.loads(experiment_path.read_text(encoding='utf-8'))['status'] == 'succeeded'
    assert any(event.kind == 'completed' for event in events)


@pytest.mark.parametrize('changed', ['package', 'training', 'ply', 'evaluation'])
def test_completed_external_experiment_rejects_changed_evidence(
        tmp_path: Path, monkeypatch, changed: str) -> None:
    root, import_id, experiment_path, output = _finished_unrecorded_experiment(tmp_path)
    if changed == 'package':
        dispatch = json.loads((output / 'dispatch.json').read_text(encoding='utf-8'))
        dispatch['package_sha256'] = '0' * 64
        json_write(output / 'dispatch.json', dispatch)
    elif changed == 'training':
        training = json.loads((output / 'training.json').read_text(encoding='utf-8'))
        training['status'] = 'trained'
        json_write(output / 'training.json', training)
    elif changed == 'evaluation':
        (output / 'evaluation' / 'metrics.json').write_text('{}', encoding='utf-8')
    else:
        (output / 'model.ply').write_bytes(b'changed')
    import gsstudio.application.external_imports as imports

    monkeypatch.setattr(imports, 'train_package', lambda *_args, **_kwargs:
                        pytest.fail('Mismatched evidence must not retrain'))
    with pytest.raises(RuntimeError):
        train_external_import(root, import_id, backend='gsplat', resume=True,
                              experiment_id=output.name)
    assert json.loads(experiment_path.read_text(encoding='utf-8'))['status'] == 'failed'


def test_source_relink_keeps_package_immutable(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    root = tmp_path / 'Data'
    result = prepare_external_import(root, 'external-moved', images=images, model=model,
                                     independent_images=True)
    package = Path(result['package_path'])
    before = (package / 'dataset.json').read_bytes()
    old_source = images.parent
    new_source = tmp_path / '搬迁后'
    assert old_source.resolve().is_relative_to(tmp_path.resolve())
    assert new_source.resolve().is_relative_to(tmp_path.resolve())
    old_source.rename(new_source)
    with pytest.raises(FileNotFoundError):
        verify_external_import(root, 'external-moved')
    relink_external_import(root, 'external-moved', new_root=new_source)
    verify_external_import(root, 'external-moved')
    assert (package / 'dataset.json').read_bytes() == before


def test_source_change_during_copy_never_publishes_package(tmp_path: Path, monkeypatch) -> None:
    images, model = _source(tmp_path)
    from gsstudio.pipeline.editor.data import Dataset
    original_prepare = Dataset.prepare

    def mutate_after_copy(self, output, max_size=0):
        result = original_prepare(self, output, max_size)
        (images / 'frame_000001.png').write_bytes(b'changed after copy')
        return result

    monkeypatch.setattr(Dataset, 'prepare', mutate_after_copy)
    root = tmp_path / 'Data'
    with pytest.raises(ValueError, match='输入已经改变'):
        prepare_external_import(root, 'external-race', images=images, model=model,
                                independent_images=True)
    project = root / '.external-imports' / 'external-race'
    assert json.loads((project / 'import.json').read_text(encoding='utf-8'))['status'] == 'failed'
    assert not (project / 'package').exists()
    assert not list(project.glob('.package-building-*'))


def test_worker_uses_same_external_operations(tmp_path: Path) -> None:
    from gsstudio.interfaces.worker.main import execute
    images, model = _source(tmp_path)
    root = tmp_path / 'Data'
    common = {'root': str(root), 'images': str(images), 'model': str(model),
              'independent_images': True}
    inspected = execute({'action': 'inspect_external_import', **common})
    prepared = execute({'action': 'prepare_external_import', 'import_id': 'worker-project',
                        'expected_input_identity': inspected['input_identity'], **common})
    assert prepared['status'] == 'prepared'
    assert execute({'action': 'list_external_imports', 'root': str(root)})['imports'][0]['id'] == 'worker-project'


def test_existing_package_keeps_its_split_and_colmap_model(tmp_path: Path) -> None:
    images, model = _source(tmp_path)
    root = tmp_path / 'Data'
    first = prepare_external_import(root, 'raw-source', images=images, model=model,
                                    independent_images=True)
    first_package = Path(first['package_path'])
    copied = prepare_external_import(root, 'shared-source', source=first_package)
    second_package = Path(copied['package_path'])
    assert copied['source_kind'] == 'colmap'  # Preserve the actual source preset.
    assert [r['split'] for r in validate_package(second_package)['images']] == [
        r['split'] for r in validate_package(first_package)['images']]
    assert (second_package / 'colmap' / 'images.bin').read_bytes() == (
        first_package / 'colmap' / 'images.bin').read_bytes()
    verify_external_import(root, 'shared-source')

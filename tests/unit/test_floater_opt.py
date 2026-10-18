"""CPU tests for floater-optimization: bilateral grid, sparse depth, dispatch flags."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gsstudio.pipeline.training.sparse_depth import export_sparse_depth_anchors, load_sparse_depth, write_sparse_depth


def test_export_sparse_depth_anchors_projects_depth():
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [0.0, 0.0, 2.0]
    point = SimpleNamespace(id=7, xyz=np.array([0.0, 0.0, 0.0]), rgb=np.array([1, 2, 3]), error=0.1,
                            image_ids=np.array([11]), point2D_idxs=np.array([0]))
    image = SimpleNamespace(point3D_ids=np.array([7, -1, 7]), xys=np.array([[1.5, 2.4], [0, 0], [3.0, 4.0]]))
    rows = [dict(world_to_camera=world_to_camera.tolist(), width=10, height=10)]
    anchors = export_sparse_depth_anchors({11: image}, rows, {7: point}, {11: 0})
    assert anchors['row_index'].tolist() == [0, 0]
    assert anchors['depth'].tolist() == pytest.approx([2.0, 2.0])
    assert anchors['u'].tolist() == [2.0, 3.0]
    assert anchors['v'].tolist() == [2.0, 4.0]
    assert anchors['point_id'].tolist() == [7, 7]


def test_export_sparse_depth_anchors_skips_invalid_depth():
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [0.0, 0.0, -1.0]
    point = SimpleNamespace(id=1, xyz=np.array([0.0, 0.0, 0.0]))
    image = SimpleNamespace(point3D_ids=np.array([1]), xys=np.array([[1.0, 1.0]]))
    rows = [dict(world_to_camera=world_to_camera.tolist(), width=8, height=8)]
    anchors = export_sparse_depth_anchors({1: image}, rows, {1: point}, {1: 0})
    assert len(anchors['depth']) == 0


def test_export_sparse_depth_anchors_missing_row():
    point = SimpleNamespace(id=1, xyz=np.zeros(3))
    image = SimpleNamespace(point3D_ids=np.array([1]), xys=np.array([[0.0, 0.0]]))
    with pytest.raises(RuntimeError, match='missing package row'):
        export_sparse_depth_anchors({1: image}, [], {1: point}, {})


def test_sparse_depth_roundtrip(tmp_path: Path):
    anchors = dict(
        row_index=np.array([0, 1], dtype=np.int32),
        u=np.array([1.0, 2.0], dtype=np.float32),
        v=np.array([3.0, 4.0], dtype=np.float32),
        depth=np.array([5.0, 6.0], dtype=np.float32),
        point_id=np.array([8, 9], dtype=np.int64),
    )
    write_sparse_depth(tmp_path / 'sparse_depth.npz', anchors)
    loaded = load_sparse_depth(tmp_path)
    assert loaded is not None
    np.testing.assert_array_equal(loaded['depth'], anchors['depth'])
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert load_sparse_depth(empty) is None


def test_train_package_command_carries_flags(tmp_path: Path, monkeypatch, mock_dispatch_package):
    from gsstudio.pipeline.training import runner as training

    package = tmp_path / 'package'
    package.mkdir()
    meta = {
        'schema_version': 1,
        'validation': 'passed',
        'files': {},
        'images': [
            {'image': 'images/a.png', 'mask': 'masks/a.png', 'frame': 1, 'timestamp_seconds': 0.0,
             'split': 'train', 'width': 4, 'height': 4,
             'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'world_to_camera': np.eye(4).tolist()},
            {'image': 'images/b.png', 'mask': 'masks/b.png', 'frame': 2, 'timestamp_seconds': 1.0,
             'split': 'validation', 'width': 4, 'height': 4,
             'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'world_to_camera': np.eye(4).tolist()},
        ],
    }
    (package / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
    fake_python = tmp_path / 'python.exe'
    fake_python.write_text('stub', encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_GSPLAT_PYTHON', str(fake_python))
    mock_dispatch_package(meta)
    monkeypatch.setattr(training, 'sha256_file', lambda path: 'abc')
    monkeypatch.setattr(training, 'gpu_session', lambda: _nullctx())
    monkeypatch.setattr(training, 'run_logged', lambda *a, **k: {})

    output = tmp_path / 'exp'
    record = training.train_package(package, output, backend='gsplat', steps=10, dry_run=True,
                                    use_bilateral_grid=False, use_sparse_depth=False)
    command = record['command']
    assert '--no-use-bilateral-grid' in command
    assert '--no-use-sparse-depth' in command
    assert record['use_bilateral_grid'] is False
    assert record['use_sparse_depth'] is False

    output2 = tmp_path / 'exp2'
    record2 = training.train_package(package, output2, backend='gsplat', steps=10, dry_run=True)
    assert '--no-use-bilateral-grid' not in record2['command']
    assert '--no-use-sparse-depth' not in record2['command']
    assert record2['use_bilateral_grid'] is True
    assert record2['use_sparse_depth'] is True

    (output / 'dispatch.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(RuntimeError, match='identity'):
        training.train_package(package, output, backend='gsplat', steps=10, dry_run=True, resume=True,
                               use_bilateral_grid=True, use_sparse_depth=False)


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_photometric_import():
    pytest.importorskip('torch', reason='BilateralGrid requires torch')
    from gsstudio.pipeline.training import photometric
    assert hasattr(photometric, 'BilateralGrid')


def test_torch_photometric_and_depth_loss():
    torch = pytest.importorskip('torch', reason='BilateralGrid requires torch')
    from gsstudio.pipeline.training.native import sparse_depth_loss
    from gsstudio.pipeline.training.photometric import BilateralGrid

    layer = BilateralGrid(num_groups=3, spatial=4, bins=4)
    rgb = torch.rand(8, 10, 3)
    for group in range(3):
        out = layer(rgb, group)
        assert out.shape == rgb.shape
        assert torch.allclose(out, rgb, atol=1e-5)

    with pytest.raises(ValueError):
        layer(torch.rand(4, 4), 0)
    with pytest.raises(IndexError):
        layer(torch.rand(4, 4, 3), 3)

    with torch.no_grad():
        layer.grid += 0.1
    value = layer.regularization()
    assert torch.isfinite(value)
    assert float(value.detach()) > 0

    layer2 = BilateralGrid(num_groups=2, spatial=4, bins=4)
    with torch.no_grad():
        layer2.grid[1, 0] = 2.0
    rgb = torch.full((4, 4, 3), 0.25)
    assert torch.allclose(layer2(rgb, 0), rgb, atol=1e-5)
    assert float(layer2(rgb, 1).mean().detach()) > 0.3

    depth = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    anchors = dict(
        row_index=np.array([0, 0, 1], dtype=np.int32),
        u=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        v=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        depth=np.array([0.0, 6.0, 9.0], dtype=np.float32),
    )
    loss, count = sparse_depth_loss(depth, anchors, 0)
    assert count == 2
    assert float(loss) == pytest.approx(1.5, abs=1e-5)  # mean(|1-0|, |4-6|)
    empty, empty_count = sparse_depth_loss(depth, anchors, 99)
    assert empty_count == 0
    assert float(empty) == 0.0


def test_train_package_dispatch_identity(tmp_path: Path, monkeypatch, mock_dispatch_package):
    """A resume must match the recorded experiment exactly, flags included."""
    from gsstudio.pipeline.training import runner as training

    package = tmp_path / 'package'
    package.mkdir()
    meta = {
        'schema_version': 1,
        'validation': 'passed',
        'files': {},
        'images': [
            {'image': 'images/a.png', 'mask': 'masks/a.png', 'frame': 1, 'timestamp_seconds': 0.0,
             'split': 'train', 'width': 4, 'height': 4,
             'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'world_to_camera': np.eye(4).tolist()},
            {'image': 'images/b.png', 'mask': 'masks/b.png', 'frame': 2, 'timestamp_seconds': 1.0,
             'split': 'validation', 'width': 4, 'height': 4,
             'K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'world_to_camera': np.eye(4).tolist()},
        ],
    }
    (package / 'dataset.json').write_text(json.dumps(meta), encoding='utf-8')
    fake_python = tmp_path / 'python.exe'
    fake_python.write_text('stub', encoding='utf-8')
    monkeypatch.setenv('GSSTUDIO_GSPLAT_PYTHON', str(fake_python))
    mock_dispatch_package(meta)
    monkeypatch.setattr(training, 'sha256_file', lambda path: 'abc')
    monkeypatch.setattr(training, 'gpu_session', lambda: _nullctx())
    monkeypatch.setattr(training, 'run_logged', lambda *a, **k: {})

    # A dispatch record without the photometric flags is a different experiment.
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'dispatch.json').write_text(json.dumps(dict(
        backend='gsplat', package_sha256='abc', command=[], requested_steps=10,
        photo_comp=True, status='prepared')), encoding='utf-8')
    with pytest.raises(RuntimeError, match='identity'):
        training.train_package(package, legacy, backend='gsplat', steps=10, dry_run=True, resume=True,
                               use_bilateral_grid=False, use_sparse_depth=False)

    # An exact match resumes and reports the same flags.
    matching = tmp_path / 'matching'
    training.train_package(package, matching, steps=10, dry_run=True,
                           use_bilateral_grid=False, use_sparse_depth=False)
    record = training.train_package(package, matching, backend='gsplat', steps=10, dry_run=True,
                                    resume=True, use_bilateral_grid=False, use_sparse_depth=False,restart_from_step_zero=True)
    assert record['use_bilateral_grid'] is False
    assert record['use_sparse_depth'] is False

    # A different flag value is a different experiment.
    changed = tmp_path / 'changed'
    changed.mkdir()
    (changed / 'dispatch.json').write_text(json.dumps(dict(
        backend='gsplat', package_sha256='abc', command=[], requested_steps=10,
        photo_comp=True, status='prepared',
        engine=training.ENGINE_ID,checkpoint_format=training.CHECKPOINT,
        training_recipe='quality_v2',sh_degree=3,
        use_bilateral_grid=True, use_sparse_depth=False)), encoding='utf-8')
    with pytest.raises(RuntimeError, match='identity'):
        training.train_package(package, changed, backend='gsplat', steps=10, dry_run=True,
                               resume=True, use_bilateral_grid=False, use_sparse_depth=False,restart_from_step_zero=True)


def test_select_photometric_routing():
    torch = pytest.importorskip('torch', reason='photometric modules require torch')
    from gsstudio.pipeline.training.native import PanoramaExposure, select_photometric
    from gsstudio.pipeline.training.photometric import BilateralGrid

    times = [0.0, 1.0]
    assert select_photometric(False, True, 2, times) is None
    assert isinstance(select_photometric(True, True, 2, times), BilateralGrid)
    assert isinstance(select_photometric(True, False, 2, times), PanoramaExposure)


def test_config_match():
    from gsstudio.pipeline.training.native import _configs_match

    current = dict(schema_version=2, package_sha256='abc', steps=10, photo_comp=True,
                   antialiased=False, sh_degree=3, seed=1, high_order_l2=1e-6, checkpoint_every=10,
                   use_bilateral_grid=True, use_sparse_depth=True, sparse_depth_weight=0.1,
                   sparse_depth_status='enabled')
    assert _configs_match(current, dict(current))
    # What the package actually contained is not part of the identity.
    assert _configs_match(current, dict(current, sparse_depth_status='empty'))
    assert not _configs_match(current, dict(current, use_bilateral_grid=False))
    assert not _configs_match(current, dict(current, sparse_depth_weight=0.0))
    # A record written before the flags existed is not a match.
    assert not _configs_match({k: v for k, v in current.items() if not k.startswith('use_')
                               and k != 'sparse_depth_weight'}, current)

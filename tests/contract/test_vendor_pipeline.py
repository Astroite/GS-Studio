"""CPU integration for the migrated data boundary; no trainer or external tools."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from PIL import Image

from gsstudio._vendor.nerfstudio import colmap_parsing_utils as parser
from gsstudio._vendor.nerfstudio import colmap_utils
from gsstudio.domain.models import ReconstructionConfig, SegmentQAConfig
from gsstudio.pipeline.reconstruction import realityscan
from gsstudio.pipeline.reconstruction.core import _write_colmap_binary_model
from gsstudio.pipeline.training import data as training_data


def test_segment_package_validates_original_poses_and_training_only_colors(tmp_path, monkeypatch):
    dataset = tmp_path/'source'
    camera = parser.Camera(1, 'PINHOLE', 20, 20, np.array([10., 10., 10., 10.]))
    records = [dict(timestamp_seconds=float(i)) for i in range(12)]
    images, names = {}, []
    for index in range(12):
        name = f'view_00/frame_{index + 1:06d}.png'
        names.append(name)
        for folder in ('images', 'masks'):
            (dataset/folder/'view_00').mkdir(parents=True, exist_ok=True)
        # Frame 8 is the held-out temporal group and must not affect colors.
        Image.new('RGB', (20, 20), (255, 255, 255) if index == 7 else (127, 127, 127)).save(dataset/'images'/name)
        Image.new('L', (20, 20), 255).save(dataset/'masks'/f'{name}.png')
        images[index + 1] = parser.Image(index + 1, np.array([1., 0., 0., 0.]),
            np.array([float(index), 0., 0.]), 1, name,
            np.array([[3., 3.], [5., 5.], [7., 7.], [9., 9.]]), np.array([1, 2, 3, 4]))
    points = {key: parser.Point3D(key, np.array([float(key), 0., 3.]), np.array([250, 250, 250]),
              0., np.arange(1, 13), np.full(12, key-1)) for key in (1, 2, 3, 4)}
    _write_colmap_binary_model(dataset/'colmap', {1: camera}, images, points)
    colmap_utils.colmap_to_json(dataset/'colmap', dataset, use_single_camera_mode=False)
    transforms = json.loads((dataset/'transforms.json').read_text(encoding='utf-8'))
    # This uses the real converted applied_transform, including the world-axis swap.
    assert transforms['applied_transform'] == [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.]]
    report = dict(training_status='passed', coverage_status='complete', segments=[dict(
        id='segment-001', status='passed', images=names, frames=list(range(1, 13)))])
    monkeypatch.setattr(training_data, 'validate_segments', lambda *a, **k: report)
    monkeypatch.setattr(training_data, 'validate_mask_filter', lambda *a, **k: dict(accepted=[{'image': name} for name in names]))
    monkeypatch.setattr(training_data, 'validate_mask_finalization', lambda *a, **k: {})
    original = {path: path.read_bytes() for path in dataset.rglob('*') if path.is_file()}
    result = training_data.prepare_segment(dataset, records, ReconstructionConfig().primary,
        SegmentQAConfig(), 'segment-001', tmp_path/'package')
    assert result['initial_points'] == 4
    assert any(row['split'] == 'validation' for row in result['images'])
    with np.load(tmp_path/'package'/'points.npz', allow_pickle=False) as initialized:
        np.testing.assert_array_equal(initialized['rgb'], np.full((4, 3), 127, np.uint8))
    assert all(path.read_bytes() == content for path, content in original.items())
    transforms['frames'][0]['transform_matrix'][0][3] = 1234
    training_data.json_write(dataset/'transforms.json', transforms)
    with pytest.raises(RuntimeError, match='pose/COLMAP mismatch'):
        training_data.prepare_segment(dataset, records, ReconstructionConfig().primary,
            SegmentQAConfig(), 'segment-001', tmp_path/'bad-package')
    assert not (tmp_path/'bad-package').exists()


def test_realityscan_conversion_mock_targets_local_module(tmp_path, monkeypatch):
    dataset = tmp_path/'dataset'
    dataset.mkdir()
    source = Path(__file__).resolve().parents[1]/'fixtures'/'nerfstudio_vendor'/'conversion'/'single_default'/'model'
    names = {image.name for image in parser.read_images_binary(source/'images.bin').values()}
    for name in names:
        for folder in ('images', 'masks'):
            path = dataset/folder/(f'{name}.png' if folder == 'masks' else name)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('L' if folder == 'masks' else 'RGB', (36, 18), 255).save(path)
    monkeypatch.setattr(realityscan, 'realityscan_executable', lambda: tmp_path/'RealityScan.exe')
    monkeypatch.setattr(realityscan, 'build_realityscan_align_command', lambda *a, **k: ['synthetic-cpu-tool'])
    def export(command, log):
        output = dataset/'colmap'/'realityscan-export'
        output.mkdir(parents=True, exist_ok=True)
        for name in ('cameras.txt', 'images.txt', 'points3D.txt'):
            shutil.copy2(source/name, output/name)
        exported = (output/'images.txt').read_text(encoding='utf-8')
        for index, name in enumerate(sorted(names)):
            exported = exported.replace(name, f'gsdb_{index:06d}.jpg')
        (output/'images.txt').write_text(exported, encoding='utf-8')
    monkeypatch.setattr(realityscan, 'run_logged', export)
    # The conversion body is exercised by parity tests; this verifies that the
    # pipeline's local import and existing mocking boundary resolve correctly.
    calls = []
    actual = colmap_utils.colmap_to_json
    def convert(*args, **kwargs):
        calls.append((args, kwargs))
        return actual(*args, **kwargs)
    monkeypatch.setattr('gsstudio._vendor.nerfstudio.colmap_utils.colmap_to_json', convert)
    monkeypatch.setattr(realityscan, 'realityscan_reconstruction_metrics', lambda *a, **k: {'registered_images': 4})
    original = {path: path.read_bytes() for folder in ('images', 'masks') for path in (dataset/folder).rglob('*') if path.is_file()}
    result = realityscan.run_realityscan_alignment(dataset, ReconstructionConfig().primary,
        tmp_path/'logs', ReconstructionConfig(), included_images=names)
    assert result['registered_images'] == 4 and len(calls) == 1
    assert calls[0][1] == {'use_single_camera_mode': False}
    assert not (dataset/'colmap'/'realityscan-export').exists()
    assert all(path.read_bytes() == content for path, content in original.items())
    assert all(frame['mask_path'].startswith('./masks/') for frame in json.loads(
        (dataset/'transforms.json').read_text(encoding='utf-8'))['frames'])

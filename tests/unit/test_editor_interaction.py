"""Picking and one-action edit history, independent of model files or CUDA."""
from __future__ import annotations

import numpy as np
import pytest

from gsstudio.pipeline.editor.edit import Editor
from gsstudio.pipeline.editor.runtime import Runtime
from gsstudio.pipeline.editor.visibility import compositing_weights
from gsstudio.pipeline.editor.ply import FLOAT_PROPERTIES, read_ply_header


def _editor():
    editor = object.__new__(Editor)
    editor.data = np.zeros((3, 62), dtype=np.float32)
    editor.data[:, 2] = [1, 2, 3]
    editor.keep = np.ones(3, dtype=bool)
    editor.selected = np.zeros(3, dtype=bool)
    editor.matrix = np.eye(4)
    editor.history, editor.future = [], []
    editor.revision = 0
    return editor


def _camera():
    return {"world_to_camera": np.eye(4).tolist(), "K": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            "width": 100, "height": 100}


def test_nonthrough_requires_rendered_visibility_and_through_does_not():
    editor = _editor()
    with pytest.raises(ValueError, match="可见贡献"):
        editor.select(_camera(), [40, 40, 60, 60], False)
    editor.select(_camera(), [40, 40, 60, 60], False, np.array([True, False, True]))
    assert editor.selected.tolist() == [True, False, True]
    editor.select(_camera(), [40, 40, 60, 60], True)
    assert editor.selected.tolist() == [True, True, True]


def test_visibility_weights_reset_for_each_pixel():
    import torch

    alpha = torch.tensor([.5, .5, .5])
    pixels = torch.tensor([0, 0, 1])
    assert torch.allclose(compositing_weights(alpha, pixels), torch.tensor([.5, .25, .5]))


def test_committed_transform_and_crop_each_take_one_undo_step():
    editor = _editor()
    editor.transform((1, 0, 0), (0, 0, 0), 1, pivot=(0, 0, 2))
    assert len(editor.history) == 1
    assert np.allclose(editor.xyz()[:, 0], 1)
    editor.crop((-1, -1, 0), (2, 2, 2.5), True, True)
    assert len(editor.history) == 2
    assert editor.keep.tolist() == [True, True, False]
    editor.undo()
    assert editor.keep.all()
    editor.undo()
    assert np.allclose(editor.xyz()[:, 0], 0)


def test_cancelled_preview_does_not_touch_edit_history_or_original(tmp_path):
    editor = _editor()
    runtime = object.__new__(Runtime)
    runtime.editor = editor
    runtime.preview_matrix = None
    runtime.preview_keep = None
    runtime.edit_preview_revision = 0
    runtime.preview_revision = 0
    runtime.set_edit_preview(transform=((2, 0, 0), (0, 0, 0), 1, (0, 0, 2)))
    assert runtime.preview_matrix is not None
    assert not editor.history
    assert np.allclose(editor.xyz()[:, 0], 0)
    runtime.clear_edit_preview()
    assert runtime.preview_matrix is None
    assert not editor.history
    original = tmp_path / "original.ply"
    original.write_bytes(b"immutable model")
    with pytest.raises(FileExistsError):
        editor.export(original)
    assert original.read_bytes() == b"immutable model"


def test_edited_ply_roundtrip_preserves_source_and_properties(tmp_path):
    original = tmp_path / "original.ply"
    rows = np.zeros((2, len(FLOAT_PROPERTIES)), dtype="<f4")
    rows[:, 2] = [1, 2]
    rows[:, FLOAT_PROPERTIES.index("rot_0")] = 1
    rows[:, FLOAT_PROPERTIES.index("f_dc_0")] = [0.2, 0.4]
    header = ("ply\nformat binary_little_endian 1.0\n"
              "element vertex 2\n" +
              "".join(f"property float {name}\n" for name in FLOAT_PROPERTIES) +
              "end_header\n").encode("ascii")
    original.write_bytes(header + rows.tobytes())
    source_bytes = original.read_bytes()

    editor = Editor(original)
    editor.transform((1, 0, 0))
    editor.crop((0, -1, 0), (2, 1, 1.5), commit=True)
    exported = tmp_path / "edited.ply"
    editor.export(exported)

    reread = Editor(exported)
    _, count, properties = read_ply_header(exported)
    assert count == 1
    assert properties == list(FLOAT_PROPERTIES)
    assert np.allclose(reread.xyz(), [[1, 0, 1]])
    assert np.isclose(reread.data[0, FLOAT_PROPERTIES.index("f_dc_0")], 0.2)
    assert original.read_bytes() == source_bytes

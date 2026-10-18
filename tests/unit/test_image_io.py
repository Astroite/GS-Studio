from pathlib import Path

import cv2
import numpy as np
import pytest

from gsstudio.infrastructure.adapters import image_io


def test_unicode_image_and_mask_roundtrip(tmp_path: Path):
    root = tmp_path / "中文 Data" / "场景"
    color = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    mask = np.where(color[:, :, 0] % 2, 255, 0).astype(np.uint8)
    for name, pixels, flags in (("源图.png", color, cv2.IMREAD_COLOR),
                                 ("遮罩.png", mask, cv2.IMREAD_GRAYSCALE)):
        target = root / name
        assert image_io.write_cv_image(target, pixels)
        assert target.is_file()
        np.testing.assert_array_equal(image_io.read_cv_image(target, flags), pixels)
    assert not list(root.glob("*.tmp"))


def test_missing_or_corrupt_image_stays_a_decode_failure(tmp_path: Path):
    assert image_io.read_cv_image(tmp_path / "不存在.png") is None
    broken = tmp_path / "损坏.png"
    broken.write_bytes(b"not an image")
    assert image_io.read_cv_image(broken) is None


def test_failed_codec_cannot_publish_a_successful_file(tmp_path: Path, monkeypatch):
    target = tmp_path / "输出.png"
    monkeypatch.setattr(image_io.cv2, "imencode", lambda *args: (False, None))
    with pytest.raises(RuntimeError, match="Failed to encode"):
        image_io.write_cv_image(target, np.zeros((2, 2), np.uint8))
    assert not target.exists()


def test_lpips_uses_its_bundled_package_path_in_an_unrelated_cwd(tmp_path: Path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from gsstudio.pipeline.training import native

    package = tmp_path / "frozen" / "_internal" / "lpips"
    weights = package / "weights" / "v0.1" / "alex.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"packaged weights")

    class Metric:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def create_metric(*, net, spatial, model_path):
        assert net == "alex" and spatial is True
        assert Path(model_path) == weights
        return Metric()

    monkeypatch.setitem(sys.modules, "lpips", SimpleNamespace(__file__=str(package / "__init__.py"), LPIPS=create_metric))
    monkeypatch.chdir(tmp_path)
    result = native.evaluate(tmp_path, {"images": []}, {"means": torch.zeros((1, 3))}, tmp_path / "评估")
    assert result["images"] == []

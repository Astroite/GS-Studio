"""Preview requests and publications keep a single experiment/camera identity."""
from __future__ import annotations

import json

import numpy as np
import pytest

from gsstudio.pipeline.training.data import json_write
from gsstudio.pipeline.training.preview import publish, read_request, scaled_camera
from gsstudio.infrastructure.adapters.media import sha256_file


def test_request_must_match_package_and_camera(tmp_path):
    rows = [{"image": "images/a.png", "split": "validation", "width": 800, "height": 400,
             "K": [[500, 0, 400], [0, 500, 200], [0, 0, 1]]}]
    request = {"schema_version": 1, "package_sha256": "abc", "request_id": "a" * 32,
               "image": "images/a.png"}
    json_write(tmp_path / "preview-request.json", request)
    assert read_request(tmp_path, "abc", rows)["row"] is rows[0]
    assert scaled_camera(rows[0])["width"] == 640
    with pytest.raises(ValueError, match="identity"):
        read_request(tmp_path, "other", rows)
    request["image"] = "images/foreign.png"
    json_write(tmp_path / "preview-request.json", request)
    with pytest.raises(ValueError, match="outside"):
        read_request(tmp_path, "abc", rows)


def test_publication_points_to_matching_complete_pair(tmp_path):
    row = {"image": "images/a.png", "split": "validation"}
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    publish(tmp_path, request_id="a" * 32, row=row, step=10, target=100,
            source_bgr=image, render_rgb=image)
    first = json.loads((tmp_path / "preview.json").read_text(encoding="utf-8"))
    assert first["request_id"] == "a" * 32
    assert (tmp_path / first["source_file"]).is_file()
    assert (tmp_path / first["render_file"]).is_file()
    assert sha256_file(tmp_path / first["render_file"]) == first["render_sha256"]
    publish(tmp_path, request_id="b" * 32, row=row, step=11, target=100,
            source_bgr=image, render_rgb=image)
    second = json.loads((tmp_path / "preview.json").read_text(encoding="utf-8"))
    assert second["request_id"] == "b" * 32
    assert (tmp_path / second["source_file"]).is_file()
    assert second["slot"] != first["slot"]
    assert sha256_file(tmp_path / second["render_file"]) == second["render_sha256"]

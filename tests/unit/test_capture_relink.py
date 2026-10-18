"""Capture source relocation must preserve content identity and the Run tree."""
from __future__ import annotations

import shutil

import pytest

from tests.factories import RUN_ID, build_tree
from gsstudio.application.captures import relink_capture_sources, verify_capture_sources
from gsstudio.infrastructure.adapters.media import sha256_file
from gsstudio.infrastructure.persistence.manifests import load_capture_manifest, save_yaml


def test_capture_relink_checks_hash_before_updating_manifest(tmp_path):
    tree = build_tree(tmp_path)
    root = tree["root"]
    manifest_path = root / "loc" / "scene" / "captures" / "capture.yaml"
    capture = load_capture_manifest(manifest_path)
    original = tmp_path / "raw" / "clip.mp4"
    capture.source.files[0].sha256 = sha256_file(original)
    save_yaml(manifest_path, capture)
    manifest_before = manifest_path.read_bytes()

    moved = tmp_path / "relocated" / "clip.mp4"
    moved.parent.mkdir()
    shutil.copyfile(original, moved)
    moved.write_bytes(b"x" * original.stat().st_size)
    with pytest.raises(RuntimeError, match="recorded hash"):
        relink_capture_sources(root, "loc", "scene", "capture", [moved])
    assert manifest_path.read_bytes() == manifest_before

    shutil.copyfile(original, moved)
    updated = relink_capture_sources(root, "loc", "scene", "capture", [moved])
    assert updated.source.files[0].windows_path == str(moved)
    verify_capture_sources(updated)
    assert original.read_bytes() == b"0" * 10
    assert (root / "loc" / "scene" / RUN_ID / "manifest.yaml").is_file()

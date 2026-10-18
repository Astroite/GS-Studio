"""CPU tests for 360 viewport split overlap rules."""
from __future__ import annotations

import math

import pytest

from gsstudio.domain.models import PanoramaProjection, ReconstructionAttempt
from gsstudio.pipeline.reconstruction.core import (
    _angular_separation_degrees,
    expected_planar_images,
    projection_overlap_summary,
    projection_view_specs,
)


def _attempt(views: int, fov: float, crop_bottom: float = 0.15):
    return ReconstructionAttempt(
        temporal_rank_limit=1,
        projection_fov_degrees=fov,
        panorama=PanoramaProjection(views=views, size=256, crop_bottom=crop_bottom),
    )


def test_separation_same_point_is_zero():
    assert _angular_separation_degrees((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert _angular_separation_degrees((10.0, 0.0), (-350.0, 0.0)) == pytest.approx(0.0, abs=1e-6)


def test_separation_orthogonal_is_ninety():
    assert _angular_separation_degrees((0.0, 0.0), (90.0, 0.0)) == pytest.approx(90.0, abs=1e-6)


def test_separation_at_pitch_shrinks_yaw_step():
    # At 45° pitch a 90° yaw step is only 60° apart on the sphere.
    assert _angular_separation_degrees((0.0, 45.0), (90.0, 45.0)) == pytest.approx(60.0, abs=1e-6)


def test_default_14x110_meets_cross_view_rule():
    summary = projection_overlap_summary(_attempt(14, 110.0))
    assert summary['view_count'] == 14
    assert summary['meets_cross_view_rule'] is True
    assert summary['min_adjacent_overlap_degrees'] >= 15.0
    # Middle ring (step 60° at near-equator) keeps ~50° yaw overlap.
    assert summary['min_adjacent_yaw_overlap_degrees'] >= 20.0


def test_eight_view_without_crop_meets_rule():
    # Outer ring 180° yaw step at ±45° pitch: great-circle sep is 90°, overlap 30°.
    summary = projection_overlap_summary(_attempt(8, 120.0, crop_bottom=0.0))
    assert summary['view_count'] == 8
    assert summary['meets_cross_view_rule'] is True
    assert summary['min_adjacent_overlap_degrees'] >= 15.0


def test_eight_view_heavy_bottom_crop_weakens_two_view_ring():
    # crop_bottom 0.20 collapses the lower ring to 2 near-equator views 180° apart.
    summary = projection_overlap_summary(_attempt(8, 120.0, crop_bottom=0.20))
    assert summary['view_count'] == 8
    assert summary['meets_cross_view_rule'] is False
    assert summary['min_adjacent_overlap_degrees'] < 15.0


def test_specs_match_layout_counts_without_crop():
    specs = projection_view_specs(_attempt(14, 110.0, crop_bottom=0.0))
    pitches = sorted({round(pitch, 6) for _, pitch in specs})
    assert pitches == [-45.0, 0.0, 45.0]
    middle = [pair for pair in specs if pair[1] == 0.0]
    outer = [pair for pair in specs if pair[1] != 0.0]
    assert len(middle) == 6
    assert len(outer) == 8


def test_narrow_fov_fails_rule():
    summary = projection_overlap_summary(_attempt(14, 50.0, crop_bottom=0.0))
    assert summary['meets_cross_view_rule'] is False
    assert summary['min_adjacent_overlap_degrees'] < 15.0


def test_overlap_formula_matches_spherical_geometry():
    # Documented identity used in CURRENT-WORKFLOW / CAPTURE-SOP.
    phi = math.radians(45.0)
    d_lambda = math.radians(90.0)
    sep = math.degrees(math.acos(math.sin(phi) ** 2 + math.cos(phi) ** 2 * math.cos(d_lambda)))
    assert sep == pytest.approx(60.0, abs=1e-6)
    assert 110.0 - sep == pytest.approx(50.0, abs=1e-6)


def test_perspective_source_is_one_view_per_frame():
    attempt = ReconstructionAttempt(temporal_rank_limit=1, projection_fov_degrees=84.0)
    assert attempt.panorama is None
    assert expected_planar_images(attempt, 37) == 37
    with pytest.raises(ValueError):
        projection_view_specs(attempt)


def test_panorama_multiplies_frames_by_views():
    assert expected_planar_images(_attempt(14, 110.0), 10) == 140
    assert expected_planar_images(_attempt(8, 120.0, crop_bottom=0.0), 10) == 80

"""Tests for focal length divergence estimation."""

import math

import numpy as np
import pytest

from pps.calibration.focal_divergence import (
    FocalDivergenceResult,
    _focal_from_orthogonal_vps,
    estimate_focal_divergence,
)


def test_focal_from_orthogonal_vps_basic():
    """Known orthogonal VPs at known focal length should recover f."""
    # For orthogonal VPs with focal length f and principal point (cx, cy):
    # f^2 = -(v1x - cx)(v2x - cx) - (v1y - cy)(v2y - cy)
    # Place vp1 at (cx + f, cy + f) and vp2 at (cx + f, cy - f)
    # f^2 = -(f)(f) - (f)(-f) = -f^2 + f^2 = 0  — no good.
    # Instead: vp1 = (cx + f, cy), vp2 = (cx, cy + f)
    # f^2 = -(f)(0) - (0)(f) = 0 — also zero.
    # Correct setup: vp1 = (cx + a, cy + b), vp2 = (cx + c, cy + d)
    # where a*c + b*d = -f^2  (orthogonality constraint)
    # Simple: vp1 = (cx + f, cy + 0), vp2 = (cx + 0, cy + f)
    # f^2 = -(f)(0) - (0)(f) = 0. Still zero.
    # The formula works for VPs that are NOT at infinity in the same plane.
    # Use: vp1 = (cx + f*tan(45), cy), vp2 = (cx, cy + f*tan(45))
    # Actually, the standard result is: for two orthogonal VPs v1, v2:
    # (v1-p).(v2-p) = -f^2  where p=(cx,cy)
    # So: (v1x-cx)(v2x-cx) + (v1y-cy)(v2y-cy) = -f^2
    # f^2 = -[(v1x-cx)(v2x-cx) + (v1y-cy)(v2y-cy)]
    # Set v1=(cx+500, cy+300), v2=(cx-300, cy+500) ->
    # (500)(-300) + (300)(500) = -150000 + 150000 = 0. Not helpful.
    # Need: v1=(cx+a, cy+b), v2=(cx+c, cy+d), a*c+b*d = -f^2
    # Try: v1=(cx+f, cy+100), v2=(cx+100, cy-f) ->
    # f*100 + 100*(-f) = 0. Still 0.
    # The key: we need NON-symmetric VP placement.
    # v1=(cx+1000, cy+200), v2=(cx-100, cy+400)
    # dot = (1000)(-100) + (200)(400) = -100000+80000 = -20000
    # f = sqrt(20000) ≈ 141.4
    cx, cy = 320.0, 240.0
    f_expected = 141.421356
    vp1 = (cx + 1000, cy + 200)
    vp2 = (cx - 100, cy + 400)
    f_sq = _focal_from_orthogonal_vps(vp1, vp2, cx, cy)
    assert f_sq is not None
    assert abs(math.sqrt(f_sq) - f_expected) < 1.0


def test_focal_from_non_orthogonal_returns_none():
    """Non-orthogonal VPs should yield f^2 <= 0."""
    # Two VPs on the same side of the principal point
    cx, cy = 320.0, 240.0
    vp1 = (1000.0, 240.0)
    vp2 = (900.0, 240.0)
    f_sq = _focal_from_orthogonal_vps(vp1, vp2, cx, cy)
    # Should be negative (not orthogonal)
    assert f_sq is None or f_sq <= 0


def test_estimate_focal_divergence_too_few_lines():
    """Blank image with no lines -> empty result, no crash."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = estimate_focal_divergence(image, grid_size=2)
    assert isinstance(result, FocalDivergenceResult)
    assert result.num_valid_regions == 0
    assert result.focal_mean == 0.0


def test_estimate_focal_divergence_runs_on_real_pattern():
    """Synthetic image with lines should produce some result without crashing."""
    # Draw a simple grid pattern
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    # Horizontal lines
    for y in range(50, 480, 50):
        image[y : y + 2, :, :] = 0
    # Vertical lines
    for x in range(50, 640, 50):
        image[:, x : x + 2, :] = 0

    result = estimate_focal_divergence(image, grid_size=2, min_lines_per_region=4)
    assert isinstance(result, FocalDivergenceResult)
    assert result.num_total_regions == 4

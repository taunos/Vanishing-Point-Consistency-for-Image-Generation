"""Tests for field consistency metrics."""

import numpy as np
import pytest

from pps.fields.field_consistency import (
    FieldConsistencyResult,
    compute_field_consistency,
    _compute_latitude_consistency,
    _compute_up_consistency,
)
from pps.fields.perspective_wrapper import PerspectiveFieldResult


def _make_result(latitude: np.ndarray, gravity: np.ndarray | None = None) -> PerspectiveFieldResult:
    h, w = latitude.shape
    if gravity is None:
        # Default: uniform upward gravity
        gravity = np.zeros((h, w, 2), dtype=np.float64)
        gravity[:, :, 1] = -1.0  # pointing up
    return PerspectiveFieldResult(
        latitude=latitude,
        gravity=gravity,
        image_shape=(h, w),
    )


def test_constant_field_high_score():
    """Constant latitude field -> all gradients zero, score near 1.0."""
    lat = np.full((120, 160), 10.0, dtype=np.float64)
    result = _make_result(lat)
    fc = compute_field_consistency(result)
    assert fc.latitude_std < 0.01
    assert fc.gradient_x_std < 0.01
    assert fc.gradient_y_std < 0.01
    assert fc.field_consistency_score > 0.8


def test_random_field_low_score():
    """Random latitude field -> high gradients, low score."""
    rng = np.random.RandomState(42)
    lat = rng.uniform(-45, 45, (120, 160)).astype(np.float64)
    result = _make_result(lat)
    fc = compute_field_consistency(result)
    assert fc.latitude_std > 10.0
    assert fc.gradient_x_std > 0.1
    assert fc.field_consistency_score < 0.35


def test_linear_vertical_gradient():
    """Linear vertical gradient (like a real photo) -> low grad_x_std, moderate lat_std."""
    h, w = 120, 160
    lat = np.linspace(-20, 30, h)[:, np.newaxis] * np.ones((1, w))
    result = _make_result(lat.astype(np.float64))
    fc = compute_field_consistency(result)
    # Horizontal gradients should be near zero
    assert fc.gradient_x_std < 0.001
    # Vertical gradient should be consistent
    assert fc.gradient_y_std < 0.01
    # Overall std should be moderate
    assert fc.latitude_std > 5.0


def test_horizontal_flip_symmetry():
    """Horizontally flipped image should have same gradient_x_std."""
    rng = np.random.RandomState(123)
    lat = rng.uniform(-30, 30, (120, 160)).astype(np.float64)
    lat_flipped = lat[:, ::-1].copy()

    r1 = _make_result(lat)
    r2 = _make_result(lat_flipped)
    fc1 = compute_field_consistency(r1)
    fc2 = compute_field_consistency(r2)
    assert abs(fc1.gradient_x_std - fc2.gradient_x_std) < 0.001


def test_grid_size_smooth():
    """Metrics should change smoothly with grid_size."""
    rng = np.random.RandomState(42)
    lat = rng.uniform(-20, 20, (120, 160)).astype(np.float64)
    result = _make_result(lat)

    scores = []
    for gs in [3, 4, 5, 6]:
        fc = compute_field_consistency(result, grid_size=gs)
        scores.append(fc.field_consistency_score)

    # No dramatic jumps
    for i in range(len(scores) - 1):
        assert abs(scores[i] - scores[i + 1]) < 0.3, f"Jump between grid {3+i} and {4+i}: {scores}"


def test_up_consistency_aligned():
    """All gravity vectors pointing the same way -> near-zero angular disagreement."""
    gravity = np.zeros((120, 160, 2), dtype=np.float64)
    gravity[:, :, 1] = -1.0
    metrics = _compute_up_consistency(gravity, grid_size=4)
    assert metrics["up_angle_mean"] < 0.01
    assert metrics["up_angle_max"] < 0.01


def test_up_consistency_random():
    """Random gravity directions -> high angular disagreement."""
    rng = np.random.RandomState(42)
    gravity = rng.randn(120, 160, 2).astype(np.float64)
    # Normalize per pixel
    norms = np.linalg.norm(gravity, axis=-1, keepdims=True)
    gravity = gravity / np.maximum(norms, 1e-8)
    metrics = _compute_up_consistency(gravity, grid_size=4)
    assert metrics["up_angle_mean"] > 0.3

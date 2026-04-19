"""Tests for the combined PPS score."""

import numpy as np
import pytest

from pps.fields.perspective_wrapper import PerspectiveFieldResult
from pps.scoring.pps_score import _compute_consistency_map


def test_consistency_map_shape():
    """Consistency map has same shape as latitude field."""
    lat = np.random.RandomState(42).uniform(-30, 30, (120, 160)).astype(np.float64)
    cmap = _compute_consistency_map(lat, grid_size=4)
    assert cmap.shape == lat.shape


def test_consistency_map_constant_field():
    """Constant field -> near-zero inconsistency everywhere."""
    lat = np.full((120, 160), 10.0, dtype=np.float64)
    cmap = _compute_consistency_map(lat, grid_size=4)
    assert np.max(cmap) < 0.1


def test_consistency_map_detects_discontinuity():
    """Step discontinuity should produce high inconsistency at the boundary."""
    lat = np.full((120, 160), 0.0, dtype=np.float64)
    lat[60:, :] = 30.0  # sharp step
    cmap = _compute_consistency_map(lat, grid_size=4)
    # Region near the step should have higher inconsistency
    mid_region = cmap[50:70, :]
    edge_region = cmap[0:20, :]
    assert np.mean(mid_region) > np.mean(edge_region)

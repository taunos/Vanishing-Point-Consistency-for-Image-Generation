"""Vanishing-point helper functions for the baseline estimator."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from pcs.geometry.lines import (
    normalize_line_angle,
    segment_midpoint,
    segment_to_homogeneous,
    smallest_undirected_angle_difference,
)
from pcs.geometry.types import LineSegment


def intersect_segments_as_lines(
    line_a: LineSegment,
    line_b: LineSegment,
    min_angle_difference_rad: float,
) -> tuple[float, float] | None:
    """Intersect two segments as infinite lines, ignoring near-parallel pairs."""

    if (
        smallest_undirected_angle_difference(line_a.angle_rad, line_b.angle_rad)
        < min_angle_difference_rad
    ):
        return None

    hom_a = segment_to_homogeneous(line_a)
    hom_b = segment_to_homogeneous(line_b)
    point = np.cross(hom_a, hom_b)
    if abs(point[2]) < 1e-9:
        return None
    x = float(point[0] / point[2])
    y = float(point[1] / point[2])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


def angular_residual_to_vp(
    segment: LineSegment,
    vp_x: float,
    vp_y: float,
) -> float:
    """Return the angular mismatch between a line and the ray toward a VP."""

    mid_x, mid_y = segment_midpoint(segment)
    ray_x = vp_x - mid_x
    ray_y = vp_y - mid_y
    if math.isclose(ray_x, 0.0, abs_tol=1e-9) and math.isclose(ray_y, 0.0, abs_tol=1e-9):
        return math.pi * 0.5
    ray_angle = normalize_line_angle(math.atan2(ray_y, ray_x))
    return smallest_undirected_angle_difference(segment.angle_rad, ray_angle)


def score_vp_candidate(
    vp_x: float,
    vp_y: float,
    lines: Iterable[LineSegment],
    angular_inlier_threshold_rad: float,
) -> tuple[float, int]:
    """Score a vanishing point candidate using angular agreement."""

    weighted_support = 0.0
    total_weight = 0.0
    inliers = 0

    for segment in lines:
        weight = max(segment.length, 1.0)
        residual = angular_residual_to_vp(segment, vp_x, vp_y)
        total_weight += weight
        if residual <= angular_inlier_threshold_rad:
            inliers += 1
            agreement = 1.0 - (residual / max(angular_inlier_threshold_rad, 1e-6))
            weighted_support += weight * max(agreement, 0.0)

    if total_weight == 0.0:
        return (0.0, 0)
    return (float(weighted_support / total_weight), inliers)


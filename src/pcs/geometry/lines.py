"""Line geometry helper functions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from pcs.geometry.types import LineSegment


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to a closed interval."""

    return max(low, min(high, value))


def normalize_line_angle(angle_rad: float) -> float:
    """Normalize a line orientation to the undirected range [0, pi)."""

    normalized = math.fmod(angle_rad, math.pi)
    if normalized < 0.0:
        normalized += math.pi
    return normalized


def line_length(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return Euclidean segment length."""

    return float(math.hypot(x2 - x1, y2 - y1))


def line_angle_rad(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return a canonical undirected orientation in radians."""

    return normalize_line_angle(math.atan2(y2 - y1, x2 - x1))


def build_line_segment(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    detector_name: str,
) -> LineSegment:
    """Construct a line segment with deterministic geometry metadata."""

    length = line_length(x1, y1, x2, y2)
    return LineSegment(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        confidence=float(clamp(confidence, 0.0, 1.0)),
        length=length,
        angle_rad=line_angle_rad(x1, y1, x2, y2),
        detector_name=detector_name,
    )


def filter_short_segments(
    segments: Iterable[LineSegment],
    min_length: float,
) -> list[LineSegment]:
    """Remove degenerate or short segments."""

    return [segment for segment in segments if segment.length >= min_length]


def segment_midpoint(segment: LineSegment) -> tuple[float, float]:
    """Return the segment midpoint."""

    return ((segment.x1 + segment.x2) * 0.5, (segment.y1 + segment.y2) * 0.5)


def smallest_undirected_angle_difference(a: float, b: float) -> float:
    """Return the smallest angle difference for undirected line orientations."""

    diff = abs(normalize_line_angle(a) - normalize_line_angle(b))
    return min(diff, math.pi - diff)


def segment_to_homogeneous(segment: LineSegment) -> np.ndarray:
    """Convert a 2D segment into an infinite homogeneous line."""

    p1 = np.array([segment.x1, segment.y1, 1.0], dtype=np.float64)
    p2 = np.array([segment.x2, segment.y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = np.linalg.norm(line[:2])
    if norm == 0.0:
        return line
    return line / norm


def clip_segment_to_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> tuple[float, float, float, float] | None:
    """Clip a segment against an axis-aligned rectangle using Liang-Barsky."""

    dx = x2 - x1
    dy = y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - x_min, x_max - x1, y1 - y_min, y_max - y1)
    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q, strict=True):
        if pi == 0.0:
            if qi < 0.0:
                return None
            continue
        u = qi / pi
        if pi < 0.0:
            u1 = max(u1, u)
        else:
            u2 = min(u2, u)
        if u1 > u2:
            return None

    return (
        x1 + u1 * dx,
        y1 + u1 * dy,
        x1 + u2 * dx,
        y1 + u2 * dy,
    )


def segment_overlap_ratio_with_rect(
    segment: LineSegment,
    rect: tuple[float, float, float, float],
) -> float:
    """Estimate how much of a segment lies inside a rectangle."""

    clipped = clip_segment_to_rect(
        segment.x1,
        segment.y1,
        segment.x2,
        segment.y2,
        rect[0],
        rect[1],
        rect[2],
        rect[3],
    )
    if clipped is None or segment.length == 0.0:
        return 0.0
    clipped_length = line_length(*clipped)
    return float(clamp(clipped_length / segment.length, 0.0, 1.0))


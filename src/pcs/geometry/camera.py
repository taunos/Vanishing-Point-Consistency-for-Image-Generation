"""Approximate camera and horizon helpers for evaluator-side consensus.

This module intentionally provides coarse evaluator-side projective utilities
rather than full camera calibration. It is the shared home for:

- a basic horizon-line type,
- VP-to-horizon proxy reasoning,
- angular comparisons in projective direction space,
- lightweight horizon fitting used by the consensus evaluator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pcs.geometry.types import PatchGeometricSignature


@dataclass(slots=True, frozen=True)
class HorizonLine:
    """A simple homogeneous 2D line representation."""

    a: float
    b: float
    c: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)


def horizon_line_from_y(y_value: float) -> HorizonLine:
    """Return the homogeneous line for y = constant."""

    return HorizonLine(0.0, 1.0, -float(y_value))


def vp_to_projective_direction(
    reference_point: tuple[float, float],
    vp: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """Convert a VP into a normalized projective direction from a reference."""

    if vp is None:
        return None
    dx = float(vp[0] - reference_point[0])
    dy = float(vp[1] - reference_point[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-8:
        return None
    return (dx / norm, dy / norm)


def projective_angular_distance_deg(
    direction_a: tuple[float, float] | None,
    direction_b: tuple[float, float] | None,
) -> float | None:
    """Return the angular distance between two projective directions."""

    if direction_a is None or direction_b is None:
        return None
    dot = max(-1.0, min(1.0, (direction_a[0] * direction_b[0]) + (direction_a[1] * direction_b[1])))
    return float(math.degrees(math.acos(dot)))


def horizon_y_proxy_from_vp(
    vp: tuple[float, float] | None,
    image_center: tuple[float, float],
    horizontal_ratio_threshold: float = 1.25,
) -> float | None:
    """Infer a coarse horizon y proxy when the VP behaves like a horizontal VP.

    This is intentionally approximate and evaluator-side only. We treat a VP as
    horizon-informative when its displacement from the image center is more
    horizontal than vertical.
    """

    if vp is None:
        return None
    dx = float(vp[0] - image_center[0])
    dy = float(vp[1] - image_center[1])
    if abs(dx) < horizontal_ratio_threshold * abs(dy):
        return None
    return float(vp[1])


def fit_weighted_horizontal_horizon(
    signatures: Iterable["PatchGeometricSignature"],
    image_center: tuple[float, float],
) -> tuple[HorizonLine | None, dict[str, float]]:
    """Fit a coarse horizontal horizon from a set of signatures."""

    weighted_sum = 0.0
    total_weight = 0.0
    count = 0

    for signature in signatures:
        y_proxy = horizon_y_proxy_from_vp(signature.dominant_vp, image_center=image_center)
        if y_proxy is None:
            continue
        weight = max(0.0, signature.support_score * signature.stability_score)
        if weight <= 0.0:
            continue
        weighted_sum += weight * y_proxy
        total_weight += weight
        count += 1

    if total_weight <= 0.0:
        return (None, {"num_horizon_contributors": 0.0})

    mean_y = weighted_sum / total_weight
    return (
        horizon_line_from_y(mean_y),
        {
            "num_horizon_contributors": float(count),
            "mean_horizon_y": float(mean_y),
        },
    )

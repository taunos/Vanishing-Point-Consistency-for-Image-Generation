"""Small reusable helpers for bounded consensus aggregation."""

from __future__ import annotations

from typing import Iterable


def clamp01(value: float) -> float:
    """Clamp a numeric value to the unit interval."""

    return max(0.0, min(1.0, float(value)))


def weighted_average(values: Iterable[tuple[float, float]]) -> float:
    """Compute a normalized weighted average over (value, weight) pairs."""

    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in values:
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def ratio_score(numerator: float, denominator: float) -> float:
    """Safely compute a bounded ratio."""

    if denominator <= 0.0:
        return 0.0
    return clamp01(float(numerator) / float(denominator))


def bounded_inverse_error(error: float, soft_limit: float) -> float:
    """Map a non-negative error into a bounded goodness score."""

    if soft_limit <= 0.0:
        return 0.0
    return clamp01(1.0 - (float(error) / float(soft_limit)))


"""Analytic applicability gate for geometry-rich scenes."""

from __future__ import annotations

import math

import numpy as np

from pcs.geometry.types import ApplicabilityResult, LineSet, RegionalHypothesis
from pcs.utils.config import ApplicabilityConfig


def _bounded_ratio(value: float, minimum: float, target: float) -> float:
    if target <= minimum:
        return 1.0 if value >= target else 0.0
    clipped = (value - minimum) / (target - minimum)
    return max(0.0, min(1.0, float(clipped)))


def compute_orientation_entropy(lines: list, bins: int) -> float:
    """Compute normalized entropy of undirected line orientations."""

    if not lines:
        return 0.0
    angles = np.asarray([segment.angle_rad for segment in lines], dtype=np.float64)
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, math.pi), density=False)
    probs = hist.astype(np.float64) / max(hist.sum(), 1.0)
    non_zero = probs[probs > 0.0]
    entropy = -np.sum(non_zero * np.log(non_zero))
    max_entropy = math.log(bins) if bins > 1 else 1.0
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def evaluate_applicability(
    line_set: LineSet,
    hypotheses: list[RegionalHypothesis],
    config: ApplicabilityConfig,
) -> ApplicabilityResult:
    """Estimate whether an image has enough perspective evidence for PCS."""

    lines = line_set.segments
    num_lines = len(lines)
    mean_length = float(np.mean([line.length for line in lines])) if lines else 0.0
    long_line_ratio = (
        float(np.mean([line.length >= config.long_line_length for line in lines]))
        if lines
        else 0.0
    )
    orientation_entropy = compute_orientation_entropy(lines, bins=config.orientation_histogram_bins)
    viable_patches = sum(1 for hypothesis in hypotheses if hypothesis.metadata.get("viable", False))
    supported_patches = sum(
        1
        for hypothesis in hypotheses
        if hypothesis.support_score >= config.supported_patch_min_support
    )
    total_patches = len(hypotheses)
    supported_patch_ratio = (
        float(supported_patches / total_patches) if total_patches > 0 else 0.0
    )

    confidence_components = [
        _bounded_ratio(num_lines, config.min_num_lines, config.target_num_lines),
        _bounded_ratio(mean_length, config.min_mean_length, config.target_mean_length),
        _bounded_ratio(
            long_line_ratio,
            config.min_long_line_ratio,
            config.target_long_line_ratio,
        ),
        _bounded_ratio(
            orientation_entropy,
            config.min_orientation_entropy,
            config.target_orientation_entropy,
        ),
        _bounded_ratio(
            viable_patches,
            config.min_viable_patches,
            config.target_viable_patches,
        ),
        _bounded_ratio(
            supported_patch_ratio,
            config.min_supported_patch_ratio,
            config.target_supported_patch_ratio,
        ),
    ]
    confidence = float(np.mean(confidence_components))
    passed = (
        confidence >= config.confidence_threshold
        and num_lines >= config.min_num_lines
        and supported_patch_ratio >= config.min_supported_patch_ratio
    )

    return ApplicabilityResult(
        confidence=max(0.0, min(1.0, confidence)),
        passed=passed,
        features={
            "num_valid_lines": float(num_lines),
            "mean_line_length": mean_length,
            "long_line_ratio": long_line_ratio,
            "orientation_entropy": orientation_entropy,
            "num_viable_patches": float(viable_patches),
            "supported_patch_ratio": supported_patch_ratio,
        },
    )


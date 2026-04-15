"""Baseline analytic PCS scoring for Milestone 1.

This module is limited to final score composition. It intentionally does not
implement cross-patch compatibility, graph reasoning, or global consensus.
"""

from __future__ import annotations

import numpy as np

from pcs.geometry.types import ApplicabilityResult, LineSet, PCSBaselineResult, RegionalHypothesis
from pcs.utils.config import ScoringConfig


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_local_score(line_set: LineSet, config: ScoringConfig) -> float:
    """Aggregate conservative line-level support quality."""

    lines = line_set.segments
    if not lines:
        return 0.0
    num_lines_ratio = _clamp01(len(lines) / max(config.expected_num_lines, 1))
    mean_length_ratio = _clamp01(
        float(np.mean([segment.length for segment in lines])) / max(config.expected_mean_length, 1e-6)
    )
    mean_confidence = _clamp01(float(np.mean([segment.confidence for segment in lines])))
    return _clamp01(
        (config.local_line_count_weight * num_lines_ratio)
        + (config.local_mean_length_weight * mean_length_ratio)
        + (config.local_confidence_weight * mean_confidence)
    )


def compute_regional_score(hypotheses: list[RegionalHypothesis], config: ScoringConfig) -> float:
    """Aggregate conservative regional hypothesis quality."""

    if not hypotheses:
        return 0.0
    viable = [hypothesis for hypothesis in hypotheses if hypothesis.vp_candidates]
    coverage_ratio = _clamp01(len(viable) / len(hypotheses))
    if not viable:
        support_mean = 0.0
        stability_mean = 0.0
    else:
        support_mean = float(np.mean([hypothesis.support_score for hypothesis in viable]))
        stability_mean = float(np.mean([hypothesis.stability_score for hypothesis in viable]))

    return _clamp01(
        (config.regional_support_weight * _clamp01(support_mean))
        + (config.regional_stability_weight * _clamp01(stability_mean))
        + (config.regional_coverage_weight * coverage_ratio)
    )


def compute_baseline_pcs(
    line_set: LineSet,
    hypotheses: list[RegionalHypothesis],
    applicability: ApplicabilityResult,
    config: ScoringConfig,
) -> PCSBaselineResult:
    """Compute the Milestone 1 baseline PCS score."""

    local_score = compute_local_score(line_set, config)
    regional_score = compute_regional_score(hypotheses, config)
    structural_score = _clamp01(
        (config.local_weight * local_score) + (config.regional_weight * regional_score)
    )
    pcs_score = _clamp01(applicability.confidence * structural_score)

    return PCSBaselineResult(
        pcs_score=pcs_score,
        applicability=applicability,
        local_score=local_score,
        regional_score=regional_score,
        num_lines=len(line_set.segments),
        num_patches=len(hypotheses),
        metadata={
            "structural_score": structural_score,
            "num_viable_patches": sum(1 for hypothesis in hypotheses if hypothesis.vp_candidates),
            "detector_name": line_set.metadata.get("detector_name", "unknown"),
        },
    )

import math

from pcs.geometry.lines import build_line_segment
from pcs.geometry.types import (
    ApplicabilityResult,
    LineSet,
    Patch,
    RegionalHypothesis,
    VanishingPointCandidate,
)
from pcs.scoring.baseline import compute_baseline_pcs
from pcs.utils.config import ScoringConfig


def _line(length: float, confidence: float, y_offset: float) -> object:
    return build_line_segment(
        0.0,
        y_offset,
        length,
        y_offset,
        confidence=confidence,
        detector_name="synthetic",
    )


def _hypothesis(index: int, support: float, stability: float) -> RegionalHypothesis:
    patch = Patch(
        patch_id=f"patch_{index}",
        x0=0,
        y0=0,
        x1=128,
        y1=128,
        scale_level=0,
        overlap_tag="ov20",
    )
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=[VanishingPointCandidate(x=200.0, y=50.0, score=support, num_inliers=5)],
        support_score=support,
        stability_score=stability,
        num_lines=5,
        metadata={"viable": True},
    )


def test_baseline_scoring_is_bounded_and_monotonic() -> None:
    config = ScoringConfig()
    line_set = LineSet(
        segments=[
            _line(120.0, 0.95, 0.0),
            _line(115.0, 0.92, 10.0),
            _line(110.0, 0.97, 20.0),
            _line(105.0, 0.9, 30.0),
        ],
        image_width=256,
        image_height=256,
    )
    low_result = compute_baseline_pcs(
        line_set=line_set,
        hypotheses=[_hypothesis(0, 0.2, 0.15)],
        applicability=ApplicabilityResult(confidence=0.2, passed=False, features={}),
        config=config,
    )
    high_result = compute_baseline_pcs(
        line_set=line_set,
        hypotheses=[_hypothesis(0, 0.8, 0.7), _hypothesis(1, 0.75, 0.65)],
        applicability=ApplicabilityResult(confidence=0.9, passed=True, features={}),
        config=config,
    )

    for result in (low_result, high_result):
        assert 0.0 <= result.local_score <= 1.0
        assert 0.0 <= result.regional_score <= 1.0
        assert 0.0 <= result.pcs_score <= 1.0

    assert high_result.regional_score > low_result.regional_score
    assert high_result.pcs_score > low_result.pcs_score


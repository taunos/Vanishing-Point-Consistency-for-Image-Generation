import math

from pcs.applicability.gate import evaluate_applicability
from pcs.geometry.lines import build_line_segment
from pcs.geometry.types import LineSet, Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.utils.config import ApplicabilityConfig


def _make_line(length: float, angle_rad: float, y_offset: float) -> object:
    x1 = 10.0
    y1 = 10.0 + y_offset
    x2 = x1 + (length * math.cos(angle_rad))
    y2 = y1 + (length * math.sin(angle_rad))
    return build_line_segment(x1, y1, x2, y2, confidence=0.95, detector_name="synthetic")


def _make_hypothesis(index: int, support: float, stability: float, viable: bool) -> RegionalHypothesis:
    patch = Patch(
        patch_id=f"patch_{index}",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        scale_level=0,
        overlap_tag="ov20",
    )
    candidates = []
    if viable:
        candidates = [VanishingPointCandidate(x=200.0, y=50.0, score=support, num_inliers=4)]
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=candidates,
        support_score=support if viable else 0.0,
        stability_score=stability if viable else 0.0,
        num_lines=6,
        metadata={"viable": viable},
    )


def test_applicability_confidence_increases_with_stronger_evidence() -> None:
    config = ApplicabilityConfig()
    weak_lines = LineSet(
        segments=[_make_line(25.0, 0.0, 0.0), _make_line(28.0, 0.0, 10.0)],
        image_width=256,
        image_height=256,
    )
    strong_lines = LineSet(
        segments=[
            _make_line(120.0, 0.0, 0.0),
            _make_line(110.0, math.pi / 6.0, 12.0),
            _make_line(115.0, math.pi / 3.0, 25.0),
            _make_line(105.0, math.pi / 2.0, 38.0),
            _make_line(130.0, math.pi * 0.75, 50.0),
            _make_line(118.0, math.pi * 0.9, 62.0),
            _make_line(122.0, 0.1, 74.0),
            _make_line(109.0, math.pi / 5.0, 86.0),
            _make_line(112.0, math.pi / 2.8, 98.0),
            _make_line(126.0, math.pi / 1.9, 110.0),
            _make_line(117.0, math.pi * 0.72, 122.0),
            _make_line(121.0, math.pi * 0.94, 134.0),
        ],
        image_width=256,
        image_height=256,
    )

    weak = evaluate_applicability(
        line_set=weak_lines,
        hypotheses=[_make_hypothesis(0, 0.0, 0.0, viable=False) for _ in range(4)],
        config=config,
    )
    strong = evaluate_applicability(
        line_set=strong_lines,
        hypotheses=[
            _make_hypothesis(0, 0.7, 0.6, viable=True),
            _make_hypothesis(1, 0.65, 0.55, viable=True),
            _make_hypothesis(2, 0.6, 0.5, viable=True),
            _make_hypothesis(3, 0.0, 0.0, viable=False),
        ],
        config=config,
    )

    assert 0.0 <= weak.confidence <= 1.0
    assert 0.0 <= strong.confidence <= 1.0
    assert strong.confidence > weak.confidence
    assert strong.passed is True

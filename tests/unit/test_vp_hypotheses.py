import math

from pcs.geometry.lines import build_line_segment
from pcs.geometry.types import LineSet, Patch
from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.utils.config import RegionalConfig


def _line_through_vp(vp_x: float, vp_y: float, start_x: float, start_y: float) -> object:
    dx = vp_x - start_x
    dy = vp_y - start_y
    scale = 0.35
    end_x = start_x + (dx * scale)
    end_y = start_y + (dy * scale)
    return build_line_segment(
        start_x,
        start_y,
        end_x,
        end_y,
        confidence=1.0,
        detector_name="synthetic",
    )


def test_estimate_regional_hypotheses_finds_converging_vp() -> None:
    vp_x, vp_y = 220.0, 90.0
    lines = [
        _line_through_vp(vp_x, vp_y, 20.0, 20.0),
        _line_through_vp(vp_x, vp_y, 20.0, 120.0),
        _line_through_vp(vp_x, vp_y, 60.0, 140.0),
        _line_through_vp(vp_x, vp_y, 80.0, 30.0),
    ]
    line_set = LineSet(segments=lines, image_width=256, image_height=256)
    patch = Patch(
        patch_id="full",
        x0=0,
        y0=0,
        x1=256,
        y1=256,
        scale_level=0,
        overlap_tag="ov20",
    )
    config = RegionalConfig(
        min_lines_per_patch=3,
        top_k_candidates=2,
        patch_line_overlap_ratio=0.2,
        min_intersection_line_angle_deg=5.0,
        angular_inlier_threshold_deg=10.0,
        intersection_dedup_radius_ratio=0.03,
        min_candidate_inliers=2,
        max_vp_distance_factor=8.0,
    )

    hypotheses = estimate_regional_hypotheses(line_set, [patch], config)
    hypothesis = hypotheses[0]

    assert hypothesis.vp_candidates
    best = hypothesis.vp_candidates[0]
    assert hypothesis.support_score > 0.45
    assert hypothesis.stability_score > 0.2
    assert hypothesis.metadata["bootstrap_stability"] > 0.2
    assert len(hypothesis.metadata["orientation_histogram"]) == config.orientation_histogram_bins
    assert abs(best.x - vp_x) < 25.0
    assert abs(best.y - vp_y) < 25.0


def test_estimate_regional_hypotheses_rejects_parallel_lines() -> None:
    lines = [
        build_line_segment(0.0, 10.0, 100.0, 10.0, confidence=1.0, detector_name="synthetic"),
        build_line_segment(0.0, 20.0, 100.0, 20.0, confidence=1.0, detector_name="synthetic"),
        build_line_segment(0.0, 30.0, 100.0, 30.0, confidence=1.0, detector_name="synthetic"),
        build_line_segment(0.0, 40.0, 100.0, 40.0, confidence=1.0, detector_name="synthetic"),
    ]
    line_set = LineSet(segments=lines, image_width=128, image_height=128)
    patch = Patch(
        patch_id="full",
        x0=0,
        y0=0,
        x1=128,
        y1=128,
        scale_level=0,
        overlap_tag="ov20",
    )

    hypotheses = estimate_regional_hypotheses(line_set, [patch], RegionalConfig())
    hypothesis = hypotheses[0]

    assert hypothesis.vp_candidates == []
    assert math.isclose(hypothesis.support_score, 0.0)
    assert len(hypothesis.metadata["orientation_histogram"]) == RegionalConfig().orientation_histogram_bins

"""Unit tests for v3 sub-component percentile global scoring."""

from pcs.geometry.lines import build_line_segment
from pcs.geometry.types import (
    ApplicabilityResult,
    LineSet,
    Patch,
    RegionalHypothesis,
    RegionalHypothesisMatch,
    VanishingPointCandidate,
)
from pcs.scoring.local_to_global import _compute_v3_global_score, compute_local_to_global_pcs
from pcs.utils.config import ConsensusConfig, LocalToGlobalScoringConfig, ScoringConfig


def _make_match(
    pid_a: str,
    pid_b: str,
    compat: float,
    horizon_score: float = 0.8,
    direction_score: float = 0.7,
) -> RegionalHypothesisMatch:
    return RegionalHypothesisMatch(
        patch_id_a=pid_a,
        patch_id_b=pid_b,
        vp_idx_a=0,
        vp_idx_b=0,
        vp_position_a=(100.0, 50.0),
        vp_position_b=(102.0, 51.0),
        compatibility_score=compat,
        geometric_error=0.1,
        metadata={
            "horizon_score": horizon_score,
            "direction_score": direction_score,
        },
    )


# --- Test 1: Empty matches ---
def test_empty_matches_return_zero() -> None:
    score, details = _compute_v3_global_score([])
    assert score == 0.0
    assert details["n_edges"] == 0


# --- Test 2: Consistent matches -> high score ---
def test_consistent_matches_high_score() -> None:
    matches = [
        _make_match("p1", "p2", 0.9, horizon_score=0.9, direction_score=0.85),
        _make_match("p2", "p3", 0.85, horizon_score=0.88, direction_score=0.82),
        _make_match("p3", "p4", 0.87, horizon_score=0.92, direction_score=0.80),
        _make_match("p1", "p3", 0.88, horizon_score=0.91, direction_score=0.83),
    ]
    score, details = _compute_v3_global_score(matches)
    assert score > 0.75
    assert details["hor_p05"] > 0.85
    assert details["dir_p05"] > 0.75


# --- Test 3: Low horizon scores -> substantially lower score ---
def test_low_horizon_reduces_score() -> None:
    good_matches = [
        _make_match("p1", "p2", 0.85, horizon_score=0.9, direction_score=0.8),
        _make_match("p2", "p3", 0.82, horizon_score=0.88, direction_score=0.78),
        _make_match("p3", "p4", 0.80, horizon_score=0.85, direction_score=0.75),
    ]
    bad_horizon_matches = [
        _make_match("p1", "p2", 0.85, horizon_score=0.2, direction_score=0.8),
        _make_match("p2", "p3", 0.82, horizon_score=0.15, direction_score=0.78),
        _make_match("p3", "p4", 0.80, horizon_score=0.1, direction_score=0.75),
    ]
    score_good, _ = _compute_v3_global_score(good_matches)
    score_bad, _ = _compute_v3_global_score(bad_horizon_matches)
    assert score_bad < score_good
    # Horizon has 50% weight, so difference should be substantial
    assert score_good - score_bad > 0.2


# --- Test 4: Low direction scores -> lower score ---
def test_low_direction_reduces_score() -> None:
    good_matches = [
        _make_match("p1", "p2", 0.85, horizon_score=0.9, direction_score=0.85),
        _make_match("p2", "p3", 0.82, horizon_score=0.88, direction_score=0.82),
        _make_match("p3", "p4", 0.80, horizon_score=0.85, direction_score=0.80),
    ]
    bad_dir_matches = [
        _make_match("p1", "p2", 0.85, horizon_score=0.9, direction_score=0.15),
        _make_match("p2", "p3", 0.82, horizon_score=0.88, direction_score=0.10),
        _make_match("p3", "p4", 0.80, horizon_score=0.85, direction_score=0.12),
    ]
    score_good, _ = _compute_v3_global_score(good_matches)
    score_bad, _ = _compute_v3_global_score(bad_dir_matches)
    assert score_bad < score_good


# --- Test 5: p10 is sensitive to outliers (corruption signal) ---
def test_p10_sensitive_to_worst_edges() -> None:
    """If a few edges have low horizon scores (corruption), p10 captures that."""
    # Mostly good, with 2 bad edges out of 10
    matches = [
        _make_match(f"p{i}", f"p{i+1}", 0.85, horizon_score=0.9, direction_score=0.8)
        for i in range(1, 9)
    ]
    # Add 2 bad edges
    matches.append(_make_match("p1", "p5", 0.3, horizon_score=0.1, direction_score=0.2))
    matches.append(_make_match("p2", "p6", 0.35, horizon_score=0.15, direction_score=0.25))

    _, details = _compute_v3_global_score(matches)
    # p10 of 10 values = approximately the 1st value when sorted
    # 2 out of 10 are low, so p10 should be pulled down significantly
    assert details["hor_p05"] < 0.5


# --- Test 6: Integration — v3 end-to-end ---
def test_v3_coherent_beats_contradictory() -> None:
    """With v3, coherent hypotheses score higher than contradictory ones."""

    def _line(y_offset: float) -> object:
        return build_line_segment(
            0.0, y_offset, 120.0, y_offset + 5.0,
            confidence=0.95, detector_name="synthetic",
        )

    def _hyp(pid: str, x0: int, vp_x: float) -> RegionalHypothesis:
        patch = Patch(
            patch_id=pid, x0=x0, y0=0, x1=x0 + 50, y1=60,
            scale_level=0, overlap_tag="ov20",
        )
        return RegionalHypothesis(
            patch=patch,
            vp_candidates=[VanishingPointCandidate(x=vp_x, y=90.0, score=0.8, num_inliers=5)],
            support_score=0.8, stability_score=0.75, num_lines=6,
            metadata={"viable": True, "orientation_histogram": [0.6, 0.3, 0.1, 0.0]},
        )

    ls = LineSet(
        segments=[_line(0.0), _line(10.0), _line(20.0), _line(30.0)],
        image_width=256, image_height=256,
    )
    coherent_hyps = [_hyp("p1", 0, 220.0), _hyp("p2", 60, 225.0), _hyp("p3", 120, 230.0)]
    contradictory_hyps = [_hyp("p1", 0, 220.0), _hyp("p2", 60, 225.0), _hyp("p3", 120, 20.0)]
    app = ApplicabilityResult(confidence=0.9, passed=True, features={})

    v3_config = LocalToGlobalScoringConfig(
        local_weight=0.15, regional_weight=0.25, global_weight=0.60,
        coherence_weight=0.0, version="v3",
    )
    consensus_cfg = ConsensusConfig(graph_mode="all_pairs", target_supported_patches=4)

    coherent_result, _ = compute_local_to_global_pcs(
        line_set=ls, hypotheses=coherent_hyps, applicability=app,
        baseline_scoring_config=ScoringConfig(), consensus_config=consensus_cfg,
        scoring_config=v3_config,
    )
    contradictory_result, _ = compute_local_to_global_pcs(
        line_set=ls, hypotheses=contradictory_hyps, applicability=app,
        baseline_scoring_config=ScoringConfig(), consensus_config=consensus_cfg,
        scoring_config=v3_config,
    )

    assert coherent_result.pcs_score > contradictory_result.pcs_score
    assert coherent_result.metadata["scoring_version"] == "v3"
    assert "v3_global_details" in coherent_result.metadata
    assert "v2_global_details" not in coherent_result.metadata


# --- Test 7: v3 config normalization ---
def test_v3_config_normalization() -> None:
    cfg = LocalToGlobalScoringConfig(
        local_weight=0.15, regional_weight=0.25, global_weight=0.60,
        coherence_weight=0.0, version="v3",
    )
    assert cfg.coherence_weight == 0.0
    assert abs(cfg.local_weight + cfg.regional_weight + cfg.global_weight - 1.0) < 1e-9

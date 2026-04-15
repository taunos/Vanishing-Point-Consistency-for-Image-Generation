from pcs.geometry.lines import build_line_segment
from pcs.geometry.types import ApplicabilityResult, LineSet, Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.scoring.local_to_global import compute_local_to_global_pcs
from pcs.utils.config import ConsensusConfig, LocalToGlobalScoringConfig, ScoringConfig


def _line(y_offset: float) -> object:
    return build_line_segment(
        0.0,
        y_offset,
        120.0,
        y_offset + 5.0,
        confidence=0.95,
        detector_name="synthetic",
    )


def _hypothesis(patch_id: str, x0: int, vp_x: float) -> RegionalHypothesis:
    patch = Patch(
        patch_id=patch_id,
        x0=x0,
        y0=0,
        x1=x0 + 50,
        y1=60,
        scale_level=0,
        overlap_tag="ov20",
    )
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=[VanishingPointCandidate(x=vp_x, y=90.0, score=0.8, num_inliers=5)],
        support_score=0.8,
        stability_score=0.75,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.6, 0.3, 0.1, 0.0]},
    )


def test_local_to_global_scoring_is_monotonic_in_consensus_and_applicability() -> None:
    line_set = LineSet(
        segments=[_line(0.0), _line(10.0), _line(20.0), _line(30.0)],
        image_width=256,
        image_height=256,
    )
    coherent_hypotheses = [
        _hypothesis("p1", 0, 220.0),
        _hypothesis("p2", 60, 225.0),
        _hypothesis("p3", 120, 230.0),
    ]
    contradictory_hypotheses = [
        _hypothesis("p1", 0, 220.0),
        _hypothesis("p2", 60, 225.0),
        _hypothesis("p3", 120, 20.0),
    ]
    high_applicability = ApplicabilityResult(confidence=0.9, passed=True, features={})
    low_applicability = ApplicabilityResult(confidence=0.15, passed=False, features={})

    coherent_result, _ = compute_local_to_global_pcs(
        line_set=line_set,
        hypotheses=coherent_hypotheses,
        applicability=high_applicability,
        baseline_scoring_config=ScoringConfig(),
        consensus_config=ConsensusConfig(graph_mode="all_pairs", target_supported_patches=4),
        scoring_config=LocalToGlobalScoringConfig(),
    )
    contradictory_result, _ = compute_local_to_global_pcs(
        line_set=line_set,
        hypotheses=contradictory_hypotheses,
        applicability=high_applicability,
        baseline_scoring_config=ScoringConfig(),
        consensus_config=ConsensusConfig(graph_mode="all_pairs", target_supported_patches=4),
        scoring_config=LocalToGlobalScoringConfig(),
    )
    damped_result, _ = compute_local_to_global_pcs(
        line_set=line_set,
        hypotheses=coherent_hypotheses,
        applicability=low_applicability,
        baseline_scoring_config=ScoringConfig(),
        consensus_config=ConsensusConfig(graph_mode="all_pairs", target_supported_patches=4),
        scoring_config=LocalToGlobalScoringConfig(),
    )

    assert coherent_result.global_consensus_score > contradictory_result.global_consensus_score
    assert coherent_result.incompatibility_penalty < contradictory_result.incompatibility_penalty
    assert coherent_result.pcs_score > contradictory_result.pcs_score
    assert damped_result.pcs_score < coherent_result.pcs_score
    assert "patch_inconsistency_map" in coherent_result.metadata

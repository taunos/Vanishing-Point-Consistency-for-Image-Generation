from pcs.consensus.compatibility import score_region_graph_matches
from pcs.consensus.global_fit import fit_global_camera_consensus
from pcs.consensus.graph import build_region_graph
from pcs.consensus.inconsistency import compute_incompatibility_penalty
from pcs.consensus.signatures import build_patch_signatures
from pcs.geometry.types import Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.utils.config import ConsensusConfig


def _hypothesis(
    patch_id: str,
    x0: int,
    vp_x: float,
    support: float = 0.8,
) -> RegionalHypothesis:
    patch = Patch(
        patch_id=patch_id,
        x0=x0,
        y0=0,
        x1=x0 + 50,
        y1=50,
        scale_level=0,
        overlap_tag="ov20",
    )
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=[VanishingPointCandidate(x=vp_x, y=80.0, score=support, num_inliers=5)],
        support_score=support,
        stability_score=0.75,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.6, 0.3, 0.1, 0.0]},
    )


def _penalty(hypotheses: list[RegionalHypothesis]) -> float:
    config = ConsensusConfig(graph_mode="all_pairs", target_supported_patches=4)
    graph = build_region_graph(hypotheses, config)
    signatures = build_patch_signatures(hypotheses, 256, 256, config)
    matches = score_region_graph_matches(graph, hypotheses, signatures, 256, 256, config)
    fit = fit_global_camera_consensus(graph, signatures, matches, 256, 256, config)
    penalty, _ = compute_incompatibility_penalty(graph, matches, fit, config)
    return penalty


def test_inconsistency_penalty_distinguishes_low_and_high_conflict() -> None:
    low_evidence_penalty = _penalty([_hypothesis("p1", 0, 220.0, support=0.3)])
    one_outlier_penalty = _penalty(
        [
            _hypothesis("p1", 0, 220.0),
            _hypothesis("p2", 60, 225.0),
            _hypothesis("p3", 120, 230.0),
            _hypothesis("p4", 180, 20.0),
        ]
    )
    strong_conflict_penalty = _penalty(
        [
            _hypothesis("p1", 0, 220.0),
            _hypothesis("p2", 60, 225.0),
            _hypothesis("p3", 120, 20.0),
            _hypothesis("p4", 180, 25.0),
        ]
    )

    assert low_evidence_penalty < 0.1
    assert one_outlier_penalty > low_evidence_penalty
    assert strong_conflict_penalty > one_outlier_penalty


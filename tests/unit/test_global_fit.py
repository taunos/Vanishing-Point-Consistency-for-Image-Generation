from pcs.consensus.compatibility import score_region_graph_matches
from pcs.consensus.global_fit import fit_global_camera_consensus
from pcs.consensus.graph import build_region_graph
from pcs.consensus.signatures import build_patch_signatures
from pcs.geometry.types import Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.utils.config import ConsensusConfig


def _regional_hypothesis(
    patch_id: str,
    bounds: tuple[int, int, int, int],
    vp: tuple[float, float],
) -> RegionalHypothesis:
    patch = Patch(
        patch_id=patch_id,
        x0=bounds[0],
        y0=bounds[1],
        x1=bounds[2],
        y1=bounds[3],
        scale_level=0,
        overlap_tag="ov20",
    )
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=[VanishingPointCandidate(x=vp[0], y=vp[1], score=0.8, num_inliers=5)],
        support_score=0.8,
        stability_score=0.7,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.7, 0.2, 0.1, 0.0]},
    )


def _run_global_fit(hypotheses: list[RegionalHypothesis]) -> object:
    config = ConsensusConfig(graph_mode="all_pairs", min_consensus_size=2, target_supported_patches=4)
    graph = build_region_graph(hypotheses, config)
    signatures = build_patch_signatures(hypotheses, 256, 256, config)
    matches = score_region_graph_matches(graph, hypotheses, signatures, 256, 256, config)
    return fit_global_camera_consensus(graph, signatures, matches, 256, 256, config)


def test_global_fit_prefers_coherent_multi_patch_explanations() -> None:
    coherent = _run_global_fit(
        [
            _regional_hypothesis("p1", (0, 0, 60, 60), (220.0, 100.0)),
            _regional_hypothesis("p2", (64, 0, 124, 60), (225.0, 102.0)),
            _regional_hypothesis("p3", (128, 0, 188, 60), (230.0, 98.0)),
        ]
    )
    conflicting = _run_global_fit(
        [
            _regional_hypothesis("p1", (0, 0, 60, 60), (220.0, 100.0)),
            _regional_hypothesis("p2", (64, 0, 124, 60), (225.0, 102.0)),
            _regional_hypothesis("p3", (128, 0, 188, 60), (10.0, 150.0)),
        ]
    )
    single = _run_global_fit([_regional_hypothesis("p1", (0, 0, 60, 60), (220.0, 100.0))])

    assert coherent.score > conflicting.score
    assert coherent.success is True
    assert single.success is False
    assert single.score < coherent.score


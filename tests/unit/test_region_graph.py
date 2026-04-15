from pcs.consensus.graph import build_region_graph
from pcs.geometry.types import Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.utils.config import ConsensusConfig


def _hypothesis(
    patch_id: str,
    bounds: tuple[int, int, int, int],
    support: float = 0.7,
    stability: float = 0.6,
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
        vp_candidates=[VanishingPointCandidate(x=200.0, y=50.0, score=support, num_inliers=5)],
        support_score=support,
        stability_score=stability,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.5, 0.5]},
    )


def test_region_graph_modes_are_deterministic() -> None:
    hypotheses = [
        _hypothesis("p1", (0, 0, 50, 50)),
        _hypothesis("p2", (40, 0, 90, 50)),
        _hypothesis("p3", (100, 0, 150, 50)),
    ]

    all_pairs_graph = build_region_graph(
        hypotheses,
        ConsensusConfig(graph_mode="all_pairs"),
    )
    overlap_graph = build_region_graph(
        hypotheses,
        ConsensusConfig(graph_mode="overlap"),
    )
    neighbor_graph = build_region_graph(
        hypotheses,
        ConsensusConfig(graph_mode="spatial_neighbors", neighbor_margin_ratio=0.2),
    )

    assert [node.patch_id for node in all_pairs_graph.nodes] == ["p1", "p2", "p3"]
    assert [(edge.source_patch_id, edge.target_patch_id) for edge in all_pairs_graph.edges] == [
        ("p1", "p2"),
        ("p1", "p3"),
        ("p2", "p3"),
    ]
    assert [(edge.source_patch_id, edge.target_patch_id) for edge in overlap_graph.edges] == [
        ("p1", "p2"),
    ]
    assert [(edge.source_patch_id, edge.target_patch_id) for edge in neighbor_graph.edges] == [
        ("p1", "p2"),
        ("p2", "p3"),
    ]


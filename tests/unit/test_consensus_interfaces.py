from pcs.consensus.graph import RegionGraph, RegionGraphEdge, RegionGraphNode
from pcs.geometry.types import GlobalCameraFitResult


def test_consensus_interfaces_are_importable() -> None:
    node = RegionGraphNode(
        node_id="node_0",
        patch_id="patch_0",
        weight=0.8,
    )
    edge = RegionGraphEdge(
        source_patch_id="patch_0",
        target_patch_id="patch_1",
        overlap_ratio=0.2,
        center_distance=10.0,
    )
    graph = RegionGraph(nodes=[node], edges=[edge])
    result = GlobalCameraFitResult(
        success=False,
        score=0.0,
        num_supported_patches=0,
        num_consistent_matches=0,
        mean_error=1.0,
        fitted_horizon=None,
    )

    assert graph.nodes[0].patch_id == "patch_0"
    assert graph.edges[0].target_patch_id == "patch_1"
    assert result.num_supported_patches == 0

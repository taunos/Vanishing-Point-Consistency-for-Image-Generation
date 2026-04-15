"""Cross-patch local-to-global consensus modules."""

from pcs.consensus.compatibility import (
    match_regional_hypotheses,
    orientation_histogram_similarity,
    score_region_graph_matches,
    score_signature_pair,
)
from pcs.consensus.global_fit import fit_global_camera_consensus
from pcs.consensus.graph import (
    RegionGraph,
    RegionGraphEdge,
    RegionGraphNode,
    build_region_graph,
    patch_center_distance,
    patch_overlap_ratio,
    patches_are_spatial_neighbors,
    patches_overlap,
)
from pcs.consensus.inconsistency import compute_incompatibility_penalty
from pcs.consensus.signatures import build_patch_signature, build_patch_signatures

__all__ = [
    "RegionGraph",
    "RegionGraphEdge",
    "RegionGraphNode",
    "build_patch_signature",
    "build_patch_signatures",
    "build_region_graph",
    "compute_incompatibility_penalty",
    "fit_global_camera_consensus",
    "match_regional_hypotheses",
    "orientation_histogram_similarity",
    "patch_center_distance",
    "patch_overlap_ratio",
    "patches_are_spatial_neighbors",
    "patches_overlap",
    "score_region_graph_matches",
    "score_signature_pair",
]

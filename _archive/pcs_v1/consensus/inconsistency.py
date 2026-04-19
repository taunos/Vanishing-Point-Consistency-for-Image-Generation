"""Incompatibility estimation for local-to-global evaluation."""

from __future__ import annotations

from pcs.consensus.aggregation import clamp01
from pcs.consensus.graph import RegionGraph
from pcs.geometry.types import GlobalCameraFitResult, RegionalHypothesisMatch
from pcs.utils.config import ConsensusConfig


def compute_incompatibility_penalty(
    graph: RegionGraph,
    matches: list[RegionalHypothesisMatch],
    global_fit: GlobalCameraFitResult,
    config: ConsensusConfig,
) -> tuple[float, dict[str, float | int]]:
    """Estimate how much strong regional evidence remains contradictory."""

    if not graph.nodes or not matches:
        return (
            0.0,
            {
                "compared_edges": 0,
                "matched_edges": 0,
                "inconsistent_edges": 0,
            },
        )

    node_weight_by_patch_id = {node.patch_id: node.weight for node in graph.nodes}
    consensus_patch_ids = set(global_fit.metadata.get("supported_patch_ids", []))
    total_edge_evidence = 0.0
    inconsistent_edge_evidence = 0.0
    inconsistent_edges = 0
    patch_evidence = {node.patch_id: 0.0 for node in graph.nodes}
    patch_inconsistent_evidence = {node.patch_id: 0.0 for node in graph.nodes}

    for match in matches:
        edge_weight = 0.5 * (
            node_weight_by_patch_id.get(match.patch_id_a, 0.0)
            + node_weight_by_patch_id.get(match.patch_id_b, 0.0)
        )
        total_edge_evidence += edge_weight
        patch_evidence[match.patch_id_a] += edge_weight
        patch_evidence[match.patch_id_b] += edge_weight
        in_consensus = (
            match.patch_id_a in consensus_patch_ids and match.patch_id_b in consensus_patch_ids
        )
        if in_consensus and match.compatibility_score >= config.compatibility_threshold:
            continue
        contradictory_weight = edge_weight * (1.0 - match.compatibility_score)
        inconsistent_edge_evidence += contradictory_weight
        patch_inconsistent_evidence[match.patch_id_a] += contradictory_weight
        patch_inconsistent_evidence[match.patch_id_b] += contradictory_weight
        if contradictory_weight > 0.0:
            inconsistent_edges += 1

    evidence_factor = clamp01(
        sum(node.weight for node in graph.nodes) / max(config.target_supported_patches, 1)
    )
    if total_edge_evidence <= 0.0:
        penalty = 0.0
    else:
        penalty = clamp01((inconsistent_edge_evidence / total_edge_evidence) * evidence_factor)

    patch_inconsistency_map = {
        patch_id: clamp01(
            (
                patch_inconsistent_evidence[patch_id] / patch_evidence[patch_id]
                if patch_evidence[patch_id] > 0.0
                else 0.0
            )
            * evidence_factor
        )
        for patch_id in sorted(patch_evidence)
    }

    return (
        penalty,
        {
            "compared_edges": len(graph.edges),
            "matched_edges": len(matches),
            "inconsistent_edges": inconsistent_edges,
            "total_edge_evidence": float(total_edge_evidence),
            "inconsistent_edge_evidence": float(inconsistent_edge_evidence),
            "evidence_factor": float(evidence_factor),
            "patch_inconsistency_map": patch_inconsistency_map,
            "consensus_patch_ids": sorted(consensus_patch_ids),
        },
    )

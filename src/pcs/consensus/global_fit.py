"""Robust evaluator-side global consensus fitting."""

from __future__ import annotations

from pcs.consensus.aggregation import bounded_inverse_error, clamp01, ratio_score
from pcs.consensus.graph import RegionGraph, RegionGraphNode
from pcs.geometry.camera import fit_weighted_horizontal_horizon
from pcs.geometry.types import GlobalCameraFitResult, PatchGeometricSignature, RegionalHypothesisMatch
from pcs.utils.config import ConsensusConfig


def _match_lookup(
    matches: list[RegionalHypothesisMatch],
) -> dict[tuple[str, str], RegionalHypothesisMatch]:
    return {
        tuple(sorted((match.patch_id_a, match.patch_id_b))): match
        for match in matches
    }


def _node_lookup(graph: RegionGraph) -> dict[str, RegionGraphNode]:
    return {node.patch_id: node for node in graph.nodes}


def _candidate_score_for_growth(
    candidate_patch_id: str,
    consensus_patch_ids: set[str],
    matches_by_pair: dict[tuple[str, str], RegionalHypothesisMatch],
    config: ConsensusConfig,
) -> tuple[float, int]:
    compatible_scores: list[float] = []
    for patch_id in sorted(consensus_patch_ids):
        key = tuple(sorted((candidate_patch_id, patch_id)))
        match = matches_by_pair.get(key)
        if match is None:
            continue
        if match.compatibility_score >= config.compatibility_threshold:
            compatible_scores.append(match.compatibility_score)
    if not compatible_scores:
        return (0.0, 0)
    return (sum(compatible_scores) / len(compatible_scores), len(compatible_scores))


def _summarize_consensus(
    consensus_patch_ids: set[str],
    graph: RegionGraph,
    matches_by_pair: dict[tuple[str, str], RegionalHypothesisMatch],
    signatures: dict[str, PatchGeometricSignature],
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> GlobalCameraFitResult:
    node_by_patch_id = _node_lookup(graph)
    supported_signatures = [signatures[patch_id] for patch_id in sorted(consensus_patch_ids)]
    compatible_matches: list[RegionalHypothesisMatch] = []
    available_internal_edges = 0

    ordered_patch_ids = sorted(consensus_patch_ids)
    for left_index in range(len(ordered_patch_ids)):
        for right_index in range(left_index + 1, len(ordered_patch_ids)):
            key = tuple(sorted((ordered_patch_ids[left_index], ordered_patch_ids[right_index])))
            match = matches_by_pair.get(key)
            if match is None:
                continue
            available_internal_edges += 1
            if match.compatibility_score >= config.compatibility_threshold:
                compatible_matches.append(match)

    mean_error = (
        sum(match.geometric_error for match in compatible_matches) / len(compatible_matches)
        if compatible_matches
        else 1.0
    )
    coverage_score = ratio_score(len(consensus_patch_ids), max(config.target_supported_patches, 1))
    support_score = (
        sum(node_by_patch_id[patch_id].weight for patch_id in consensus_patch_ids)
        / max(len(consensus_patch_ids), 1)
    )
    connectivity_score = ratio_score(len(compatible_matches), max(available_internal_edges, 1))
    error_score = bounded_inverse_error(mean_error, config.error_soft_limit)
    size_factor = ratio_score(len(consensus_patch_ids), max(config.min_consensus_size, 1))
    global_score = clamp01(
        size_factor * (coverage_score + support_score + connectivity_score + error_score) / 4.0
    )

    image_center = (image_width * 0.5, image_height * 0.5)
    fitted_horizon, horizon_metadata = fit_weighted_horizontal_horizon(
        supported_signatures,
        image_center=image_center,
    )

    return GlobalCameraFitResult(
        success=len(consensus_patch_ids) >= config.min_consensus_size and bool(compatible_matches),
        score=global_score,
        num_supported_patches=len(consensus_patch_ids),
        num_consistent_matches=len(compatible_matches),
        mean_error=float(mean_error),
        fitted_horizon=fitted_horizon,
        metadata={
            "coverage_score": float(coverage_score),
            "support_score": float(support_score),
            "connectivity_score": float(connectivity_score),
            "error_score": float(error_score),
            "size_factor": float(size_factor),
            "supported_patch_ids": ordered_patch_ids,
            "matched_patch_pairs": [
                [match.patch_id_a, match.patch_id_b] for match in compatible_matches
            ],
            "available_internal_edges": float(available_internal_edges),
            **horizon_metadata,
        },
    )


def fit_global_camera_consensus(
    graph: RegionGraph,
    signatures: dict[str, PatchGeometricSignature],
    matches: list[RegionalHypothesisMatch],
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> GlobalCameraFitResult:
    """Fit an approximate global explanation by greedy seed-and-grow consensus."""

    if not graph.nodes:
        return GlobalCameraFitResult(
            success=False,
            score=0.0,
            num_supported_patches=0,
            num_consistent_matches=0,
            mean_error=1.0,
            fitted_horizon=None,
            metadata={"reason": "no_supported_patches"},
        )

    matches_by_pair = _match_lookup(matches)
    ordered_nodes = sorted(graph.nodes, key=lambda node: (-node.weight, node.patch_id))
    best_result: GlobalCameraFitResult | None = None

    for seed_node in ordered_nodes:
        consensus_patch_ids = {seed_node.patch_id}
        remaining_patch_ids = {
            node.patch_id for node in graph.nodes if node.patch_id != seed_node.patch_id
        }

        while remaining_patch_ids:
            scored_candidates = []
            for candidate_patch_id in sorted(remaining_patch_ids):
                avg_score, num_supporting_matches = _candidate_score_for_growth(
                    candidate_patch_id=candidate_patch_id,
                    consensus_patch_ids=consensus_patch_ids,
                    matches_by_pair=matches_by_pair,
                    config=config,
                )
                scored_candidates.append((avg_score, num_supporting_matches, candidate_patch_id))

            best_avg_score, best_supporting_matches, best_candidate_patch_id = max(scored_candidates)
            if (
                best_avg_score < config.min_edge_score_for_growth
                or best_supporting_matches <= 0
            ):
                break

            consensus_patch_ids.add(best_candidate_patch_id)
            remaining_patch_ids.remove(best_candidate_patch_id)

        result = _summarize_consensus(
            consensus_patch_ids=consensus_patch_ids,
            graph=graph,
            matches_by_pair=matches_by_pair,
            signatures=signatures,
            image_width=image_width,
            image_height=image_height,
            config=config,
        )
        if best_result is None or (
            result.score,
            result.num_supported_patches,
            result.num_consistent_matches,
            -result.mean_error,
        ) > (
            best_result.score,
            best_result.num_supported_patches,
            best_result.num_consistent_matches,
            -best_result.mean_error,
        ):
            best_result = result

    assert best_result is not None
    return best_result


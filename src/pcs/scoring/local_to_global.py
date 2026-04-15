"""Local-to-global Milestone 2 PCS scoring."""

from __future__ import annotations

from pcs.consensus.aggregation import clamp01
from pcs.consensus.compatibility import score_region_graph_matches
from pcs.consensus.global_fit import fit_global_camera_consensus
from pcs.consensus.graph import build_region_graph
from pcs.consensus.inconsistency import compute_incompatibility_penalty
from pcs.consensus.signatures import build_patch_signatures
from pcs.geometry.types import ApplicabilityResult, LineSet, LocalToGlobalPCSResult, RegionalHypothesis
from pcs.scoring.baseline import compute_local_score, compute_regional_score
from pcs.utils.config import ConsensusConfig, LocalToGlobalScoringConfig, ScoringConfig


def compute_local_to_global_pcs(
    line_set: LineSet,
    hypotheses: list[RegionalHypothesis],
    applicability: ApplicabilityResult,
    baseline_scoring_config: ScoringConfig,
    consensus_config: ConsensusConfig,
    scoring_config: LocalToGlobalScoringConfig,
) -> tuple[LocalToGlobalPCSResult, dict[str, object]]:
    """Compute the Milestone 2 local-to-global evaluator score."""

    local_quality_score = compute_local_score(line_set, baseline_scoring_config)
    regional_quality_score = compute_regional_score(hypotheses, baseline_scoring_config)
    signatures = build_patch_signatures(
        hypotheses=hypotheses,
        image_width=line_set.image_width,
        image_height=line_set.image_height,
        config=consensus_config,
    )
    graph = build_region_graph(hypotheses, consensus_config)
    matches = score_region_graph_matches(
        graph=graph,
        hypotheses=hypotheses,
        signatures=signatures,
        image_width=line_set.image_width,
        image_height=line_set.image_height,
        config=consensus_config,
    )
    global_fit = fit_global_camera_consensus(
        graph=graph,
        signatures=signatures,
        matches=matches,
        image_width=line_set.image_width,
        image_height=line_set.image_height,
        config=consensus_config,
    )
    incompatibility_penalty, inconsistency_metadata = compute_incompatibility_penalty(
        graph=graph,
        matches=matches,
        global_fit=global_fit,
        config=consensus_config,
    )

    coherence_score = 1.0 - incompatibility_penalty
    structural_score = clamp01(
        (scoring_config.local_weight * local_quality_score)
        + (scoring_config.regional_weight * regional_quality_score)
        + (scoring_config.global_weight * global_fit.score)
        + (scoring_config.coherence_weight * coherence_score)
    )
    pcs_score = clamp01(applicability.confidence * structural_score)

    not_applicable_reason = None
    if not applicability.passed:
        not_applicable_reason = "low_applicability_confidence"
    elif not graph.nodes:
        not_applicable_reason = "no_supported_patches_for_consensus"

    result = LocalToGlobalPCSResult(
        pcs_score=pcs_score,
        applicability_confidence=applicability.confidence,
        local_quality_score=local_quality_score,
        regional_quality_score=regional_quality_score,
        global_consensus_score=global_fit.score,
        incompatibility_penalty=incompatibility_penalty,
        num_patches=len(hypotheses),
        num_supported_patches=len(graph.nodes),
        metadata={
            "applicability_passed": applicability.passed,
            "structural_score": structural_score,
            "coherence_score": coherence_score,
            "compared_edges": len(graph.edges),
            "matched_edges": len(matches),
            "inconsistent_edges": inconsistency_metadata["inconsistent_edges"],
            "patch_inconsistency_map": inconsistency_metadata["patch_inconsistency_map"],
            "not_applicable_reason": not_applicable_reason,
        },
    )
    artifacts: dict[str, object] = {
        "signatures": signatures,
        "graph": graph,
        "matches": matches,
        "global_fit": global_fit,
        "inconsistency": inconsistency_metadata,
    }
    return (result, artifacts)

"""Local-to-global Milestone 2 PCS scoring."""

from __future__ import annotations

import numpy as np

from pcs.consensus.aggregation import clamp01
from pcs.consensus.compatibility import score_region_graph_matches
from pcs.consensus.global_fit import fit_global_camera_consensus
from pcs.consensus.graph import build_region_graph
from pcs.consensus.inconsistency import compute_incompatibility_penalty
from pcs.consensus.signatures import build_patch_signatures
from pcs.geometry.types import (
    ApplicabilityResult,
    GlobalCameraFitResult,
    LineSet,
    LocalToGlobalPCSResult,
    RegionalHypothesis,
    RegionalHypothesisMatch,
)
from pcs.scoring.baseline import compute_local_score, compute_regional_score
from pcs.utils.config import ConsensusConfig, LocalToGlobalScoringConfig, ScoringConfig


def _compute_v2_global_score(
    global_fit: GlobalCameraFitResult,
    matches: list[RegionalHypothesisMatch],
    consensus_patch_ids: set[str],
    total_supported_patches: int,
) -> tuple[float, dict[str, float]]:
    """Compute the coverage-weighted consistency global score (v2).

    global_score_new = coverage_ratio * internal_quality * (1 - incompatibility_severity)
    """
    # Coverage ratio: fraction of supported patches in consensus
    if total_supported_patches <= 0:
        coverage_ratio = 0.0
    else:
        coverage_ratio = clamp01(len(consensus_patch_ids) / total_supported_patches)

    # Internal quality: mean consensus compatibility from global_fit
    internal_quality = clamp01(global_fit.metadata.get("mean_consensus_compatibility", 0.0))

    # Incompatibility severity: mean (1 - compat) for excluded patches vs consensus
    excluded_patch_ids = set()
    # Build a lookup of all supported patch IDs from the graph
    all_supported = set(global_fit.metadata.get("supported_patch_ids", []))
    # The "total_supported_patches" count includes patches NOT in this consensus
    # We need to find excluded patches by looking at matches
    all_patch_ids_in_matches: set[str] = set()
    for m in matches:
        all_patch_ids_in_matches.add(m.patch_id_a)
        all_patch_ids_in_matches.add(m.patch_id_b)
    excluded_patch_ids = (all_patch_ids_in_matches | all_supported) - consensus_patch_ids

    if not excluded_patch_ids:
        incompatibility_severity = 0.0
    else:
        severities: list[float] = []
        for ex_pid in sorted(excluded_patch_ids):
            compat_scores: list[float] = []
            for m in matches:
                if m.patch_id_a == ex_pid and m.patch_id_b in consensus_patch_ids:
                    compat_scores.append(m.compatibility_score)
                elif m.patch_id_b == ex_pid and m.patch_id_a in consensus_patch_ids:
                    compat_scores.append(m.compatibility_score)
            if compat_scores:
                severities.append(1.0 - (sum(compat_scores) / len(compat_scores)))
        incompatibility_severity = clamp01(
            sum(severities) / len(severities) if severities else 0.0
        )

    global_score_v2 = clamp01(coverage_ratio * internal_quality * (1.0 - incompatibility_severity))

    details = {
        "coverage_ratio": float(coverage_ratio),
        "internal_quality": float(internal_quality),
        "incompatibility_severity": float(incompatibility_severity),
        "num_excluded_patches": len(excluded_patch_ids),
        "consensus_patches": len(consensus_patch_ids),
        "total_supported_patches": total_supported_patches,
    }
    return global_score_v2, details


def _compute_v3_global_score(
    matches: list[RegionalHypothesisMatch],
) -> tuple[float, dict[str, float]]:
    """Compute global score from sub-component percentiles (v3).

    The combined compatibility_score dilutes the most sensitive signals
    (horizon_score has 15% weight, direction_score has 50%). V3 extracts
        individual component scores from match metadata and emphasizes the
        lower tail (5th percentile) to capture corruption-sensitive edges.

        global_v3 = (0.55 * hor_p05 + 0.30 * dir_p05 + 0.15 * compat_p05)
                    * (1 - 0.5 * low_tail_ratio)
    """
    if not matches:
        return 0.0, {
            "hor_p05": 0.0,
            "dir_p05": 0.0,
            "compat_p05": 0.0,
            "mean_compat": 0.0,
            "low_tail_ratio": 0.0,
            "n_edges": 0,
        }

    horizon_scores: list[float] = []
    direction_scores: list[float] = []
    compat_scores: list[float] = []

    for m in matches:
        compat_scores.append(m.compatibility_score)
        hs = m.metadata.get("horizon_score")
        if hs is not None:
            horizon_scores.append(float(hs))
        ds = m.metadata.get("direction_score")
        if ds is not None:
            direction_scores.append(float(ds))

    mean_compat = float(np.mean(compat_scores))
    compat_p05 = float(np.percentile(compat_scores, 5)) if len(compat_scores) >= 2 else mean_compat
    hor_p05 = float(np.percentile(horizon_scores, 5)) if len(horizon_scores) >= 2 else compat_p05
    dir_p05 = float(np.percentile(direction_scores, 5)) if len(direction_scores) >= 2 else compat_p05

    # Tail-ratio captures localized inconsistencies that only affect a subset of edges.
    low_horizon_ratio = float(np.mean(np.array(horizon_scores) < 0.4)) if horizon_scores else 0.0
    low_direction_ratio = float(np.mean(np.array(direction_scores) < 0.4)) if direction_scores else 0.0
    low_tail_ratio = 0.5 * low_horizon_ratio + 0.5 * low_direction_ratio

    base_score = 0.55 * hor_p05 + 0.30 * dir_p05 + 0.15 * compat_p05
    global_score = clamp01(base_score * (1.0 - 0.5 * low_tail_ratio))

    details = {
        "hor_p05": hor_p05,
        "dir_p05": dir_p05,
        "compat_p05": compat_p05,
        "mean_compat": mean_compat,
        "low_tail_ratio": low_tail_ratio,
        "n_edges": len(matches),
        "n_horizon_scores": len(horizon_scores),
        "n_direction_scores": len(direction_scores),
    }
    return global_score, details


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

    use_v2 = scoring_config.version == "v2"
    use_v3 = scoring_config.version == "v3"
    use_v4 = scoring_config.version == "v4"

    if use_v3:
        global_score_v3, v3_details = _compute_v3_global_score(matches=matches)
        structural_score = clamp01(
            (scoring_config.local_weight * local_quality_score)
            + (scoring_config.regional_weight * regional_quality_score)
            + (scoring_config.global_weight * global_score_v3)
        )
        effective_global_score = global_score_v3
        coherence_score = 1.0 - incompatibility_penalty  # computed for reporting
        v2_details: dict[str, float] = {}
    elif use_v4:
        consensus_patch_ids = set(global_fit.metadata.get("supported_patch_ids", []))
        total_supported = global_fit.metadata.get("total_supported_patches", len(graph.nodes))
        global_score_v4, v2_details = _compute_v2_global_score(
            global_fit=global_fit,
            matches=matches,
            consensus_patch_ids=consensus_patch_ids,
            total_supported_patches=total_supported,
        )
        coherence_score = 1.0 - incompatibility_penalty
        structural_score = clamp01(
            (scoring_config.local_weight * local_quality_score)
            + (scoring_config.regional_weight * regional_quality_score)
            + (scoring_config.global_weight * global_score_v4)
            + (scoring_config.coherence_weight * coherence_score)
        )
        effective_global_score = global_score_v4
    elif use_v2:
        consensus_patch_ids = set(global_fit.metadata.get("supported_patch_ids", []))
        total_supported = global_fit.metadata.get("total_supported_patches", len(graph.nodes))
        global_score_v2, v2_details = _compute_v2_global_score(
            global_fit=global_fit,
            matches=matches,
            consensus_patch_ids=consensus_patch_ids,
            total_supported_patches=total_supported,
        )
        structural_score = clamp01(
            (scoring_config.local_weight * local_quality_score)
            + (scoring_config.regional_weight * regional_quality_score)
            + (scoring_config.global_weight * global_score_v2)
        )
        effective_global_score = global_score_v2
        coherence_score = 1.0 - incompatibility_penalty  # still computed for reporting
    else:
        coherence_score = 1.0 - incompatibility_penalty
        structural_score = clamp01(
            (scoring_config.local_weight * local_quality_score)
            + (scoring_config.regional_weight * regional_quality_score)
            + (scoring_config.global_weight * global_fit.score)
            + (scoring_config.coherence_weight * coherence_score)
        )
        effective_global_score = global_fit.score
        v2_details = {}

    pcs_score = clamp01(applicability.confidence * structural_score)

    not_applicable_reason = None
    if not applicability.passed:
        not_applicable_reason = "low_applicability_confidence"
    elif not graph.nodes:
        not_applicable_reason = "no_supported_patches_for_consensus"

    metadata: dict[str, object] = {
        "applicability_passed": applicability.passed,
        "structural_score": structural_score,
        "coherence_score": coherence_score,
        "compared_edges": len(graph.edges),
        "matched_edges": len(matches),
        "inconsistent_edges": inconsistency_metadata["inconsistent_edges"],
        "patch_inconsistency_map": inconsistency_metadata["patch_inconsistency_map"],
        "not_applicable_reason": not_applicable_reason,
        "scoring_version": scoring_config.version,
    }
    if v2_details and not use_v4:
        metadata["v2_global_details"] = v2_details
    if use_v4 and v2_details:
        metadata["v4_global_details"] = v2_details
    if use_v3:
        metadata["v3_global_details"] = v3_details

    result = LocalToGlobalPCSResult(
        pcs_score=pcs_score,
        applicability_confidence=applicability.confidence,
        local_quality_score=local_quality_score,
        regional_quality_score=regional_quality_score,
        global_consensus_score=effective_global_score,
        incompatibility_penalty=incompatibility_penalty,
        num_patches=len(hypotheses),
        num_supported_patches=len(graph.nodes),
        metadata=metadata,
    )
    artifacts: dict[str, object] = {
        "signatures": signatures,
        "graph": graph,
        "matches": matches,
        "global_fit": global_fit,
        "inconsistency": inconsistency_metadata,
    }
    return (result, artifacts)

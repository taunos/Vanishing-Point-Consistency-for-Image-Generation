"""Patch-pair compatibility scoring for local-to-global evaluation."""

from __future__ import annotations

import math

from pcs.consensus.aggregation import clamp01, weighted_average
from pcs.consensus.graph import RegionGraph
from pcs.geometry.camera import (
    horizon_y_proxy_from_vp,
    projective_angular_distance_deg,
    vp_to_projective_direction,
)
from pcs.geometry.types import PatchGeometricSignature, RegionalHypothesis, RegionalHypothesisMatch
from pcs.utils.config import ConsensusConfig


def _angle_between_unit_vectors(
    vec_a: tuple[float, float] | None,
    vec_b: tuple[float, float] | None,
) -> float | None:
    return projective_angular_distance_deg(vec_a, vec_b)


def _normalize_direction(
    patch_center: tuple[float, float],
    vp: tuple[float, float],
) -> tuple[float, float] | None:
    return vp_to_projective_direction(patch_center, vp)


def orientation_histogram_similarity(
    hist_a: list[float] | None,
    hist_b: list[float] | None,
) -> float | None:
    """Return histogram-intersection similarity for orientation summaries."""

    if hist_a is None or hist_b is None or len(hist_a) != len(hist_b):
        return None
    return clamp01(sum(min(float(a), float(b)) for a, b in zip(hist_a, hist_b, strict=True)))


def score_signature_pair(
    signature_a: PatchGeometricSignature,
    signature_b: PatchGeometricSignature,
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> tuple[float, float, dict[str, float]]:
    """Score compatibility between two patch-level geometric signatures."""

    component_scores: list[tuple[float, float]] = []
    component_errors: list[tuple[float, float]] = []
    metadata: dict[str, float] = {}

    direction_angle = _angle_between_unit_vectors(
        signature_a.normalized_direction,
        signature_b.normalized_direction,
    )
    if direction_angle is not None:
        sigma = max(config.directional_sigma_deg, 1e-6)
        direction_score = math.exp(-((direction_angle / sigma) ** 2))
        direction_error = direction_angle / 180.0
        component_scores.append((direction_score, config.direction_weight))
        component_errors.append((direction_error, config.direction_weight))
        metadata["direction_angle_deg"] = float(direction_angle)
        metadata["direction_score"] = float(direction_score)

    orientation_score = orientation_histogram_similarity(
        signature_a.orientation_histogram,
        signature_b.orientation_histogram,
    )
    if orientation_score is not None:
        component_scores.append((orientation_score, config.orientation_weight))
        component_errors.append((1.0 - orientation_score, config.orientation_weight))
        metadata["orientation_score"] = float(orientation_score)

    y_proxy_a = signature_a.metadata.get("horizon_y_proxy")
    y_proxy_b = signature_b.metadata.get("horizon_y_proxy")
    if y_proxy_a is not None and y_proxy_b is not None:
        y_diff_ratio = abs(float(y_proxy_a) - float(y_proxy_b)) / max(float(image_height), 1.0)
        horizon_score = math.exp(-(y_diff_ratio / max(config.horizon_y_tolerance_ratio, 1e-6)))
        component_scores.append((horizon_score, config.horizon_weight))
        component_errors.append((y_diff_ratio, config.horizon_weight))
        metadata["horizon_score"] = float(horizon_score)
        metadata["horizon_y_diff_ratio"] = float(y_diff_ratio)

    contradiction_penalty = 0.0
    if direction_angle is not None and direction_angle >= config.contradiction_angle_deg:
        contradiction_penalty = clamp01(
            (direction_angle - config.contradiction_angle_deg)
            / max(180.0 - config.contradiction_angle_deg, 1e-6)
        )
    contradiction_score = 1.0 - contradiction_penalty

    if config.vp_position_weight > 0.0:
        image_diagonal = math.hypot(image_width, image_height)
        vp_position_score = compute_vp_position_divergence(
            signature_a, signature_b, image_diagonal, config
        )
        component_scores.append((vp_position_score, config.vp_position_weight))
        component_errors.append((1.0 - vp_position_score, config.vp_position_weight))
        metadata["vp_position_score"] = float(vp_position_score)

    if config.manhattan_assisted and orientation_score is not None:
        dominant_bin_a = max(
            range(len(signature_a.orientation_histogram or [])),
            key=lambda index: signature_a.orientation_histogram[index],
            default=0,
        )
        dominant_bin_b = max(
            range(len(signature_b.orientation_histogram or [])),
            key=lambda index: signature_b.orientation_histogram[index],
            default=0,
        )
        bin_gap = abs(dominant_bin_a - dominant_bin_b)
        orthogonal_gap = len(signature_a.orientation_histogram or []) // 2
        manhattan_bonus = 0.0
        if bin_gap == 0 or bin_gap == orthogonal_gap:
            manhattan_bonus = 0.05
        contradiction_score = clamp01(contradiction_score + manhattan_bonus)
        metadata["manhattan_bonus"] = float(manhattan_bonus)

    component_scores.append((contradiction_score, config.contradiction_weight))
    component_errors.append((contradiction_penalty, config.contradiction_weight))
    metadata["contradiction_penalty"] = float(contradiction_penalty)

    compatibility_score = weighted_average(component_scores)
    geometric_error = weighted_average(component_errors)
    return (compatibility_score, geometric_error, metadata)


def compute_vp_position_divergence(
    sig_a: PatchGeometricSignature,
    sig_b: PatchGeometricSignature,
    image_diagonal: float,
    config: ConsensusConfig,
) -> float:
    """Measure whether two patches imply compatible VP positions.

    For each VP direction shared by both patches (within angular tolerance),
    compares spatial VP positions. Returns a score in [0, 1] where 1.0 means
    perfect agreement and 0.0 means maximal divergence.
    """
    cands_a = sig_a.vp_candidates
    cands_b = sig_b.vp_candidates

    if not cands_a or not cands_b:
        return config.vp_position_no_match_default

    img_cx = float(sig_a.metadata.get("image_center_x", 0.0))
    img_cy = float(sig_a.metadata.get("image_center_y", 0.0))
    tolerance_rad = math.radians(config.vp_direction_match_tolerance_deg)
    infinity_threshold = config.vp_position_infinity_factor * max(image_diagonal, 1e-6)
    sigma = max(config.vp_position_sigma, 1e-6)

    def _is_infinity(x: float, y: float) -> bool:
        return max(abs(x), abs(y)) > infinity_threshold

    def _vp_angle(x: float, y: float) -> float:
        return math.atan2(y - img_cy, x - img_cx) % math.pi

    def _angle_diff(a1: float, a2: float) -> float:
        d = abs(a1 - a2) % math.pi
        return min(d, math.pi - d)

    # Precompute B angles for efficiency
    b_angles = [_vp_angle(x, y) for x, y, *_ in cands_b]

    used_b: set[int] = set()
    pair_scores: list[tuple[float, float]] = []

    for x_a, y_a, score_a, _n_a in cands_a:
        angle_a = _vp_angle(x_a, y_a)

        # Find best unmatched B candidate by direction
        best_b_idx: int | None = None
        best_diff = float("inf")
        for i_b, angle_b in enumerate(b_angles):
            if i_b in used_b:
                continue
            diff = _angle_diff(angle_a, angle_b)
            if diff < best_diff:
                best_diff = diff
                best_b_idx = i_b

        if best_b_idx is None or best_diff > tolerance_rad:
            continue

        used_b.add(best_b_idx)
        x_b, y_b, score_b, _n_b = cands_b[best_b_idx]

        a_inf = _is_infinity(x_a, y_a)
        b_inf = _is_infinity(x_b, y_b)

        if a_inf and b_inf:
            pos_score = 1.0
        elif a_inf or b_inf:
            pos_score = 0.0
        else:
            dist = math.hypot(x_a - x_b, y_a - y_b) / max(image_diagonal, 1e-6)
            pos_score = math.exp(-dist / sigma)

        weight = min(float(score_a), float(score_b))
        pair_scores.append((pos_score, weight))

    if not pair_scores:
        return config.vp_position_no_match_default

    total_weight = sum(w for _, w in pair_scores)
    if total_weight <= 0.0:
        return clamp01(sum(s for s, _ in pair_scores) / len(pair_scores))
    return clamp01(sum(s * w for s, w in pair_scores) / total_weight)


def _signature_for_candidate(
    base_signature: PatchGeometricSignature,
    candidate_vp: tuple[float, float],
) -> PatchGeometricSignature:
    patch_center = (
        float(base_signature.metadata["patch_center_x"]),
        float(base_signature.metadata["patch_center_y"]),
    )
    image_center = (
        float(base_signature.metadata["image_center_x"]),
        float(base_signature.metadata["image_center_y"]),
    )
    return PatchGeometricSignature(
        patch_id=base_signature.patch_id,
        dominant_vp=candidate_vp,
        support_score=base_signature.support_score,
        stability_score=base_signature.stability_score,
        orientation_histogram=base_signature.orientation_histogram,
        normalized_direction=_normalize_direction(patch_center, candidate_vp),
        vp_candidates=base_signature.vp_candidates,
        metadata={
            **base_signature.metadata,
            "horizon_y_proxy": horizon_y_proxy_from_vp(candidate_vp, image_center=image_center),
        },
    )


def match_regional_hypotheses(
    hypothesis_a: RegionalHypothesis,
    hypothesis_b: RegionalHypothesis,
    signatures: dict[str, PatchGeometricSignature],
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> RegionalHypothesisMatch:
    """Find the best candidate-to-candidate compatibility match between patches."""

    signature_a = signatures[hypothesis_a.patch.patch_id]
    signature_b = signatures[hypothesis_b.patch.patch_id]
    best_match: RegionalHypothesisMatch | None = None

    for vp_idx_a, candidate_a in enumerate(hypothesis_a.vp_candidates):
        for vp_idx_b, candidate_b in enumerate(hypothesis_b.vp_candidates):
            candidate_signature_a = _signature_for_candidate(signature_a, (candidate_a.x, candidate_a.y))
            candidate_signature_b = _signature_for_candidate(signature_b, (candidate_b.x, candidate_b.y))
            compatibility_score, geometric_error, metadata = score_signature_pair(
                signature_a=candidate_signature_a,
                signature_b=candidate_signature_b,
                image_width=image_width,
                image_height=image_height,
                config=config,
            )
            match = RegionalHypothesisMatch(
                patch_id_a=hypothesis_a.patch.patch_id,
                patch_id_b=hypothesis_b.patch.patch_id,
                vp_idx_a=vp_idx_a,
                vp_idx_b=vp_idx_b,
                vp_position_a=(candidate_a.x, candidate_a.y),
                vp_position_b=(candidate_b.x, candidate_b.y),
                compatibility_score=compatibility_score,
                geometric_error=geometric_error,
                metadata=metadata,
            )
            if best_match is None or (
                match.compatibility_score,
                -match.geometric_error,
                -match.vp_idx_a,
                -match.vp_idx_b,
            ) > (
                best_match.compatibility_score,
                -best_match.geometric_error,
                -best_match.vp_idx_a,
                -best_match.vp_idx_b,
            ):
                best_match = match

    if best_match is None:
        return RegionalHypothesisMatch(
            patch_id_a=hypothesis_a.patch.patch_id,
            patch_id_b=hypothesis_b.patch.patch_id,
            vp_idx_a=-1,
            vp_idx_b=-1,
            vp_position_a=None,
            vp_position_b=None,
            compatibility_score=0.0,
            geometric_error=1.0,
            metadata={"compatible": False},
        )

    return RegionalHypothesisMatch(
        patch_id_a=best_match.patch_id_a,
        patch_id_b=best_match.patch_id_b,
        vp_idx_a=best_match.vp_idx_a,
        vp_idx_b=best_match.vp_idx_b,
        vp_position_a=best_match.vp_position_a,
        vp_position_b=best_match.vp_position_b,
        compatibility_score=best_match.compatibility_score,
        geometric_error=best_match.geometric_error,
        metadata={
            **best_match.metadata,
            "compatible": best_match.compatibility_score >= config.compatibility_threshold,
        },
    )


def score_region_graph_matches(
    graph: RegionGraph,
    hypotheses: list[RegionalHypothesis],
    signatures: dict[str, PatchGeometricSignature],
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> list[RegionalHypothesisMatch]:
    """Score all eligible graph edges using best patch-pair compatibility."""

    hypothesis_by_patch_id = {hypothesis.patch.patch_id: hypothesis for hypothesis in hypotheses}
    matches: list[RegionalHypothesisMatch] = []
    for edge in graph.edges:
        matches.append(
            match_regional_hypotheses(
                hypothesis_a=hypothesis_by_patch_id[edge.source_patch_id],
                hypothesis_b=hypothesis_by_patch_id[edge.target_patch_id],
                signatures=signatures,
                image_width=image_width,
                image_height=image_height,
                config=config,
            )
        )
    matches.sort(key=lambda match: (match.patch_id_a, match.patch_id_b))
    return matches

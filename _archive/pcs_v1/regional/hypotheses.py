"""Baseline patch-local vanishing-point hypothesis estimation.

This module only estimates per-patch hypotheses and patch-level support or
stability signals. Cross-patch compatibility and global consensus logic belong
in `pcs.consensus`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

from pcs.geometry.lines import segment_overlap_ratio_with_rect
from pcs.geometry.types import LineSet, Patch, RegionalHypothesis, VanishingPointCandidate
from pcs.geometry.vp import intersect_segments_as_lines, score_vp_candidate
from pcs.utils.config import RegionalConfig


@dataclass(slots=True)
class _Cluster:
    x: float
    y: float
    count: int


def _select_patch_lines(
    lines: Iterable,
    patch: Patch,
    min_overlap_ratio: float,
) -> list:
    rect = (float(patch.x0), float(patch.y0), float(patch.x1), float(patch.y1))
    selected = []
    for line in lines:
        if segment_overlap_ratio_with_rect(line, rect) >= min_overlap_ratio:
            selected.append(line)
    return selected


def _extract_line_arrays(patch_lines: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-extract NumPy arrays from line segments for vectorized ops.

    Returns (homogeneous_lines (N,3), angles (N,), lengths (N,)).
    """
    n = len(patch_lines)
    hom = np.empty((n, 3), dtype=np.float64)
    angles = np.empty(n, dtype=np.float64)
    lengths = np.empty(n, dtype=np.float64)
    for i, seg in enumerate(patch_lines):
        p1 = np.array([seg.x1, seg.y1, 1.0])
        p2 = np.array([seg.x2, seg.y2, 1.0])
        line = np.cross(p1, p2)
        norm = math.hypot(line[0], line[1])
        if norm > 0.0:
            line /= norm
        hom[i] = line
        angles[i] = seg.angle_rad
        lengths[i] = seg.length
    return hom, angles, lengths


def _vectorized_pairwise_intersections(
    hom_lines: np.ndarray,
    angles: np.ndarray,
    patch_center: tuple[float, float],
    image_diagonal: float,
    min_angle_diff_rad: float,
    max_vp_distance: float,
) -> np.ndarray:
    """Vectorized computation of pairwise line intersections.

    Returns an (M, 2) array of valid intersection points.
    """
    n = hom_lines.shape[0]
    if n < 2:
        return np.empty((0, 2), dtype=np.float64)

    # Build upper-triangle index pairs
    idx_i, idx_j = np.triu_indices(n, k=1)

    # Angle filtering: smallest undirected angle difference
    a_i = angles[idx_i]
    a_j = angles[idx_j]
    diff = np.abs(a_i - a_j) % math.pi
    diff = np.minimum(diff, math.pi - diff)
    angle_mask = diff >= min_angle_diff_rad

    idx_i = idx_i[angle_mask]
    idx_j = idx_j[angle_mask]
    if len(idx_i) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Vectorized cross product for homogeneous line intersections
    li = hom_lines[idx_i]  # (M, 3)
    lj = hom_lines[idx_j]  # (M, 3)
    pts = np.cross(li, lj)  # (M, 3)

    # Filter degenerate (parallel) intersections
    w = pts[:, 2]
    finite_mask = np.abs(w) > 1e-9
    pts = pts[finite_mask]
    w = pts[:, 2]

    # Dehomogenize
    xy = pts[:, :2] / w[:, np.newaxis]

    # Filter non-finite results
    finite_mask2 = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite_mask2]

    if xy.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Distance from patch center
    cx, cy = patch_center
    dist_sq = (xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2
    max_dist = max_vp_distance * image_diagonal
    dist_mask = dist_sq <= max_dist * max_dist
    return xy[dist_mask]


def _cluster_intersections(
    intersections: list[tuple[float, float]] | np.ndarray,
    radius_px: float,
) -> list[_Cluster]:
    """Cluster intersections via connected components using scipy sparse graph.

    All heavy lifting (neighbor search + connected components) runs in C.
    """
    if isinstance(intersections, np.ndarray):
        if intersections.shape[0] == 0:
            return []
        points = intersections
    else:
        if not intersections:
            return []
        points = np.array(sorted(intersections), dtype=np.float64)

    # Sort for determinism (lexicographic on x, then y)
    sort_idx = np.lexsort((points[:, 1], points[:, 0]))
    points = points[sort_idx]
    n = points.shape[0]

    # Use cKDTree.query_pairs for C-level pairwise distance filtering
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius_px, output_type='ndarray')

    if pairs.shape[0] == 0:
        # No pairs within radius → each point is its own cluster
        return [
            _Cluster(x=float(points[i, 0]), y=float(points[i, 1]), count=1)
            for i in range(n)
        ]

    # Build sparse adjacency matrix and find connected components (all C-level)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    row = pairs[:, 0]
    col = pairs[:, 1]
    data = np.ones(len(row), dtype=np.int8)
    # Symmetric adjacency
    adj = coo_matrix(
        (np.concatenate([data, data]),
         (np.concatenate([row, col]), np.concatenate([col, row]))),
        shape=(n, n),
    )

    n_components, labels = connected_components(adj, directed=False)

    clusters: list[_Cluster] = []
    for comp_id in range(n_components):
        mask = labels == comp_id
        component_pts = points[mask]
        clusters.append(
            _Cluster(
                x=float(component_pts[:, 0].mean()),
                y=float(component_pts[:, 1].mean()),
                count=int(mask.sum()),
            )
        )
    return clusters


def _empty_hypothesis(patch: Patch, num_lines: int) -> RegionalHypothesis:
    return RegionalHypothesis(
        patch=patch,
        vp_candidates=[],
        support_score=0.0,
        stability_score=0.0,
        num_lines=num_lines,
        metadata={"viable": False, "orientation_histogram": None},
    )


def _compute_orientation_histogram(
    patch_lines: list,
    bins: int,
) -> list[float] | None:
    if not patch_lines:
        return None
    angles = np.asarray([line.angle_rad for line in patch_lines], dtype=np.float64)
    hist, _ = np.histogram(angles, bins=bins, range=(0.0, math.pi), density=False)
    total = float(hist.sum())
    if total <= 0.0:
        return None
    return [float(value / total) for value in hist]


def _generate_pairwise_intersections(
    patch_lines: list,
    patch_center: tuple[float, float],
    image_diagonal: float,
    config: RegionalConfig,
    _line_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[tuple[float, float]]:
    """Compute pairwise intersections, using vectorized path when arrays are provided."""
    if _line_arrays is not None:
        hom, angles, _lengths = _line_arrays
        pts = _vectorized_pairwise_intersections(
            hom, angles, patch_center, image_diagonal,
            config.min_intersection_line_angle_rad,
            config.max_vp_distance_factor,
        )
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in range(pts.shape[0])]

    # Fallback: original scalar path (kept for compatibility)
    raw_intersections: list[tuple[float, float]] = []
    for left_index in range(len(patch_lines)):
        for right_index in range(left_index + 1, len(patch_lines)):
            point = intersect_segments_as_lines(
                patch_lines[left_index],
                patch_lines[right_index],
                min_angle_difference_rad=config.min_intersection_line_angle_rad,
            )
            if point is None:
                continue
            if (
                math.hypot(point[0] - patch_center[0], point[1] - patch_center[1])
                > config.max_vp_distance_factor * image_diagonal
            ):
                continue
            raw_intersections.append(point)
    return raw_intersections


def _vectorized_score_vp_candidates(
    cluster_xy: np.ndarray,
    line_midpoints: np.ndarray,
    line_angles: np.ndarray,
    line_lengths: np.ndarray,
    angular_inlier_threshold_rad: float,
) -> list[tuple[float, int]]:
    """Score multiple VP candidates against all lines in one vectorized pass.

    Args:
        cluster_xy: (C, 2) array of cluster center coordinates.
        line_midpoints: (N, 2) array of line segment midpoints.
        line_angles: (N,) array of line angles in [0, pi).
        line_lengths: (N,) array of line lengths.
        angular_inlier_threshold_rad: inlier threshold in radians.

    Returns:
        List of (score, num_inliers) for each candidate.
    """
    n_clusters = cluster_xy.shape[0]
    n_lines = line_midpoints.shape[0]

    if n_lines == 0 or n_clusters == 0:
        return [(0.0, 0)] * n_clusters

    # Ray from midpoint to each VP: (C, N, 2)
    ray = cluster_xy[:, np.newaxis, :] - line_midpoints[np.newaxis, :, :]  # (C, N, 2)
    ray_angles = np.arctan2(ray[:, :, 1], ray[:, :, 0])  # (C, N)

    # Normalize ray angles to [0, pi) for undirected comparison
    ray_angles = ray_angles % math.pi

    # Smallest undirected angle difference
    diff = np.abs(ray_angles - line_angles[np.newaxis, :]) % math.pi
    residuals = np.minimum(diff, math.pi - diff)  # (C, N)

    weights = np.maximum(line_lengths, 1.0)  # (N,)
    total_weight = float(weights.sum())
    if total_weight == 0.0:
        return [(0.0, 0)] * n_clusters

    results: list[tuple[float, int]] = []
    for c in range(n_clusters):
        inlier_mask = residuals[c] <= angular_inlier_threshold_rad
        num_inliers = int(inlier_mask.sum())
        agreement = 1.0 - residuals[c] / max(angular_inlier_threshold_rad, 1e-6)
        agreement = np.clip(agreement, 0.0, 1.0)
        weighted_support = float((weights * agreement * inlier_mask).sum())
        results.append((weighted_support / total_weight, num_inliers))

    return results


def _estimate_patch_candidates(
    patch_lines: list,
    patch_center: tuple[float, float],
    image_diagonal: float,
    config: RegionalConfig,
    _line_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[VanishingPointCandidate], dict[str, int]]:
    raw_intersections = _generate_pairwise_intersections(
        patch_lines=patch_lines,
        patch_center=patch_center,
        image_diagonal=image_diagonal,
        config=config,
        _line_arrays=_line_arrays,
    )
    if not raw_intersections:
        return ([], {"num_raw_intersections": 0, "num_clustered_intersections": 0})

    cluster_radius_px = max(config.intersection_dedup_radius_ratio * image_diagonal, 1.0)
    clusters = _cluster_intersections(raw_intersections, radius_px=cluster_radius_px)

    if not clusters:
        return ([], {"num_raw_intersections": len(raw_intersections), "num_clustered_intersections": 0})

    # Vectorized VP scoring: score all clusters at once
    cluster_xy = np.array([[c.x, c.y] for c in clusters], dtype=np.float64)

    if _line_arrays is not None:
        _hom, angles, lengths = _line_arrays
        midpoints = np.array(
            [((s.x1 + s.x2) * 0.5, (s.y1 + s.y2) * 0.5) for s in patch_lines],
            dtype=np.float64,
        )
        scores_inliers = _vectorized_score_vp_candidates(
            cluster_xy, midpoints, angles, lengths,
            config.angular_inlier_threshold_rad,
        )
    else:
        scores_inliers = []
        for cluster in clusters:
            score, num_inliers = score_vp_candidate(
                cluster.x, cluster.y, patch_lines,
                angular_inlier_threshold_rad=config.angular_inlier_threshold_rad,
            )
            scores_inliers.append((score, num_inliers))

    scored_candidates: list[tuple[VanishingPointCandidate, int]] = []
    for idx, cluster in enumerate(clusters):
        score, num_inliers = scores_inliers[idx]
        if num_inliers < config.min_candidate_inliers:
            continue
        scored_candidates.append(
            (
                VanishingPointCandidate(
                    x=cluster.x,
                    y=cluster.y,
                    score=score,
                    num_inliers=num_inliers,
                ),
                cluster.count,
            )
        )

    scored_candidates.sort(
        key=lambda item: (
            item[0].score,
            item[0].num_inliers,
            item[1],
            -abs(item[0].x - patch_center[0]),
            -abs(item[0].y - patch_center[1]),
        ),
        reverse=True,
    )
    return (
        [item[0] for item in scored_candidates[: config.top_k_candidates]],
        {
            "num_raw_intersections": len(raw_intersections),
            "num_clustered_intersections": len(clusters),
        },
    )


def _stable_patch_seed(patch: Patch) -> int:
    digest = hashlib.sha256(patch.patch_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _precompute_intersection_matrix(
    hom_lines: np.ndarray,
    angles: np.ndarray,
    patch_center: tuple[float, float],
    image_diagonal: float,
    min_angle_diff_rad: float,
    max_vp_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute ALL valid pairwise intersections and track which line pair produced each.

    Returns:
        xy: (M, 2) intersection coordinates
        pair_i: (M,) indices of the first line in each pair
        pair_j: (M,) indices of the second line in each pair
        valid_mask: boolean mask that was applied
    """
    n = hom_lines.shape[0]
    if n < 2:
        empty = np.empty((0, 2), dtype=np.float64)
        empty_idx = np.empty(0, dtype=np.intp)
        return empty, empty_idx, empty_idx, np.empty(0, dtype=bool)

    idx_i, idx_j = np.triu_indices(n, k=1)

    a_i = angles[idx_i]
    a_j = angles[idx_j]
    diff = np.abs(a_i - a_j) % math.pi
    diff = np.minimum(diff, math.pi - diff)
    angle_mask = diff >= min_angle_diff_rad

    idx_i = idx_i[angle_mask]
    idx_j = idx_j[angle_mask]
    if len(idx_i) == 0:
        empty = np.empty((0, 2), dtype=np.float64)
        empty_idx = np.empty(0, dtype=np.intp)
        return empty, empty_idx, empty_idx, np.empty(0, dtype=bool)

    li = hom_lines[idx_i]
    lj = hom_lines[idx_j]
    pts = np.cross(li, lj)

    w = pts[:, 2]
    finite_mask = np.abs(w) > 1e-9

    # Apply filter in-place tracking
    valid = finite_mask.copy()
    xy_all = np.full((len(pts), 2), np.nan, dtype=np.float64)
    ok = finite_mask
    xy_all[ok] = pts[ok, :2] / w[ok, np.newaxis]

    finite2 = np.all(np.isfinite(xy_all), axis=1)
    valid &= finite2

    cx, cy = patch_center
    max_dist = max_vp_distance * image_diagonal
    dist_sq = (xy_all[:, 0] - cx) ** 2 + (xy_all[:, 1] - cy) ** 2
    dist_ok = dist_sq <= max_dist * max_dist
    valid &= dist_ok

    xy = xy_all[valid]
    return xy, idx_i[valid], idx_j[valid], valid


def _compute_bootstrap_stability(
    patch: Patch,
    patch_lines: list,
    full_candidate: VanishingPointCandidate,
    patch_center: tuple[float, float],
    image_diagonal: float,
    config: RegionalConfig,
) -> tuple[float, dict[str, float]]:
    if config.bootstrap_rounds <= 0 or len(patch_lines) < config.min_lines_per_patch:
        return (
            0.0,
            {
                "bootstrap_rounds": 0.0,
                "bootstrap_success_rate": 0.0,
            },
        )

    sample_size = max(
        config.min_lines_per_patch,
        int(math.ceil(config.bootstrap_sample_ratio * len(patch_lines))),
    )
    sample_size = min(sample_size, len(patch_lines))
    rng = np.random.default_rng(_stable_patch_seed(patch))
    match_radius_px = max(config.bootstrap_match_radius_ratio * image_diagonal, 1.0)

    # Pre-extract arrays and pre-compute the FULL intersection matrix once
    full_hom, full_angles, full_lengths = _extract_line_arrays(patch_lines)
    full_xy, full_pair_i, full_pair_j, _ = _precompute_intersection_matrix(
        full_hom, full_angles, patch_center, image_diagonal,
        config.min_intersection_line_angle_rad,
        config.max_vp_distance_factor,
    )

    # Pre-compute midpoints for vectorized VP scoring
    midpoints = np.array(
        [((s.x1 + s.x2) * 0.5, (s.y1 + s.y2) * 0.5) for s in patch_lines],
        dtype=np.float64,
    )

    cluster_radius_px = max(config.intersection_dedup_radius_ratio * image_diagonal, 1.0)

    round_scores: list[float] = []
    successful_rounds = 0
    for _round_index in range(config.bootstrap_rounds):
        if sample_size == len(patch_lines):
            # Use all intersections directly
            boot_xy = full_xy
            boot_lines = patch_lines
            boot_midpoints = midpoints
            boot_angles = full_angles
            boot_lengths = full_lengths
        else:
            sample_indices = sorted(
                int(index)
                for index in rng.choice(len(patch_lines), size=sample_size, replace=False).tolist()
            )
            # Vectorized filter: keep only intersections where both lines are in the sample
            if len(full_pair_i) > 0:
                in_sample = np.zeros(len(patch_lines), dtype=bool)
                in_sample[sample_indices] = True
                pair_mask = in_sample[full_pair_i] & in_sample[full_pair_j]
            else:
                pair_mask = np.empty(0, dtype=bool)
            boot_xy = full_xy[pair_mask] if pair_mask.any() else np.empty((0, 2), dtype=np.float64)
            boot_lines = [patch_lines[i] for i in sample_indices]
            idx_arr = np.array(sample_indices, dtype=np.intp)
            boot_midpoints = midpoints[idx_arr]
            boot_angles = full_angles[idx_arr]
            boot_lengths = full_lengths[idx_arr]

        if boot_xy.shape[0] == 0:
            round_scores.append(0.0)
            continue

        clusters = _cluster_intersections(boot_xy, radius_px=cluster_radius_px)
        if not clusters:
            round_scores.append(0.0)
            continue

        cluster_xy = np.array([[c.x, c.y] for c in clusters], dtype=np.float64)
        scores_inliers = _vectorized_score_vp_candidates(
            cluster_xy, boot_midpoints, boot_angles, boot_lengths,
            config.angular_inlier_threshold_rad,
        )

        # Filter and pick best
        best_score = -1.0
        best_candidate = None
        for idx, cluster in enumerate(clusters):
            score, num_inliers = scores_inliers[idx]
            if num_inliers < config.min_candidate_inliers:
                continue
            if score > best_score:
                best_score = score
                best_candidate = VanishingPointCandidate(
                    x=cluster.x, y=cluster.y, score=score, num_inliers=num_inliers,
                )

        if best_candidate is None:
            round_scores.append(0.0)
            continue

        successful_rounds += 1
        distance = math.hypot(best_candidate.x - full_candidate.x, best_candidate.y - full_candidate.y)
        position_consistency = math.exp(-((distance / match_radius_px) ** 2))
        support_ratio = min(1.0, best_candidate.score / max(full_candidate.score, 1e-6))
        inlier_ratio = min(1.0, best_candidate.num_inliers / max(full_candidate.num_inliers, 1))
        round_scores.append(
            float((0.5 * position_consistency) + (0.25 * support_ratio) + (0.25 * inlier_ratio))
        )

    stability_score = float(sum(round_scores) / len(round_scores)) if round_scores else 0.0
    return (
        stability_score,
        {
            "bootstrap_rounds": float(config.bootstrap_rounds),
            "bootstrap_success_rate": float(successful_rounds / max(config.bootstrap_rounds, 1)),
            "bootstrap_sample_size": float(sample_size),
            "bootstrap_match_radius_px": float(match_radius_px),
        },
    )


def estimate_regional_hypotheses(
    line_set: LineSet,
    patches: Iterable[Patch],
    config: RegionalConfig,
) -> list[RegionalHypothesis]:
    """Estimate baseline VP hypotheses independently for each patch."""

    image_diagonal = math.hypot(line_set.image_width, line_set.image_height)
    cluster_radius_px = max(config.intersection_dedup_radius_ratio * image_diagonal, 1.0)
    hypotheses: list[RegionalHypothesis] = []

    for patch in patches:
        patch_lines = _select_patch_lines(
            line_set.segments,
            patch=patch,
            min_overlap_ratio=config.patch_line_overlap_ratio,
        )
        orientation_histogram = _compute_orientation_histogram(
            patch_lines,
            bins=config.orientation_histogram_bins,
        )
        if len(patch_lines) < config.min_lines_per_patch:
            hypothesis = _empty_hypothesis(patch, len(patch_lines))
            hypothesis.metadata["orientation_histogram"] = orientation_histogram
            hypotheses.append(hypothesis)
            continue

        # Pre-extract NumPy arrays for this patch's lines (used by all vectorized ops)
        line_arrays = _extract_line_arrays(patch_lines)

        patch_center = patch.center
        top_candidates, candidate_metadata = _estimate_patch_candidates(
            patch_lines=patch_lines,
            patch_center=patch_center,
            image_diagonal=image_diagonal,
            config=config,
            _line_arrays=line_arrays,
        )

        if not top_candidates:
            hypotheses.append(
                RegionalHypothesis(
                    patch=patch,
                    vp_candidates=[],
                    support_score=0.0,
                    stability_score=0.0,
                    num_lines=len(patch_lines),
                    metadata={
                        "viable": False,
                        **candidate_metadata,
                        "orientation_histogram": orientation_histogram,
                    },
                )
            )
            continue

        bootstrap_stability, bootstrap_metadata = _compute_bootstrap_stability(
            patch=patch,
            patch_lines=patch_lines,
            full_candidate=top_candidates[0],
            patch_center=patch_center,
            image_diagonal=image_diagonal,
            config=config,
        )

        top_score = top_candidates[0].score
        support_margin = top_score - (top_candidates[1].score if len(top_candidates) > 1 else 0.0)
        stability_score = max(0.0, min(1.0, bootstrap_stability))

        hypotheses.append(
            RegionalHypothesis(
                patch=patch,
                vp_candidates=top_candidates,
                support_score=max(0.0, min(1.0, top_score)),
                stability_score=stability_score,
                num_lines=len(patch_lines),
                metadata={
                    "viable": True,
                    **candidate_metadata,
                    "orientation_histogram": orientation_histogram,
                    "orientation_histogram_bins": config.orientation_histogram_bins,
                    "dominant_vp_index": 0,
                    "bootstrap_stability": bootstrap_stability,
                    "support_margin": support_margin,
                    **bootstrap_metadata,
                },
            )
        )

    return hypotheses

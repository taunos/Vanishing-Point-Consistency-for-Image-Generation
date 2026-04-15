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


def _cluster_intersections(
    intersections: list[tuple[float, float]],
    radius_px: float,
) -> list[_Cluster]:
    """Cluster intersections via deterministic connected components."""

    if not intersections:
        return []

    ordered_points = sorted(intersections)
    adjacency: list[list[int]] = [[] for _ in ordered_points]
    for left_index in range(len(ordered_points)):
        for right_index in range(left_index + 1, len(ordered_points)):
            if math.hypot(
                ordered_points[left_index][0] - ordered_points[right_index][0],
                ordered_points[left_index][1] - ordered_points[right_index][1],
            ) <= radius_px:
                adjacency[left_index].append(right_index)
                adjacency[right_index].append(left_index)

    visited = [False] * len(ordered_points)
    clusters: list[_Cluster] = []
    for start_index in range(len(ordered_points)):
        if visited[start_index]:
            continue
        stack = [start_index]
        component_indices: list[int] = []
        visited[start_index] = True
        while stack:
            current_index = stack.pop()
            component_indices.append(current_index)
            for neighbor_index in adjacency[current_index]:
                if visited[neighbor_index]:
                    continue
                visited[neighbor_index] = True
                stack.append(neighbor_index)

        component_points = [ordered_points[index] for index in sorted(component_indices)]
        clusters.append(
            _Cluster(
                x=float(sum(point[0] for point in component_points) / len(component_points)),
                y=float(sum(point[1] for point in component_points) / len(component_points)),
                count=len(component_points),
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
) -> list[tuple[float, float]]:
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


def _estimate_patch_candidates(
    patch_lines: list,
    patch_center: tuple[float, float],
    image_diagonal: float,
    config: RegionalConfig,
) -> tuple[list[VanishingPointCandidate], dict[str, int]]:
    raw_intersections = _generate_pairwise_intersections(
        patch_lines=patch_lines,
        patch_center=patch_center,
        image_diagonal=image_diagonal,
        config=config,
    )
    if not raw_intersections:
        return ([], {"num_raw_intersections": 0, "num_clustered_intersections": 0})

    cluster_radius_px = max(config.intersection_dedup_radius_ratio * image_diagonal, 1.0)
    clusters = _cluster_intersections(raw_intersections, radius_px=cluster_radius_px)
    scored_candidates: list[tuple[VanishingPointCandidate, int]] = []
    for cluster in clusters:
        score, num_inliers = score_vp_candidate(
            cluster.x,
            cluster.y,
            patch_lines,
            angular_inlier_threshold_rad=config.angular_inlier_threshold_rad,
        )
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

    round_scores: list[float] = []
    successful_rounds = 0
    for _round_index in range(config.bootstrap_rounds):
        if sample_size == len(patch_lines):
            sample_indices = list(range(len(patch_lines)))
        else:
            sample_indices = sorted(
                int(index)
                for index in rng.choice(len(patch_lines), size=sample_size, replace=False).tolist()
            )
        sampled_lines = [patch_lines[index] for index in sample_indices]
        boot_candidates, _boot_metadata = _estimate_patch_candidates(
            patch_lines=sampled_lines,
            patch_center=patch_center,
            image_diagonal=image_diagonal,
            config=config,
        )
        if not boot_candidates:
            round_scores.append(0.0)
            continue

        successful_rounds += 1
        best_boot = boot_candidates[0]
        distance = math.hypot(best_boot.x - full_candidate.x, best_boot.y - full_candidate.y)
        position_consistency = math.exp(-((distance / match_radius_px) ** 2))
        support_ratio = min(1.0, best_boot.score / max(full_candidate.score, 1e-6))
        inlier_ratio = min(1.0, best_boot.num_inliers / max(full_candidate.num_inliers, 1))
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

        patch_center = patch.center
        top_candidates, candidate_metadata = _estimate_patch_candidates(
            patch_lines=patch_lines,
            patch_center=patch_center,
            image_diagonal=image_diagonal,
            config=config,
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

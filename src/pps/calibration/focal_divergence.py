"""Estimate focal length from VP configurations in different image regions.

Measure divergence as a camera-intrinsic consistency signal.

Theory: Given two orthogonal VPs v1, v2 and assuming principal point at
image center (cx, cy), the focal length f satisfies:
    f^2 = -(v1x - cx)(v2x - cx) - (v1y - cy)(v2y - cy)

If different image regions yield different f values, the image does not
admit a single coherent camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pcs.detectors import create_detector
from pcs.geometry.lines import build_line_segment, filter_short_segments
from pcs.geometry.types import LineSegment


@dataclass
class FocalDivergenceResult:
    """Focal length divergence analysis for one image."""

    region_focal_lengths: list[float]
    focal_mean: float
    focal_std: float
    focal_cv: float  # coefficient of variation (std/mean)
    focal_range: float
    num_valid_regions: int
    num_total_regions: int


def estimate_focal_divergence(
    image: np.ndarray,
    grid_size: int = 2,
    detector_name: str = "opencv_lsd",
    min_lines_per_region: int = 8,
) -> FocalDivergenceResult:
    """Estimate focal length independently in each image region.

    1. Divide image into grid_size x grid_size regions
    2. In each region, detect lines and estimate VPs
    3. From orthogonal VP pairs, compute focal length
    4. Report divergence statistics
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Detect lines on the full image
    detector = create_detector(detector_name)
    line_set = detector.detect(image)
    segments = line_set.segments

    rh, rw = h // grid_size, w // grid_size
    num_total = grid_size * grid_size
    focal_lengths = []

    for r in range(grid_size):
        for c in range(grid_size):
            x0, y0 = c * rw, r * rh
            x1, y1 = (c + 1) * rw, (r + 1) * rh

            # Select lines within this region
            region_lines = _select_lines_in_rect(segments, x0, y0, x1, y1)
            if len(region_lines) < min_lines_per_region:
                continue

            # Estimate VPs in this region
            vps = _estimate_vps_from_lines(region_lines, (x0 + x1) / 2, (y0 + y1) / 2, max(h, w))
            if len(vps) < 2:
                continue

            # Try all pairs of VPs, compute focal length from orthogonality
            for i in range(len(vps)):
                for j in range(i + 1, len(vps)):
                    f_sq = _focal_from_orthogonal_vps(vps[i], vps[j], cx, cy)
                    if f_sq is not None and f_sq > 0:
                        focal_lengths.append(math.sqrt(f_sq))

    if not focal_lengths:
        return FocalDivergenceResult(
            region_focal_lengths=[],
            focal_mean=0.0,
            focal_std=0.0,
            focal_cv=0.0,
            focal_range=0.0,
            num_valid_regions=0,
            num_total_regions=num_total,
        )

    fl = np.array(focal_lengths)
    mean = float(np.mean(fl))
    std = float(np.std(fl))

    return FocalDivergenceResult(
        region_focal_lengths=focal_lengths,
        focal_mean=mean,
        focal_std=std,
        focal_cv=std / mean if mean > 0 else 0.0,
        focal_range=float(np.max(fl) - np.min(fl)),
        num_valid_regions=len(focal_lengths),
        num_total_regions=num_total,
    )


def _focal_from_orthogonal_vps(
    vp1: tuple[float, float],
    vp2: tuple[float, float],
    cx: float,
    cy: float,
) -> float | None:
    """Compute f^2 from two assumed-orthogonal VPs and principal point."""
    f_sq = -(vp1[0] - cx) * (vp2[0] - cx) - (vp1[1] - cy) * (vp2[1] - cy)
    if f_sq <= 0:
        return None
    return f_sq


def _select_lines_in_rect(
    segments: list[LineSegment],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    margin: float = 0.3,
) -> list[LineSegment]:
    """Select segments whose midpoints fall within the rectangle."""
    result = []
    for seg in segments:
        mx = (seg.x1 + seg.x2) / 2
        my = (seg.y1 + seg.y2) / 2
        if x0 <= mx <= x1 and y0 <= my <= y1:
            result.append(seg)
    return result


def _estimate_vps_from_lines(
    lines: list[LineSegment],
    region_cx: float,
    region_cy: float,
    image_size: float,
    min_angle_diff_rad: float = math.radians(8.0),
    cluster_radius_ratio: float = 0.03,
    max_vp_distance_factor: float = 6.0,
    angular_inlier_threshold_rad: float = math.radians(7.5),
    min_inliers: int = 3,
    top_k: int = 3,
) -> list[tuple[float, float]]:
    """Estimate VP candidates from a set of line segments.

    Uses pairwise intersection + spatial clustering + angular scoring.
    Self-contained version extracted from the archived regional.hypotheses module.
    """
    n = len(lines)
    if n < 2:
        return []

    # Build homogeneous line representations
    hom = np.empty((n, 3), dtype=np.float64)
    angles = np.empty(n, dtype=np.float64)
    for i, seg in enumerate(lines):
        p1 = np.array([seg.x1, seg.y1, 1.0])
        p2 = np.array([seg.x2, seg.y2, 1.0])
        line = np.cross(p1, p2)
        norm = math.hypot(line[0], line[1])
        if norm > 0:
            line /= norm
        hom[i] = line
        angles[i] = seg.angle_rad

    # Pairwise intersections
    idx_i, idx_j = np.triu_indices(n, k=1)
    a_i, a_j = angles[idx_i], angles[idx_j]
    diff = np.abs(a_i - a_j) % math.pi
    diff = np.minimum(diff, math.pi - diff)
    mask = diff >= min_angle_diff_rad
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    if len(idx_i) == 0:
        return []

    pts = np.cross(hom[idx_i], hom[idx_j])
    w = pts[:, 2]
    ok = np.abs(w) > 1e-9
    pts, w = pts[ok], w[ok]
    xy = pts[:, :2] / w[:, np.newaxis]
    ok2 = np.all(np.isfinite(xy), axis=1)
    xy = xy[ok2]

    if len(xy) == 0:
        return []

    # Distance filter
    max_dist = max_vp_distance_factor * image_size
    dist_sq = (xy[:, 0] - region_cx) ** 2 + (xy[:, 1] - region_cy) ** 2
    xy = xy[dist_sq <= max_dist ** 2]

    if len(xy) == 0:
        return []

    # Cluster intersections
    from scipy.spatial import cKDTree

    radius = max(cluster_radius_ratio * image_size, 1.0)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=radius, output_type="ndarray")

    if len(pairs) == 0:
        # Each point is its own cluster — pick by density
        clusters = [(float(xy[i, 0]), float(xy[i, 1]), 1) for i in range(min(len(xy), 50))]
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        nn = len(xy)
        row, col = pairs[:, 0], pairs[:, 1]
        data = np.ones(len(row), dtype=np.int8)
        adj = coo_matrix(
            (np.concatenate([data, data]), (np.concatenate([row, col]), np.concatenate([col, row]))),
            shape=(nn, nn),
        )
        n_comp, labels = connected_components(adj, directed=False)
        clusters = []
        for comp_id in range(n_comp):
            comp_mask = labels == comp_id
            comp_pts = xy[comp_mask]
            clusters.append(
                (float(comp_pts[:, 0].mean()), float(comp_pts[:, 1].mean()), int(comp_mask.sum()))
            )

    # Score clusters
    midpoints = np.array([((s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2) for s in lines], dtype=np.float64)
    lengths = np.array([s.length for s in lines], dtype=np.float64)

    scored = []
    for cx_, cy_, count in clusters:
        rays = np.array([cx_, cy_]) - midpoints
        ray_angles = np.arctan2(rays[:, 1], rays[:, 0]) % math.pi
        residuals = np.abs(ray_angles - angles) % math.pi
        residuals = np.minimum(residuals, math.pi - residuals)
        inlier_mask = residuals <= angular_inlier_threshold_rad
        n_inliers = int(inlier_mask.sum())
        if n_inliers < min_inliers:
            continue
        weights = np.maximum(lengths, 1.0)
        agreement = np.clip(1.0 - residuals / angular_inlier_threshold_rad, 0.0, 1.0)
        score = float((weights * agreement * inlier_mask).sum() / weights.sum())
        scored.append((score, n_inliers, count, cx_, cy_))

    scored.sort(reverse=True)
    return [(x, y) for _, _, _, x, y in scored[:top_k]]

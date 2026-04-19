"""Compute consistency metrics from Perspective Fields output.

These measure whether the field is compatible with a single camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps.fields.perspective_wrapper import PerspectiveFieldResult


@dataclass
class FieldConsistencyResult:
    """All consistency metrics for one image."""

    # Latitude field metrics
    latitude_std: float
    gradient_x_std: float  # KEY METRIC (p=0.005 in feasibility)
    gradient_y_std: float
    gradient_y_mean: float
    patch_mean_range: float

    # Up-vector metrics
    up_angle_mean: float
    up_angle_max: float
    up_angle_std: float

    # Camera parameter metrics (if Paramnet available)
    focal_length: float | None

    # Combined
    field_consistency_score: float


def compute_field_consistency(
    result: PerspectiveFieldResult,
    grid_size: int = 4,
) -> FieldConsistencyResult:
    """Compute all consistency metrics from a PerspectiveFieldResult.

    Args:
        result: Output from PerspectiveFieldsWrapper.predict().
        grid_size: Patch grid for cross-region comparison (default 4x4 = 16 patches).
    """
    lat_metrics = _compute_latitude_consistency(result.latitude, grid_size)
    up_metrics = _compute_up_consistency(result.gravity, grid_size)
    score = compute_combined_score(lat_metrics, up_metrics)

    return FieldConsistencyResult(
        latitude_std=lat_metrics["latitude_std"],
        gradient_x_std=lat_metrics["gradient_x_std"],
        gradient_y_std=lat_metrics["gradient_y_std"],
        gradient_y_mean=lat_metrics["gradient_y_mean"],
        patch_mean_range=lat_metrics["patch_mean_range"],
        up_angle_mean=up_metrics["up_angle_mean"],
        up_angle_max=up_metrics["up_angle_max"],
        up_angle_std=up_metrics["up_angle_std"],
        focal_length=result.focal_length,
        field_consistency_score=score,
    )


def compute_combined_score(
    lat_metrics: dict[str, float],
    up_metrics: dict[str, float],
) -> float:
    """Combine metrics into a single consistency score in [0, 1].

    Uses sigmoid-based normalization calibrated from the feasibility test:
    - gradient_x_std: real mean=0.006, gen mean=0.020 -> midpoint ~0.012
    - latitude_std: real mean=12.3, gen mean=17.7 -> midpoint ~15.0
    - patch_mean_range: real mean=33.7, gen mean=49.9 -> midpoint ~42.0
    - up_angle_mean: catches extremes, midpoint ~0.1 rad
    """
    gx = _sigmoid_score(lat_metrics["gradient_x_std"], midpoint=0.012, steepness=200.0)
    ls = _sigmoid_score(lat_metrics["latitude_std"], midpoint=15.0, steepness=0.3)
    pr = _sigmoid_score(lat_metrics["patch_mean_range"], midpoint=42.0, steepness=0.08)
    ua = _sigmoid_score(up_metrics["up_angle_mean"], midpoint=0.1, steepness=15.0)

    # Weighted average: gradient_x_std is primary signal
    return float(0.40 * gx + 0.25 * ls + 0.20 * pr + 0.15 * ua)


def _sigmoid_score(value: float, midpoint: float, steepness: float) -> float:
    """Map a metric value to [0, 1] where 0=consistent, 1=inconsistent.

    Returns 1 - sigmoid so that lower metric values -> higher scores (more consistent).
    """
    x = steepness * (value - midpoint)
    x = max(-50.0, min(50.0, x))  # clamp to avoid overflow
    return 1.0 / (1.0 + np.exp(x))


def _compute_latitude_consistency(
    latitude: np.ndarray, grid_size: int
) -> dict[str, float]:
    H, W = latitude.shape
    ph, pw = H // grid_size, W // grid_size

    patch_stats = []
    for r in range(grid_size):
        for c in range(grid_size):
            patch = latitude[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            patch_stats.append(
                {
                    "mean": float(np.mean(patch)),
                    "grad_y": float(np.mean(np.diff(patch, axis=0))),
                    "grad_x": float(np.mean(np.diff(patch, axis=1))),
                }
            )

    grad_ys = [p["grad_y"] for p in patch_stats]
    grad_xs = [p["grad_x"] for p in patch_stats]
    means = [p["mean"] for p in patch_stats]

    return {
        "latitude_std": float(np.std(latitude)),
        "gradient_y_std": float(np.std(grad_ys)),
        "gradient_x_std": float(np.std(grad_xs)),
        "gradient_y_mean": float(np.mean(grad_ys)),
        "patch_mean_range": float(max(means) - min(means)),
    }


def _compute_up_consistency(
    gravity: np.ndarray, grid_size: int
) -> dict[str, float]:
    """Compute pairwise angular disagreement between patch mean gravity vectors.

    Args:
        gravity: (H, W, 2) per-pixel gravity direction.
    """
    H, W = gravity.shape[:2]
    ph, pw = H // grid_size, W // grid_size

    mean_ups = []
    for r in range(grid_size):
        for c in range(grid_size):
            patch = gravity[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            mu = np.mean(patch.reshape(-1, patch.shape[-1]), axis=0)
            norm = np.linalg.norm(mu)
            if norm > 1e-8:
                mu = mu / norm
            mean_ups.append(mu)

    angles = []
    for i in range(len(mean_ups)):
        for j in range(i + 1, len(mean_ups)):
            cos_a = np.clip(np.dot(mean_ups[i], mean_ups[j]), -1.0, 1.0)
            angles.append(float(np.arccos(cos_a)))

    if not angles:
        return {"up_angle_mean": 0.0, "up_angle_max": 0.0, "up_angle_std": 0.0}

    return {
        "up_angle_mean": float(np.mean(angles)),
        "up_angle_max": float(np.max(angles)),
        "up_angle_std": float(np.std(angles)),
    }


def compute_batch(
    results: list[PerspectiveFieldResult],
    names: list[str],
    grid_size: int = 4,
) -> list[dict]:
    """Process a list of results and return a list of dicts (DataFrame-ready)."""
    rows = []
    for name, result in zip(names, results):
        fc = compute_field_consistency(result, grid_size)
        row = {
            "name": name,
            "latitude_std": fc.latitude_std,
            "gradient_x_std": fc.gradient_x_std,
            "gradient_y_std": fc.gradient_y_std,
            "gradient_y_mean": fc.gradient_y_mean,
            "patch_mean_range": fc.patch_mean_range,
            "up_angle_mean": fc.up_angle_mean,
            "up_angle_max": fc.up_angle_max,
            "up_angle_std": fc.up_angle_std,
            "focal_length": fc.focal_length,
            "field_consistency_score": fc.field_consistency_score,
        }
        rows.append(row)
    return rows

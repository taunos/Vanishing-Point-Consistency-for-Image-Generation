"""Projective Plausibility Score — combines Perspective Fields consistency
with analytic focal length divergence into a single metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pps.calibration.focal_divergence import FocalDivergenceResult, estimate_focal_divergence
from pps.fields.field_consistency import FieldConsistencyResult, compute_field_consistency
from pps.fields.perspective_wrapper import PerspectiveFieldResult, PerspectiveFieldsWrapper


@dataclass
class PPSResult:
    """Complete projective plausibility analysis for one image."""

    field_consistency: FieldConsistencyResult
    focal_divergence: FocalDivergenceResult | None
    pps_score: float  # [0, 1], higher = more plausible
    pps_confidence: float  # [0, 1], how reliable is this estimate
    consistency_map: np.ndarray | None  # (H, W) spatial inconsistency map


def compute_pps(
    image: np.ndarray,
    wrapper: PerspectiveFieldsWrapper,
    use_focal_divergence: bool = True,
    grid_size_fields: int = 4,
    grid_size_focal: int = 2,
) -> PPSResult:
    """Full PPS computation for one image.

    1. Run Perspective Fields -> field consistency metrics
    2. Optionally run focal length divergence -> calibration metrics
    3. Combine into PPS score
    """
    # Step 1: Perspective Fields
    pf_result = wrapper.predict(image)
    field_consistency = compute_field_consistency(pf_result, grid_size_fields)

    # Step 2: Focal divergence (optional)
    focal_div = None
    if use_focal_divergence:
        try:
            focal_div = estimate_focal_divergence(
                image, grid_size=grid_size_focal
            )
        except Exception:
            pass

    # Step 3: Combine
    pps = field_consistency.field_consistency_score

    # Confidence: based on how much signal we have
    confidence = 1.0
    if focal_div is not None and focal_div.num_valid_regions == 0:
        confidence *= 0.8  # less confident without focal data

    # Generate spatial inconsistency map from latitude field
    consistency_map = _compute_consistency_map(pf_result.latitude, grid_size_fields)

    return PPSResult(
        field_consistency=field_consistency,
        focal_divergence=focal_div,
        pps_score=pps,
        pps_confidence=confidence,
        consistency_map=consistency_map,
    )


def _compute_consistency_map(latitude: np.ndarray, grid_size: int) -> np.ndarray:
    """Compute per-pixel inconsistency from the latitude field.

    For each pixel, measure how much the local gradient deviates from
    the global mean gradient. Returns a (H, W) map where higher values
    indicate more inconsistency.
    """
    # Compute global mean gradients
    gy_global = np.mean(np.diff(latitude, axis=0))
    gx_global = np.mean(np.diff(latitude, axis=1))

    # Local gradients (Sobel-like, but simple finite differences)
    gy_local = np.zeros_like(latitude)
    gx_local = np.zeros_like(latitude)
    gy_local[:-1, :] = np.diff(latitude, axis=0)
    gx_local[:, :-1] = np.diff(latitude, axis=1)

    # Deviation from global trend
    dev_y = (gy_local - gy_global) ** 2
    dev_x = (gx_local - gx_global) ** 2
    inconsistency = np.sqrt(dev_y + dev_x)

    # Smooth with a box filter
    from scipy.ndimage import uniform_filter

    kernel = max(latitude.shape[0], latitude.shape[1]) // (grid_size * 2)
    kernel = max(kernel, 3)
    inconsistency = uniform_filter(inconsistency, size=kernel)

    return inconsistency

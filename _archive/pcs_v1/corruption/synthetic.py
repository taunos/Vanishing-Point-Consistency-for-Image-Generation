"""Synthetic geometric corruptions for PCS sanity checks.

Each corruption degrades projective consistency in a controlled way so that
PCS can be validated independently of real-vs-generated comparisons.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class CorruptionType(Enum):
    PATCH_SHUFFLE = "patch_shuffle"
    HORIZON_TILT = "horizon_tilt"
    LOCAL_PERSPECTIVE_WARP = "local_perspective_warp"
    VP_DRIFT = "vp_drift"


@dataclass
class CorruptionConfig:
    corruption_type: CorruptionType
    severity: float  # 0.0 = none, 1.0 = maximum

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _perspective_transform(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Apply a perspective warp using PIL (avoids hard OpenCV dependency)."""
    h, w = image.shape[:2]
    coeffs = _find_perspective_coeffs(dst, src)
    pil_img = Image.fromarray(image)
    warped = pil_img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
    return np.asarray(warped)


def _find_perspective_coeffs(
    src_points: np.ndarray, dst_points: np.ndarray
) -> Sequence[float]:
    """Compute 8 perspective transform coefficients from 4 point pairs.

    Solves for coefficients (a, b, c, d, e, f, g, h) such that:
        x' = (a*x + b*y + c) / (g*x + h*y + 1)
        y' = (d*x + e*y + f) / (g*x + h*y + 1)
    """
    matrix: list[list[float]] = []
    for s, d in zip(src_points, dst_points):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0] * d[0], -s[0] * d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1] * d[0], -s[1] * d[1]])
    A = np.array(matrix, dtype=np.float64)
    B = np.array([coord for pair in src_points for coord in pair], dtype=np.float64)
    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return tuple(float(v) for v in res)


# ---------------------------------------------------------------------------
# Corruption implementations
# ---------------------------------------------------------------------------

def _patch_shuffle(image: np.ndarray, severity: float, rng: np.random.RandomState) -> np.ndarray:
    """Divide into grid, randomly permute patches, then apply alternating vertical shifts.

    The vertical shifts move content up or down within each patch in a checkerboard
    pattern. This displaces VP positions in opposite directions for adjacent patches,
    creating cross-patch VP position divergence detectable by the PCS metric.
    Vertical shifts produce only horizontal black bars (not diagonal edges), so they
    do not generate spurious diagonal line evidence that would confuse LSD.
    """
    if severity < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    # Grid size scales with severity: 2x2 at low, up to 4x4 at max
    grid_n = max(2, int(round(2 + 2 * severity)))
    cell_h, cell_w = h // grid_n, w // grid_n
    if cell_h < 8 or cell_w < 8:
        return image.copy()

    # Extract patches
    patches: list[tuple[int, int, np.ndarray]] = []
    for r in range(grid_n):
        for c in range(grid_n):
            y0, x0 = r * cell_h, c * cell_w
            y1 = y0 + cell_h if r < grid_n - 1 else h
            x1 = x0 + cell_w if c < grid_n - 1 else w
            patches.append((r, c, image[y0:y1, x0:x1].copy()))

    # Determine how many swaps based on severity
    total_patches = len(patches)
    num_swaps = max(1, int(round(severity * total_patches)))
    indices = list(range(total_patches))
    for _ in range(num_swaps):
        i, j = rng.choice(total_patches, size=2, replace=False)
        indices[i], indices[j] = indices[j], indices[i]

    # Reassemble (resize patches to target cell size to handle edge cells)
    result = image.copy()
    for target_idx, source_idx in enumerate(indices):
        tr, tc = target_idx // grid_n, target_idx % grid_n
        _, _, src_patch = patches[source_idx]

        ty0, tx0 = tr * cell_h, tc * cell_w
        ty1 = ty0 + cell_h if tr < grid_n - 1 else h
        tx1 = tx0 + cell_w if tc < grid_n - 1 else w
        target_h, target_w = ty1 - ty0, tx1 - tx0

        if src_patch.shape[0] != target_h or src_patch.shape[1] != target_w:
            pil_patch = Image.fromarray(src_patch)
            pil_patch = pil_patch.resize((target_w, target_h), Image.Resampling.BILINEAR)
            src_patch = np.asarray(pil_patch)

        result[ty0:ty1, tx0:tx1] = src_patch

    # Apply alternating vertical shifts to create VP position divergence.
    # Even patches (tr+tc even): shift content DOWN → VP shifts down.
    # Odd patches (tr+tc odd): shift content UP → VP shifts up.
    # Adjacent patches then have VPs displaced in opposite directions.
    max_dy_frac = 0.35  # 35% of cell height at severity=1
    for target_idx in range(total_patches):
        tr, tc = target_idx // grid_n, target_idx % grid_n
        ty0, tx0 = tr * cell_h, tc * cell_w
        ty1 = ty0 + cell_h if tr < grid_n - 1 else h
        tx1 = tx0 + cell_w if tc < grid_n - 1 else w
        ph_p = ty1 - ty0

        dy = int(round(severity * max_dy_frac * ph_p))
        if dy == 0:
            continue

        patch = result[ty0:ty1, tx0:tx1].copy()
        shifted = np.zeros_like(patch)
        if (tr + tc) % 2 == 0:
            shifted[dy:, :] = patch[: ph_p - dy, :]  # shift DOWN
        else:
            shifted[: ph_p - dy, :] = patch[dy:, :]  # shift UP
        result[ty0:ty1, tx0:tx1] = shifted

    return result


def _horizon_tilt(image: np.ndarray, severity: float, rng: np.random.RandomState) -> np.ndarray:
    """Apply projective transform that tilts the implied horizon."""
    if severity < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    # Max tilt angle: 15 degrees at severity 1.0
    angle_deg = severity * 15.0
    # Randomly pick direction
    if rng.rand() > 0.5:
        angle_deg = -angle_deg
    angle_rad = math.radians(angle_deg)

    # Compute vertical displacement at left/right edges
    dy = (w / 2.0) * math.tan(angle_rad)

    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    dst = np.array([
        [0, -dy],
        [w, dy],
        [w, h + dy],
        [0, h - dy],
    ], dtype=np.float64)

    return _perspective_transform(image, src, dst)


def _local_perspective_warp(
    image: np.ndarray, severity: float, rng: np.random.RandomState
) -> np.ndarray:
    """Create incompatible projective interpretations in left and right image halves.

    Shifts the left half UP and the right half DOWN by the same amount. This
    displaces VP positions in opposite vertical directions between halves, creating
    the kind of cross-region inconsistency that real single-camera images never have.
    Pure numpy translation: no perspective warp, no diagonal boundaries, so LSD
    does not produce spurious diagonal lines that would mask the VP divergence signal.
    """
    if severity < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    mid = w // 2
    # 35% of image height at full severity creates a VP shift of ~0.21 image diagonals
    # between halves — large enough to detect cleanly even with noisy VP estimation.
    dy = int(round(severity * h * 0.35))
    if dy == 0:
        return image.copy()

    result = image.copy()

    # Left half: shift content UP by dy (VP shifts up)
    shifted_left = np.zeros((h, mid, image.shape[2]), dtype=image.dtype)
    shifted_left[: h - dy, :] = image[dy:, :mid]
    result[:, :mid] = shifted_left

    # Right half: shift content DOWN by dy (VP shifts down)
    shifted_right = np.zeros((h, w - mid, image.shape[2]), dtype=image.dtype)
    shifted_right[dy:, :] = image[: h - dy, mid:]
    result[:, mid:] = shifted_right

    return result


def _vp_drift(image: np.ndarray, severity: float, rng: np.random.RandomState) -> np.ndarray:
    """Apply spatially-varying homography simulating VP direction drift."""
    if severity < 1e-6:
        return image.copy()

    h, w = image.shape[:2]
    # Split into vertical strips and apply progressively stronger warps
    num_strips = 4
    strip_w = w // num_strips
    if strip_w < 8:
        return image.copy()

    # 0.30h at full severity ensures VP positions diverge detectably across strips.
    max_shift = severity * h * 0.30
    result = image.copy()

    for i in range(num_strips):
        x0 = i * strip_w
        x1 = x0 + strip_w if i < num_strips - 1 else w
        sw = x1 - x0
        strip = image[:, x0:x1].copy()

        # Progressive vertical shift: increases left to right
        progress = (i + 0.5) / num_strips
        drift_y = max_shift * (2.0 * progress - 1.0)
        drift_y += rng.uniform(-max_shift * 0.1, max_shift * 0.1)

        src = np.array([[0, 0], [sw, 0], [sw, h], [0, h]], dtype=np.float64)
        dst = np.array([
            [0, drift_y * 0.5],
            [sw, drift_y],
            [sw, h + drift_y],
            [0, h + drift_y * 0.5],
        ], dtype=np.float64)

        warped = _perspective_transform(strip, src, dst)
        result[:, x0:x1] = warped

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CORRUPTION_FNS = {
    CorruptionType.PATCH_SHUFFLE: _patch_shuffle,
    CorruptionType.HORIZON_TILT: _horizon_tilt,
    CorruptionType.LOCAL_PERSPECTIVE_WARP: _local_perspective_warp,
    CorruptionType.VP_DRIFT: _vp_drift,
}


def apply_corruption(
    image: np.ndarray, config: CorruptionConfig, seed: int = 42
) -> np.ndarray:
    """Apply a geometric corruption to an image. Returns corrupted uint8 RGB."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")

    rng = np.random.RandomState(seed)
    fn = _CORRUPTION_FNS[config.corruption_type]
    result = fn(image, config.severity, rng)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_all_corruptions(
    image: np.ndarray,
    severities: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 42,
) -> dict[str, dict[float, np.ndarray]]:
    """Apply all corruption types at multiple severity levels.

    Returns ``{corruption_name: {severity: corrupted_image}}``.
    """
    results: dict[str, dict[float, np.ndarray]] = {}
    for ctype in CorruptionType:
        results[ctype.value] = {}
        for sev in severities:
            cfg = CorruptionConfig(corruption_type=ctype, severity=sev)
            results[ctype.value][sev] = apply_corruption(image, cfg, seed=seed)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a synthetic geometric corruption to an image."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument(
        "--type",
        required=True,
        choices=[ct.value for ct in CorruptionType],
        help="Corruption type.",
    )
    parser.add_argument(
        "--severity", type=float, default=0.5, help="Severity in [0, 1]."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", required=True, help="Output image path.")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.input).convert("RGB"))
    config = CorruptionConfig(
        corruption_type=CorruptionType(args.type), severity=args.severity
    )
    result = apply_corruption(image, config, seed=args.seed)
    Image.fromarray(result).save(args.output)
    print(f"Saved corrupted image to {args.output}")


if __name__ == "__main__":
    _cli()

"""Deterministic multi-scale overlapping patch generation."""

from __future__ import annotations

import math
from typing import Iterable

from pcs.geometry.types import Patch


def _normalize_scale(scale: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(scale, int):
        return (scale, scale)
    return scale


def generate_overlapping_grid_patches(
    image_width: int,
    image_height: int,
    scales: Iterable[int | tuple[int, int]],
    overlap_ratio: float,
    min_patch_size: int = 16,
) -> list[Patch]:
    """Generate deterministic overlapping grid patches over an image."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if overlap_ratio < 0.0:
        raise ValueError("overlap_ratio must be non-negative")

    overlap_pct = int(round(overlap_ratio * 100.0))
    patches: list[Patch] = []

    for scale_level, scale in enumerate(scales):
        rows, cols = _normalize_scale(scale)
        if rows <= 0 or cols <= 0:
            raise ValueError("scales must contain positive grid dimensions")

        cell_width = image_width / cols
        cell_height = image_height / rows
        pad_x = cell_width * overlap_ratio
        pad_y = cell_height * overlap_ratio

        for row in range(rows):
            for col in range(cols):
                base_x0 = col * cell_width
                base_y0 = row * cell_height
                base_x1 = (col + 1) * cell_width
                base_y1 = (row + 1) * cell_height

                x0 = max(0, int(math.floor(base_x0 - pad_x)))
                y0 = max(0, int(math.floor(base_y0 - pad_y)))
                x1 = min(image_width, int(math.ceil(base_x1 + pad_x)))
                y1 = min(image_height, int(math.ceil(base_y1 + pad_y)))

                if (x1 - x0) < min_patch_size or (y1 - y0) < min_patch_size:
                    continue

                patches.append(
                    Patch(
                        patch_id=f"s{rows}x{cols}_r{row:02d}_c{col:02d}_o{overlap_pct:02d}",
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        scale_level=scale_level,
                        overlap_tag=f"ov{overlap_pct:02d}",
                    )
                )

    return patches


"""Patch generation and patch-local geometric hypothesis estimation only.

This package intentionally stops at the patch boundary. Cross-patch
compatibility, region-graph reasoning, and global explanation logic belong in
`pcs.consensus`.
"""

from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.regional.patching import generate_overlapping_grid_patches

__all__ = ["estimate_regional_hypotheses", "generate_overlapping_grid_patches"]

"""Final score composition only.

This package may combine outputs from `pcs.regional`, `pcs.consensus`, and
other evaluator modules, but it should not contain the core consensus
algorithms themselves.
"""

from pcs.scoring.baseline import compute_baseline_pcs
from pcs.scoring.local_to_global import compute_local_to_global_pcs

__all__ = ["compute_baseline_pcs", "compute_local_to_global_pcs"]

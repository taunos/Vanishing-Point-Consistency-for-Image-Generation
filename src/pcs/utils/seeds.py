"""Deterministic seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_deterministic_seeds(seed: int) -> None:
    """Set deterministic seeds for supported libraries."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


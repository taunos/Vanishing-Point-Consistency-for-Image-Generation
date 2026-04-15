"""Abstract detector interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pcs.geometry.types import LineSet


class LineDetector(ABC):
    """Abstract line detector interface."""

    name: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> LineSet:
        """Detect line segments in an image."""


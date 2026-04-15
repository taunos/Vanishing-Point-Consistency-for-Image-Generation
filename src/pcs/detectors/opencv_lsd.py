"""OpenCV LSD detector adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from pcs.detectors.base import LineDetector
from pcs.detectors.registry import register_detector
from pcs.geometry.lines import build_line_segment, filter_short_segments
from pcs.geometry.types import LineSet

try:
    import cv2
except ImportError:  # pragma: no cover - exercised via runtime error path
    cv2 = None


def _require_opencv() -> Any:
    if cv2 is None or not hasattr(cv2, "createLineSegmentDetector"):
        raise RuntimeError(
            "OpenCV LSD is unavailable. Install the optional dependency with "
            "`python -m pip install -e .[opencv]`."
        )
    return cv2


class OpenCVLSDDetector(LineDetector):
    """Adapter around OpenCV's built-in LSD implementation."""

    name = "opencv_lsd"

    def __init__(
        self,
        min_line_length: float = 20.0,
        refine_mode: str = "standard",
    ) -> None:
        self.min_line_length = float(min_line_length)
        self.refine_mode = refine_mode.lower()

    def _create_detector(self) -> Any:
        cv = _require_opencv()
        refine_modes = {
            "none": getattr(cv, "LSD_REFINE_NONE", 0),
            "standard": getattr(cv, "LSD_REFINE_STD", 1),
            "advanced": getattr(cv, "LSD_REFINE_ADV", 2),
        }
        refine_value = refine_modes.get(self.refine_mode, refine_modes["standard"])
        return cv.createLineSegmentDetector(refine_value)

    def detect(self, image: np.ndarray) -> LineSet:
        cv = _require_opencv()
        if image.ndim == 3:
            grayscale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            grayscale = image
        else:
            raise ValueError(f"Unsupported image shape for detection: {image.shape}")

        detector = self._create_detector()
        raw_lines, widths, precisions, _nfas = detector.detect(grayscale)
        segments = []

        if raw_lines is not None:
            precision_values = precisions.reshape(-1) if precisions is not None else None
            for index, raw_line in enumerate(raw_lines.reshape(-1, 4)):
                x1, y1, x2, y2 = [float(value) for value in raw_line]
                confidence = 1.0
                if precision_values is not None:
                    confidence = 1.0 / (1.0 + max(float(precision_values[index]), 0.0))
                segments.append(
                    build_line_segment(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        detector_name=self.name,
                    )
                )

        filtered_segments = filter_short_segments(segments, self.min_line_length)
        return LineSet(
            segments=filtered_segments,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            metadata={
                "detector_name": self.name,
                "refine_mode": self.refine_mode,
                "min_line_length": self.min_line_length,
                "raw_line_count": len(segments),
                "filtered_line_count": len(filtered_segments),
                "has_widths": widths is not None,
                "has_precisions": precisions is not None,
            },
        )


register_detector(OpenCVLSDDetector.name, OpenCVLSDDetector)


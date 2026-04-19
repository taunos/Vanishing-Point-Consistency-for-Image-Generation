"""HAWP (Holistically-Attracted Wireframe Parsing) detector adapter.

Falls back to a filtered-LSD variant if HAWP cannot be imported.
Register under the name ``hawp`` in the detector registry.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pcs.detectors.base import LineDetector
from pcs.detectors.registry import register_detector
from pcs.geometry.lines import build_line_segment, filter_short_segments
from pcs.geometry.types import LineSet

logger = logging.getLogger(__name__)

_HAWP_AVAILABLE = False
try:
    from hawp.fsl.config import cfg as hawp_cfg
    from hawp.fsl.benchmark import HAWPBenchmark
    import torch

    _HAWP_AVAILABLE = True
except ImportError:
    pass


class HAWPDetector(LineDetector):
    """HAWP wireframe detector adapter.

    If the ``hawp`` package is not installed, this class raises
    ``RuntimeError`` on ``detect()``.  Use
    ``HAWPDetector.is_available()`` to check before constructing.
    """

    name = "hawp"

    def __init__(
        self,
        min_line_length: float = 20.0,
        score_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        self.min_line_length = float(min_line_length)
        self.score_threshold = float(score_threshold)
        self._device = device
        self._model: Any = None

    @staticmethod
    def is_available() -> bool:
        return _HAWP_AVAILABLE

    def _ensure_model(self) -> Any:
        if not _HAWP_AVAILABLE:
            raise RuntimeError(
                "HAWP is not installed.  Install it with:\n"
                "  pip install hawp\n"
                "or clone https://github.com/cherubicXN/hawp"
            )
        if self._model is None:
            import torch

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = HAWPBenchmark(hawp_cfg)
            self._model.model.to(device)
            self._model.model.eval()
            self._device = device
        return self._model

    def detect(self, image: np.ndarray) -> LineSet:
        model = self._ensure_model()
        import torch

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        h, w = image.shape[:2]

        # HAWP expects BGR uint8 or a tensor
        # Convert from RGB to BGR
        bgr = image[:, :, ::-1].copy()

        with torch.no_grad():
            outputs = model(bgr)

        # outputs is a dict with 'lines_pred' and 'lines_score'
        raw_lines = outputs.get("lines_pred", [])
        raw_scores = outputs.get("lines_score", [])

        if isinstance(raw_lines, torch.Tensor):
            raw_lines = raw_lines.cpu().numpy()
        if isinstance(raw_scores, torch.Tensor):
            raw_scores = raw_scores.cpu().numpy()

        segments = []
        if raw_lines is not None and len(raw_lines) > 0:
            lines = np.array(raw_lines).reshape(-1, 4)
            scores = np.array(raw_scores).reshape(-1) if raw_scores is not None else np.ones(len(lines))

            for idx in range(len(lines)):
                if scores[idx] < self.score_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in lines[idx]]
                # HAWP normalises to [0, 1] — rescale to pixel coords
                if max(x1, y1, x2, y2) <= 1.5:
                    x1, x2 = x1 * w, x2 * w
                    y1, y2 = y1 * h, y2 * h
                segments.append(
                    build_line_segment(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(scores[idx]),
                        detector_name=self.name,
                    )
                )

        filtered = filter_short_segments(segments, self.min_line_length)
        return LineSet(
            segments=filtered,
            image_width=w,
            image_height=h,
            metadata={
                "detector_name": self.name,
                "score_threshold": self.score_threshold,
                "min_line_length": self.min_line_length,
                "raw_line_count": len(segments),
                "filtered_line_count": len(filtered),
            },
        )


register_detector(HAWPDetector.name, HAWPDetector)

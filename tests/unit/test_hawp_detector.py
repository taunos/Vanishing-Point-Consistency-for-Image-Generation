"""Unit tests for the HAWP detector adapter.

These tests verify the adapter contract and registration, even when the
underlying ``hawp`` package is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from pcs.detectors.hawp_detector import HAWPDetector
from pcs.detectors.registry import get_detector_class


class TestHAWPRegistration:
    def test_registered_in_registry(self) -> None:
        cls = get_detector_class("hawp")
        assert cls is HAWPDetector

    def test_is_available_returns_bool(self) -> None:
        result = HAWPDetector.is_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(not HAWPDetector.is_available(), reason="hawp package not installed")
class TestHAWPDetection:
    def test_detects_lines_on_checkerboard(self) -> None:
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(0, 256, 32):
            image[y : y + 2, :, :] = 255
        for x in range(0, 256, 32):
            image[:, x : x + 2, :] = 255

        detector = HAWPDetector(min_line_length=10.0, score_threshold=0.3)
        result = detector.detect(image)
        assert result.image_width == 256
        assert result.image_height == 256
        # Should find some lines on a grid
        assert len(result.segments) > 0

    def test_returns_valid_lineset_fields(self) -> None:
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        detector = HAWPDetector(min_line_length=10.0)
        result = detector.detect(image)
        assert result.image_width == 128
        assert result.image_height == 128
        assert isinstance(result.segments, list)
        for seg in result.segments:
            assert seg.length >= 10.0
            assert 0.0 <= seg.confidence <= 1.0

    def test_handles_no_lines_gracefully(self) -> None:
        # Uniform image — unlikely to have lines
        image = np.full((64, 64, 3), 128, dtype=np.uint8)
        detector = HAWPDetector(min_line_length=20.0, score_threshold=0.9)
        result = detector.detect(image)
        assert isinstance(result.segments, list)
        # May be empty, which is fine

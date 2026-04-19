"""Unit tests for the synthetic corruption module."""

from __future__ import annotations

import numpy as np
import pytest

from pcs.corruption.synthetic import (
    CorruptionConfig,
    CorruptionType,
    apply_all_corruptions,
    apply_corruption,
)


def _make_test_image(h: int = 256, w: int = 384) -> np.ndarray:
    """Create a synthetic test image with visible geometric structure."""
    rng = np.random.RandomState(0)
    image = rng.randint(50, 200, (h, w, 3), dtype=np.uint8)
    # Add horizontal and vertical lines for structure
    for y in range(0, h, 32):
        image[y : y + 2, :, :] = 255
    for x in range(0, w, 32):
        image[:, x : x + 2, :] = 255
    return image


class TestCorruptionBasics:
    """Corruption preserves shape, dtype, and determinism."""

    @pytest.mark.parametrize("ctype", list(CorruptionType))
    def test_output_shape_and_dtype(self, ctype: CorruptionType) -> None:
        image = _make_test_image()
        cfg = CorruptionConfig(corruption_type=ctype, severity=0.5)
        result = apply_corruption(image, cfg, seed=42)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("ctype", list(CorruptionType))
    def test_severity_zero_returns_identical(self, ctype: CorruptionType) -> None:
        image = _make_test_image()
        cfg = CorruptionConfig(corruption_type=ctype, severity=0.0)
        result = apply_corruption(image, cfg, seed=42)
        np.testing.assert_array_equal(result, image)

    @pytest.mark.parametrize("ctype", list(CorruptionType))
    def test_deterministic_with_same_seed(self, ctype: CorruptionType) -> None:
        image = _make_test_image()
        cfg = CorruptionConfig(corruption_type=ctype, severity=0.7)
        r1 = apply_corruption(image, cfg, seed=99)
        r2 = apply_corruption(image, cfg, seed=99)
        np.testing.assert_array_equal(r1, r2)

    @pytest.mark.parametrize("ctype", list(CorruptionType))
    def test_high_severity_differs_from_input(self, ctype: CorruptionType) -> None:
        image = _make_test_image()
        cfg = CorruptionConfig(corruption_type=ctype, severity=1.0)
        result = apply_corruption(image, cfg, seed=42)
        assert not np.array_equal(result, image)


class TestApplyAllCorruptions:
    def test_returns_all_types_and_severities(self) -> None:
        image = _make_test_image()
        results = apply_all_corruptions(image, severities=(0.0, 0.5, 1.0), seed=42)
        assert set(results.keys()) == {ct.value for ct in CorruptionType}
        for ctype_name, sev_dict in results.items():
            assert set(sev_dict.keys()) == {0.0, 0.5, 1.0}
            for sev, img in sev_dict.items():
                assert img.shape == image.shape
                assert img.dtype == np.uint8


class TestCorruptionConfigValidation:
    def test_rejects_severity_below_zero(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            CorruptionConfig(corruption_type=CorruptionType.PATCH_SHUFFLE, severity=-0.1)

    def test_rejects_severity_above_one(self) -> None:
        with pytest.raises(ValueError, match="severity"):
            CorruptionConfig(corruption_type=CorruptionType.PATCH_SHUFFLE, severity=1.5)

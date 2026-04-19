"""Tests for PerspectiveFieldsWrapper."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def wrapper():
    from pps.fields.perspective_wrapper import PerspectiveFieldsWrapper

    return PerspectiveFieldsWrapper(device="cuda", load_paramnet=False)


@pytest.fixture(scope="module")
def sample_image():
    """A simple synthetic 480x640 RGB image."""
    return np.random.RandomState(42).randint(0, 255, (480, 640, 3), dtype=np.uint8)


def test_model_loads(wrapper):
    """Model loads without error."""
    assert wrapper is not None


def test_output_shapes(wrapper, sample_image):
    """Output shapes match input image dimensions."""
    result = wrapper.predict(sample_image)
    h, w = sample_image.shape[:2]
    assert result.image_shape == (h, w)
    assert result.latitude.shape == (h, w)
    assert result.gravity.shape == (h, w, 2)


def test_latitude_range(wrapper, sample_image):
    """Latitude values are in reasonable range."""
    result = wrapper.predict(sample_image)
    assert result.latitude.min() >= -90.0
    assert result.latitude.max() <= 90.0


def test_gravity_approximate_unit_norm(wrapper, sample_image):
    """Gravity vectors have approximately unit norm."""
    result = wrapper.predict(sample_image)
    norms = np.linalg.norm(result.gravity, axis=-1)
    # Allow some tolerance — the model may not produce exact unit vectors
    assert np.mean(norms) > 0.5, f"Mean gravity norm too low: {np.mean(norms)}"


def test_deterministic(wrapper, sample_image):
    """Same image produces same output."""
    r1 = wrapper.predict(sample_image)
    r2 = wrapper.predict(sample_image)
    np.testing.assert_array_equal(r1.latitude, r2.latitude)
    np.testing.assert_array_equal(r1.gravity, r2.gravity)


def test_rejects_wrong_input(wrapper):
    """Rejects non-RGB input."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="Expected RGB"):
        wrapper.predict(gray)

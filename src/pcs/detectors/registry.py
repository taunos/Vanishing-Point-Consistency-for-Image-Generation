"""Detector registry for swappable implementations."""

from __future__ import annotations

from typing import Any

from pcs.detectors.base import LineDetector

_DETECTOR_REGISTRY: dict[str, type[LineDetector]] = {}


def register_detector(name: str, detector_cls: type[LineDetector]) -> None:
    """Register a detector implementation."""

    _DETECTOR_REGISTRY[name] = detector_cls


def get_detector_class(name: str) -> type[LineDetector]:
    """Look up a registered detector class by name."""

    if name not in _DETECTOR_REGISTRY:
        available = ", ".join(sorted(_DETECTOR_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown detector '{name}'. Available detectors: {available}")
    return _DETECTOR_REGISTRY[name]


def create_detector(name: str, **kwargs: Any) -> LineDetector:
    """Instantiate a registered detector."""

    detector_cls = get_detector_class(name)
    return detector_cls(**kwargs)


from pcs.detectors import opencv_lsd as _opencv_lsd  # noqa: E402,F401

try:
    from pcs.detectors import hawp_detector as _hawp_detector  # noqa: E402,F401
except Exception:  # pragma: no cover — optional dependency
    pass


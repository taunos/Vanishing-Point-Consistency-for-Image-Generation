"""Image loading and path iteration helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def iter_image_paths(path: str | Path) -> list[Path]:
    """Return a deterministic list of image files from a file or directory."""

    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES
    )


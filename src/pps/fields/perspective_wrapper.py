"""Thin wrapper around Jin et al.'s Perspective Fields models.

Handles model loading, inference, and output normalization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerspectiveFieldResult:
    """Output of Perspective Fields inference."""

    latitude: np.ndarray  # (H, W) latitude in degrees
    gravity: np.ndarray  # (H, W, 2) per-pixel gravity direction
    image_shape: tuple[int, int]  # (H, W)
    focal_length: float | None = None  # estimated focal length in pixels
    principal_point: tuple[float, float] | None = None  # (cx, cy)
    roll: float | None = None  # camera roll in degrees
    pitch: float | None = None  # camera pitch in degrees
    fov: float | None = None  # field of view in degrees


class PerspectiveFieldsWrapper:
    """Wraps PersNet (dense field) and optionally Paramnet (camera params)."""

    def __init__(self, device: str = "cuda", load_paramnet: bool = True) -> None:
        self._device = device
        self._load_paramnet = load_paramnet
        self._persnet = None
        self._paramnet = None

    def _ensure_persnet(self):
        if self._persnet is None:
            from perspective2d import PerspectiveFields

            self._persnet = PerspectiveFields("PersNet-360Cities").eval()
            if self._device == "cuda" and torch.cuda.is_available():
                self._persnet = self._persnet.cuda()
            logger.info("Loaded PersNet-360Cities on %s", self._device)

    def _ensure_paramnet(self):
        if self._paramnet is None and self._load_paramnet:
            try:
                from perspective2d import PerspectiveFields

                self._paramnet = PerspectiveFields(
                    "Paramnet-360Cities-edina-centered"
                ).eval()
                if self._device == "cuda" and torch.cuda.is_available():
                    self._paramnet = self._paramnet.cuda()
                logger.info("Loaded Paramnet-360Cities-edina-centered on %s", self._device)
            except Exception as e:
                logger.warning("Failed to load Paramnet model: %s", e)
                self._load_paramnet = False

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> PerspectiveFieldResult:
        """Run inference on an image.

        Args:
            image: RGB uint8 numpy array (H, W, 3).

        Returns:
            PerspectiveFieldResult with all available fields.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

        # PersNet expects BGR
        img_bgr = image[:, :, ::-1].copy()
        h, w = image.shape[:2]

        self._ensure_persnet()
        preds = self._persnet.inference(img_bgr=img_bgr)

        latitude = _to_numpy(preds["pred_latitude_original"])
        gravity_raw = _to_numpy(preds["pred_gravity_original"])
        # gravity comes as (2, H, W) -> transpose to (H, W, 2)
        if gravity_raw.ndim == 3 and gravity_raw.shape[0] == 2:
            gravity = np.transpose(gravity_raw, (1, 2, 0))
        else:
            gravity = gravity_raw

        # Camera params from Paramnet
        focal_length = None
        principal_point = None
        roll = pitch = fov = None

        self._ensure_paramnet()
        if self._paramnet is not None:
            try:
                cam_preds = self._paramnet.inference(img_bgr=img_bgr)
                roll = float(cam_preds["pred_roll"].item())
                pitch = float(cam_preds["pred_pitch"].item())
                fov = float(cam_preds["pred_general_vfov"].item())
                # Derive focal length from vertical FOV
                focal_length = (h / 2.0) / np.tan(np.radians(fov / 2.0))
                cx_rel = float(cam_preds.get("pred_rel_cx", torch.tensor(0.0)).item())
                cy_rel = float(cam_preds.get("pred_rel_cy", torch.tensor(0.0)).item())
                principal_point = (w / 2.0 + cx_rel * w, h / 2.0 + cy_rel * h)
            except Exception as e:
                logger.warning("Paramnet inference failed: %s", e)

        return PerspectiveFieldResult(
            latitude=latitude,
            gravity=gravity,
            image_shape=(h, w),
            focal_length=focal_length,
            principal_point=principal_point,
            roll=roll,
            pitch=pitch,
            fov=fov,
        )


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    return np.asarray(x)

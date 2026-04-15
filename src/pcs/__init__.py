"""Baseline evaluator package for projective consistency."""

from pcs.geometry.types import (
    ApplicabilityResult,
    GlobalCameraFitResult,
    LineSegment,
    LineSet,
    LocalToGlobalPCSResult,
    Patch,
    PatchGeometricSignature,
    PCSBaselineResult,
    RegionalHypothesisMatch,
    RegionalHypothesis,
    VanishingPointCandidate,
)
from pcs.geometry.camera import HorizonLine

__all__ = [
    "ApplicabilityResult",
    "GlobalCameraFitResult",
    "HorizonLine",
    "LineSegment",
    "LineSet",
    "LocalToGlobalPCSResult",
    "Patch",
    "PatchGeometricSignature",
    "PCSBaselineResult",
    "RegionalHypothesisMatch",
    "RegionalHypothesis",
    "VanishingPointCandidate",
]

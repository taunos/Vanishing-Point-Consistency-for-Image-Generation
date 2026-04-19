"""Typed data structures shared across the evaluator stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from pcs.geometry.camera import HorizonLine


def _check_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(slots=True, frozen=True)
class LineSegment:
    """One detected line segment with precomputed geometry metadata."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    length: float
    angle_rad: float
    detector_name: str

    def __post_init__(self) -> None:
        for name, value in (
            ("x1", self.x1),
            ("y1", self.y1),
            ("x2", self.x2),
            ("y2", self.y2),
            ("confidence", self.confidence),
            ("length", self.length),
            ("angle_rad", self.angle_rad),
        ):
            _check_finite(name, float(value))
        if self.length < 0.0:
            raise ValueError("length must be non-negative")


@dataclass(slots=True)
class LineSet:
    """Detected line segments for one image."""

    segments: list[LineSegment]
    image_width: int
    image_height: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Patch:
    """Deterministic image patch definition."""

    patch_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    scale_level: int
    overlap_tag: str

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) * 0.5, (self.y0 + self.y1) * 0.5)


@dataclass(slots=True, frozen=True)
class VanishingPointCandidate:
    """A scored candidate vanishing point."""

    x: float
    y: float
    score: float
    num_inliers: int


@dataclass(slots=True)
class RegionalHypothesis:
    """Baseline patch-local projective hypothesis."""

    patch: Patch
    vp_candidates: list[VanishingPointCandidate]
    support_score: float
    stability_score: float
    num_lines: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RegionalHypothesisMatch:
    """Best compatibility match between two regional hypotheses."""

    patch_id_a: str
    patch_id_b: str
    vp_idx_a: int
    vp_idx_b: int
    vp_position_a: tuple[float, float] | None
    vp_position_b: tuple[float, float] | None
    compatibility_score: float
    geometric_error: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PatchGeometricSignature:
    """Compact patch-level summary for cross-patch comparison."""

    patch_id: str
    dominant_vp: tuple[float, float] | None
    support_score: float
    stability_score: float
    orientation_histogram: list[float] | None
    normalized_direction: tuple[float, float] | None
    vp_candidates: tuple = field(default_factory=tuple)  # (x, y, score, num_inliers) per VP
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ApplicabilityResult:
    """Applicability gate output for one image."""

    confidence: float
    passed: bool
    features: dict[str, float]


@dataclass(slots=True, frozen=True)
class GlobalCameraFitResult:
    """Approximate evaluator-side global consensus fit result."""

    success: bool
    score: float
    num_supported_patches: int
    num_consistent_matches: int
    mean_error: float
    fitted_horizon: HorizonLine | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PCSBaselineResult:
    """Baseline PCS output for one image."""

    pcs_score: float
    applicability: ApplicabilityResult
    local_score: float
    regional_score: float
    num_lines: int
    num_patches: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class LocalToGlobalPCSResult:
    """Local-to-global Milestone 2 PCS output for one image."""

    pcs_score: float
    applicability_confidence: float
    local_quality_score: float
    regional_quality_score: float
    global_consensus_score: float
    incompatibility_penalty: float
    num_patches: int
    num_supported_patches: int
    metadata: dict[str, Any] = field(default_factory=dict)

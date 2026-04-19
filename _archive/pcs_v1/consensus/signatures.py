"""Patch-level geometric signatures for cross-patch comparison."""

from __future__ import annotations

from pcs.geometry.camera import horizon_y_proxy_from_vp, vp_to_projective_direction
from pcs.geometry.types import PatchGeometricSignature, RegionalHypothesis
from pcs.utils.config import ConsensusConfig


def _normalize_direction(
    source: tuple[float, float],
    target: tuple[float, float],
) -> tuple[float, float] | None:
    return vp_to_projective_direction(source, target)


def build_patch_signature(
    hypothesis: RegionalHypothesis,
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> PatchGeometricSignature:
    """Build a deterministic patch-level signature from a regional hypothesis."""

    dominant_vp = None
    dominant_vp_index = None
    if hypothesis.vp_candidates:
        dominant_vp = (hypothesis.vp_candidates[0].x, hypothesis.vp_candidates[0].y)
        dominant_vp_index = 0

    patch_center = hypothesis.patch.center
    image_center = (image_width * 0.5, image_height * 0.5)
    orientation_histogram = hypothesis.metadata.get("orientation_histogram")
    normalized_direction = (
        _normalize_direction(patch_center, dominant_vp)
        if dominant_vp is not None
        else None
    )
    horizon_y = horizon_y_proxy_from_vp(dominant_vp, image_center=image_center)

    vp_candidates = tuple(
        (c.x, c.y, c.score, c.num_inliers) for c in hypothesis.vp_candidates
    )

    return PatchGeometricSignature(
        patch_id=hypothesis.patch.patch_id,
        dominant_vp=dominant_vp,
        support_score=hypothesis.support_score,
        stability_score=hypothesis.stability_score,
        orientation_histogram=orientation_histogram,
        normalized_direction=normalized_direction,
        vp_candidates=vp_candidates,
        metadata={
            "patch_center_x": float(patch_center[0]),
            "patch_center_y": float(patch_center[1]),
            "image_center_x": float(image_center[0]),
            "image_center_y": float(image_center[1]),
            "candidate_count": float(len(hypothesis.vp_candidates)),
            "dominant_vp_index": float(dominant_vp_index if dominant_vp_index is not None else -1),
            "num_lines": float(hypothesis.num_lines),
            "horizon_y_proxy": horizon_y,
            "actual_orientation_bins": float(len(orientation_histogram)) if orientation_histogram else 0.0,
            "requested_orientation_bins": float(config.signature_orientation_bins),
        },
    )


def build_patch_signatures(
    hypotheses: list[RegionalHypothesis],
    image_width: int,
    image_height: int,
    config: ConsensusConfig,
) -> dict[str, PatchGeometricSignature]:
    """Build deterministic signatures indexed by patch id."""

    return {
        hypothesis.patch.patch_id: build_patch_signature(
            hypothesis=hypothesis,
            image_width=image_width,
            image_height=image_height,
            config=config,
        )
        for hypothesis in sorted(hypotheses, key=lambda item: item.patch.patch_id)
    }

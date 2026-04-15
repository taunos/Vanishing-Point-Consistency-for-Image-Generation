from pcs.consensus.compatibility import match_regional_hypotheses, score_signature_pair
from pcs.geometry.types import Patch, PatchGeometricSignature, RegionalHypothesis, VanishingPointCandidate
from pcs.utils.config import ConsensusConfig


def _signature(
    patch_id: str,
    direction: tuple[float, float] | None,
    horizon_y: float | None,
    orientation_histogram: list[float],
) -> PatchGeometricSignature:
    return PatchGeometricSignature(
        patch_id=patch_id,
        dominant_vp=(200.0, 50.0) if direction is not None else None,
        support_score=0.8,
        stability_score=0.7,
        orientation_histogram=orientation_histogram,
        normalized_direction=direction,
        metadata={
            "patch_center_x": 40.0,
            "patch_center_y": 40.0,
            "image_center_x": 128.0,
            "image_center_y": 128.0,
            "horizon_y_proxy": horizon_y,
        },
    )


def test_compatibility_is_high_for_similar_structure_and_low_for_contradictions() -> None:
    config = ConsensusConfig()
    similar_a = _signature("a", (1.0, 0.0), 55.0, [0.6, 0.3, 0.1, 0.0])
    similar_b = _signature("b", (0.98, 0.05), 57.0, [0.55, 0.35, 0.1, 0.0])
    conflicting = _signature("c", (-1.0, 0.0), 140.0, [0.0, 0.1, 0.3, 0.6])

    similar_score, similar_error, _ = score_signature_pair(
        similar_a, similar_b, image_width=256, image_height=256, config=config
    )
    conflicting_score, conflicting_error, _ = score_signature_pair(
        similar_a, conflicting, image_width=256, image_height=256, config=config
    )

    assert 0.0 <= similar_score <= 1.0
    assert 0.0 <= conflicting_score <= 1.0
    assert similar_score > 0.7
    assert conflicting_score < 0.4
    assert similar_error < conflicting_error


def test_match_records_vp_positions_alongside_indices() -> None:
    config = ConsensusConfig()
    hypothesis_a = RegionalHypothesis(
        patch=Patch("a", 0, 0, 50, 50, 0, "ov20"),
        vp_candidates=[VanishingPointCandidate(200.0, 60.0, 0.8, 5)],
        support_score=0.8,
        stability_score=0.7,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.6, 0.4]},
    )
    hypothesis_b = RegionalHypothesis(
        patch=Patch("b", 60, 0, 110, 50, 0, "ov20"),
        vp_candidates=[VanishingPointCandidate(205.0, 62.0, 0.79, 5)],
        support_score=0.8,
        stability_score=0.7,
        num_lines=6,
        metadata={"viable": True, "orientation_histogram": [0.58, 0.42]},
    )
    signatures = {
        "a": _signature("a", (1.0, 0.1), 60.0, [0.6, 0.4]),
        "b": _signature("b", (0.99, 0.08), 62.0, [0.58, 0.42]),
    }

    match = match_regional_hypotheses(
        hypothesis_a,
        hypothesis_b,
        signatures=signatures,
        image_width=256,
        image_height=256,
        config=config,
    )

    assert match.vp_idx_a == 0
    assert match.vp_idx_b == 0
    assert match.vp_position_a == (200.0, 60.0)
    assert match.vp_position_b == (205.0, 62.0)

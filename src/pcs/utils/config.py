"""Config loading and typed experiment settings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _normalize_weights(values: tuple[float, ...]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return tuple(value / total for value in values)


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 1234


@dataclass(slots=True)
class OutputConfig:
    summary: bool = True


@dataclass(slots=True)
class DetectorConfig:
    name: str = "opencv_lsd"
    min_line_length: float = 20.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatchingConfig:
    scales: list[int] = field(default_factory=lambda: [2, 3, 4])
    overlap_ratio: float = 0.2
    min_patch_size: int = 24


@dataclass(slots=True)
class RegionalConfig:
    min_lines_per_patch: int = 3
    top_k_candidates: int = 3
    orientation_histogram_bins: int = 12
    patch_line_overlap_ratio: float = 0.35
    min_intersection_line_angle_deg: float = 8.0
    angular_inlier_threshold_deg: float = 7.5
    intersection_dedup_radius_ratio: float = 0.03
    min_candidate_inliers: int = 2
    max_vp_distance_factor: float = 6.0
    bootstrap_rounds: int = 12
    bootstrap_sample_ratio: float = 0.75
    bootstrap_match_radius_ratio: float = 0.04

    @property
    def min_intersection_line_angle_rad(self) -> float:
        return math.radians(self.min_intersection_line_angle_deg)

    @property
    def angular_inlier_threshold_rad(self) -> float:
        return math.radians(self.angular_inlier_threshold_deg)


@dataclass(slots=True)
class ApplicabilityConfig:
    min_num_lines: int = 12
    target_num_lines: int = 40
    min_mean_length: float = 20.0
    target_mean_length: float = 60.0
    long_line_length: float = 80.0
    min_long_line_ratio: float = 0.15
    target_long_line_ratio: float = 0.4
    min_orientation_entropy: float = 0.35
    target_orientation_entropy: float = 0.75
    min_viable_patches: int = 3
    target_viable_patches: int = 8
    min_supported_patch_ratio: float = 0.2
    target_supported_patch_ratio: float = 0.5
    supported_patch_min_support: float = 0.35
    confidence_threshold: float = 0.45
    orientation_histogram_bins: int = 12


@dataclass(slots=True)
class ScoringConfig:
    local_weight: float = 0.45
    regional_weight: float = 0.55
    local_line_count_weight: float = 0.25
    local_mean_length_weight: float = 0.35
    local_confidence_weight: float = 0.4
    regional_support_weight: float = 0.45
    regional_stability_weight: float = 0.25
    regional_coverage_weight: float = 0.3
    expected_num_lines: int = 60
    expected_mean_length: float = 80.0

    def __post_init__(self) -> None:
        self.local_weight, self.regional_weight = _normalize_weights(
            (self.local_weight, self.regional_weight)
        )
        (
            self.local_line_count_weight,
            self.local_mean_length_weight,
            self.local_confidence_weight,
        ) = _normalize_weights(
            (
                self.local_line_count_weight,
                self.local_mean_length_weight,
                self.local_confidence_weight,
            )
        )
        (
            self.regional_support_weight,
            self.regional_stability_weight,
            self.regional_coverage_weight,
        ) = _normalize_weights(
            (
                self.regional_support_weight,
                self.regional_stability_weight,
                self.regional_coverage_weight,
            )
        )


@dataclass(slots=True)
class ConsensusConfig:
    graph_mode: str = "all_pairs"
    neighbor_margin_ratio: float = 0.15
    min_patch_support_score: float = 0.25
    min_patch_stability_score: float = 0.15
    signature_orientation_bins: int = 12
    compatibility_mode: str = "generic_projective"
    direction_weight: float = 0.5
    orientation_weight: float = 0.25
    horizon_weight: float = 0.15
    contradiction_weight: float = 0.1
    vp_position_weight: float = 0.0
    vp_position_sigma: float = 0.15
    vp_direction_match_tolerance_deg: float = 15.0
    vp_position_infinity_factor: float = 5.0
    vp_position_no_match_default: float = 0.3
    directional_sigma_deg: float = 25.0
    horizon_y_tolerance_ratio: float = 0.12
    contradiction_angle_deg: float = 130.0
    compatibility_threshold: float = 0.55
    consensus_strategy: str = "greedy_seed_grow"
    min_consensus_size: int = 2
    target_supported_patches: int = 6
    min_edge_score_for_growth: float = 0.5
    error_soft_limit: float = 0.45
    manhattan_assisted: bool = False

    def __post_init__(self) -> None:
        if self.vp_position_weight > 0.0:
            (
                self.direction_weight,
                self.vp_position_weight,
                self.orientation_weight,
                self.horizon_weight,
                self.contradiction_weight,
            ) = _normalize_weights(
                (
                    self.direction_weight,
                    self.vp_position_weight,
                    self.orientation_weight,
                    self.horizon_weight,
                    self.contradiction_weight,
                )
            )
        else:
            (
                self.direction_weight,
                self.orientation_weight,
                self.horizon_weight,
                self.contradiction_weight,
            ) = _normalize_weights(
                (
                    self.direction_weight,
                    self.orientation_weight,
                    self.horizon_weight,
                    self.contradiction_weight,
                )
            )


@dataclass(slots=True)
class LocalToGlobalScoringConfig:
    local_weight: float = 0.2
    regional_weight: float = 0.25
    global_weight: float = 0.4
    coherence_weight: float = 0.15
    version: str = "v1"

    def __post_init__(self) -> None:
        if self.version in ("v2", "v3"):
            # v2/v3: 3-component structure (coherence absorbed into global)
            self.local_weight, self.regional_weight, self.global_weight = _normalize_weights(
                (self.local_weight, self.regional_weight, self.global_weight)
            )
            self.coherence_weight = 0.0
        else:
            # v1 and v4: 4-component structure with explicit coherence weight
            (
                self.local_weight,
                self.regional_weight,
                self.global_weight,
                self.coherence_weight,
            ) = _normalize_weights(
                (
                    self.local_weight,
                    self.regional_weight,
                    self.global_weight,
                    self.coherence_weight,
                )
            )


@dataclass(slots=True)
class ExperimentConfig:
    evaluator_mode: str = "baseline"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    regional: RegionalConfig = field(default_factory=RegionalConfig)
    applicability: ApplicabilityConfig = field(default_factory=ApplicabilityConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    consensus_v4: ConsensusConfig = field(default_factory=ConsensusConfig)
    scoring_v2: LocalToGlobalScoringConfig = field(default_factory=LocalToGlobalScoringConfig)
    scoring_v3: LocalToGlobalScoringConfig = field(default_factory=LocalToGlobalScoringConfig)
    scoring_v4: LocalToGlobalScoringConfig = field(default_factory=LocalToGlobalScoringConfig)


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML or JSON experiment config into typed dataclasses."""

    raw = _load_raw_config(Path(path))
    return ExperimentConfig(
        evaluator_mode=raw.get("evaluator_mode", "baseline"),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        output=OutputConfig(**raw.get("output", {})),
        detector=DetectorConfig(**raw.get("detector", {})),
        patching=PatchingConfig(**raw.get("patching", {})),
        regional=RegionalConfig(**raw.get("regional", {})),
        applicability=ApplicabilityConfig(**raw.get("applicability", {})),
        scoring=ScoringConfig(**raw.get("scoring", {})),
        consensus=ConsensusConfig(**raw.get("consensus", {})),
        consensus_v4=ConsensusConfig(**raw.get("consensus_v4", raw.get("consensus", {}))),
        scoring_v2=LocalToGlobalScoringConfig(**raw.get("scoring_v2", {})),
        scoring_v3=LocalToGlobalScoringConfig(**raw.get("scoring_v3", {})),
        scoring_v4=LocalToGlobalScoringConfig(**raw.get("scoring_v4", {})),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert typed config objects into plain dictionaries."""

    return asdict(config)

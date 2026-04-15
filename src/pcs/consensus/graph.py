"""Deterministic region graph construction for cross-patch comparison."""

from __future__ import annotations

from dataclasses import dataclass, field

from pcs.geometry.types import Patch, RegionalHypothesis
from pcs.utils.config import ConsensusConfig


@dataclass(slots=True, frozen=True)
class RegionGraphNode:
    """One supported regional hypothesis node."""

    node_id: str
    patch_id: str
    weight: float
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RegionGraphEdge:
    """One eligible cross-patch comparison edge."""

    source_patch_id: str
    target_patch_id: str
    overlap_ratio: float
    center_distance: float
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RegionGraph:
    """Cross-patch comparison graph over supported patches."""

    nodes: list[RegionGraphNode]
    edges: list[RegionGraphEdge]
    metadata: dict[str, float] = field(default_factory=dict)


def patch_center_distance(patch_a: Patch, patch_b: Patch) -> float:
    """Return Euclidean distance between patch centers."""

    ax, ay = patch_a.center
    bx, by = patch_b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def patch_overlap_ratio(patch_a: Patch, patch_b: Patch) -> float:
    """Return the symmetric overlap ratio between two patches."""

    x_overlap = max(0, min(patch_a.x1, patch_b.x1) - max(patch_a.x0, patch_b.x0))
    y_overlap = max(0, min(patch_a.y1, patch_b.y1) - max(patch_a.y0, patch_b.y0))
    overlap_area = float(x_overlap * y_overlap)
    if overlap_area <= 0.0:
        return 0.0
    area_a = float(max(patch_a.width, 1) * max(patch_a.height, 1))
    area_b = float(max(patch_b.width, 1) * max(patch_b.height, 1))
    return overlap_area / max(min(area_a, area_b), 1.0)


def patches_overlap(patch_a: Patch, patch_b: Patch) -> bool:
    """Return whether two patches overlap in area."""

    return patch_overlap_ratio(patch_a, patch_b) > 0.0


def patches_are_spatial_neighbors(
    patch_a: Patch,
    patch_b: Patch,
    margin_ratio: float,
) -> bool:
    """Return whether two patches are nearby under an expanded-box test."""

    margin_x = max(patch_a.width, patch_b.width) * margin_ratio
    margin_y = max(patch_a.height, patch_b.height) * margin_ratio
    expanded_a = (
        patch_a.x0 - margin_x,
        patch_a.y0 - margin_y,
        patch_a.x1 + margin_x,
        patch_a.y1 + margin_y,
    )
    expanded_b = (
        patch_b.x0 - margin_x,
        patch_b.y0 - margin_y,
        patch_b.x1 + margin_x,
        patch_b.y1 + margin_y,
    )
    x_overlap = min(expanded_a[2], expanded_b[2]) - max(expanded_a[0], expanded_b[0])
    y_overlap = min(expanded_a[3], expanded_b[3]) - max(expanded_a[1], expanded_b[1])
    return x_overlap >= 0.0 and y_overlap >= 0.0


def hypothesis_is_supported(hypothesis: RegionalHypothesis, config: ConsensusConfig) -> bool:
    """Check whether a patch hypothesis is strong enough for graph inclusion."""

    return (
        bool(hypothesis.vp_candidates)
        and hypothesis.support_score >= config.min_patch_support_score
        and hypothesis.stability_score >= config.min_patch_stability_score
        and hypothesis.metadata.get("viable", False)
    )


def hypothesis_weight(hypothesis: RegionalHypothesis) -> float:
    """Return a compact node weight from support and stability."""

    return max(0.0, min(1.0, 0.5 * (hypothesis.support_score + hypothesis.stability_score)))


def build_region_graph(
    hypotheses: list[RegionalHypothesis],
    config: ConsensusConfig,
) -> RegionGraph:
    """Build a deterministic region graph over supported patch hypotheses."""

    supported_hypotheses = sorted(
        (
            hypothesis
            for hypothesis in hypotheses
            if hypothesis_is_supported(hypothesis, config)
        ),
        key=lambda hypothesis: hypothesis.patch.patch_id,
    )
    nodes = [
        RegionGraphNode(
            node_id=hypothesis.patch.patch_id,
            patch_id=hypothesis.patch.patch_id,
            weight=hypothesis_weight(hypothesis),
            metadata={"scale_level": float(hypothesis.patch.scale_level)},
        )
        for hypothesis in supported_hypotheses
    ]

    edges: list[RegionGraphEdge] = []
    for left_index in range(len(supported_hypotheses)):
        for right_index in range(left_index + 1, len(supported_hypotheses)):
            left = supported_hypotheses[left_index]
            right = supported_hypotheses[right_index]
            overlap_ratio = patch_overlap_ratio(left.patch, right.patch)
            center_distance = patch_center_distance(left.patch, right.patch)

            include_edge = False
            if config.graph_mode == "all_pairs":
                include_edge = True
            elif config.graph_mode == "overlap":
                include_edge = overlap_ratio > 0.0
            elif config.graph_mode == "spatial_neighbors":
                include_edge = patches_are_spatial_neighbors(
                    left.patch,
                    right.patch,
                    margin_ratio=config.neighbor_margin_ratio,
                )
            else:
                raise ValueError(f"Unsupported graph_mode: {config.graph_mode}")

            if not include_edge:
                continue

            edges.append(
                RegionGraphEdge(
                    source_patch_id=left.patch.patch_id,
                    target_patch_id=right.patch.patch_id,
                    overlap_ratio=overlap_ratio,
                    center_distance=center_distance,
                    metadata={
                        "source_scale_level": float(left.patch.scale_level),
                        "target_scale_level": float(right.patch.scale_level),
                    },
                )
            )

    edges.sort(key=lambda edge: (edge.source_patch_id, edge.target_patch_id))
    return RegionGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "num_supported_patches": float(len(nodes)),
            "num_edges": float(len(edges)),
        },
    )


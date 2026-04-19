"""Backward-compatible imports for consensus interfaces."""

from __future__ import annotations

from pcs.consensus.graph import RegionGraph as ConsensusGraph
from pcs.consensus.graph import RegionGraphEdge as PairwiseCompatibility
from pcs.consensus.graph import RegionGraphNode as ConsensusNode
from pcs.geometry.types import GlobalCameraFitResult as ConsensusResult

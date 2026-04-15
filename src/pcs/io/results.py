"""Helpers for serializing evaluator outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pcs.geometry.types import LineSet, RegionalHypothesis


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_aggregate_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate metrics to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_image_result_payload(
    image_path: str,
    line_set: LineSet,
    hypotheses: list[RegionalHypothesis],
    result: Any,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-friendly payload for one evaluated image."""

    payload = {
        "image_path": image_path,
        "result": _to_serializable(result),
        "line_set": {
            "image_width": line_set.image_width,
            "image_height": line_set.image_height,
            "num_segments": len(line_set.segments),
            "metadata": _to_serializable(line_set.metadata),
        },
        "regional_hypotheses": [
            {
                "patch": _to_serializable(hypothesis.patch),
                "num_lines": hypothesis.num_lines,
                "support_score": hypothesis.support_score,
                "stability_score": hypothesis.stability_score,
                "vp_candidates": _to_serializable(hypothesis.vp_candidates),
                "metadata": _to_serializable(hypothesis.metadata),
            }
            for hypothesis in hypotheses
        ],
    }
    if extra_payload:
        payload.update(_to_serializable(extra_payload))
    return payload

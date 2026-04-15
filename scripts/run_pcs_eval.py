"""Run the PCS evaluator in baseline or local-to-global mode."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from pcs.applicability.gate import evaluate_applicability
from pcs.consensus import build_region_graph
from pcs.detectors import create_detector
from pcs.io.results import build_image_result_payload, save_aggregate_csv, save_json
from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.regional.patching import generate_overlapping_grid_patches
from pcs.scoring.baseline import compute_baseline_pcs
from pcs.scoring.local_to_global import compute_local_to_global_pcs
from pcs.utils.config import config_to_dict, load_experiment_config
from pcs.utils.image import iter_image_paths, load_image
from pcs.utils.logging import configure_logging
from pcs.utils.seeds import set_deterministic_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image path or directory.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML or JSON experiment config path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store config snapshots and metrics.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise aggregate summary after evaluation.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "local_to_global"],
        help="Optional scoring mode override. Defaults to the config file mode.",
    )
    return parser.parse_args()


def _safe_result_name(root_input: Path, image_path: Path) -> str:
    if root_input.is_dir():
        relative = image_path.relative_to(root_input)
        return "__".join(relative.with_suffix("").parts)
    return image_path.stem


def main() -> int:
    args = parse_args()
    logger = configure_logging(logging.INFO)
    config = load_experiment_config(args.config)
    set_deterministic_seeds(config.runtime.seed)
    scoring_mode = args.mode or config.evaluator_mode
    summary_requested = args.summary or config.output.summary
    logger.info("Using scoring mode: %s", scoring_mode)

    output_dir = Path(args.output_dir)
    per_image_dir = output_dir / "per_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_to_dict(config),
    }
    save_json(output_dir / "config_snapshot.json", config_snapshot)

    detector = create_detector(
        config.detector.name,
        min_line_length=config.detector.min_line_length,
        **config.detector.params,
    )

    input_path = Path(args.input)
    image_paths = iter_image_paths(input_path)
    aggregate_rows: list[dict[str, object]] = []
    aggregate_payload: dict[str, object] = {
        "config_snapshot": config_snapshot,
        "images": [],
    }

    for image_path in image_paths:
        logger.info("Evaluating %s", image_path)
        image = load_image(image_path)
        line_set = detector.detect(image)
        patches = generate_overlapping_grid_patches(
            image_width=line_set.image_width,
            image_height=line_set.image_height,
            scales=config.patching.scales,
            overlap_ratio=config.patching.overlap_ratio,
            min_patch_size=config.patching.min_patch_size,
        )
        hypotheses = estimate_regional_hypotheses(
            line_set=line_set,
            patches=patches,
            config=config.regional,
        )
        applicability = evaluate_applicability(
            line_set=line_set,
            hypotheses=hypotheses,
            config=config.applicability,
        )
        extra_payload: dict[str, object] | None = None
        if scoring_mode == "baseline":
            result = compute_baseline_pcs(
                line_set=line_set,
                hypotheses=hypotheses,
                applicability=applicability,
                config=config.scoring,
            )
        elif scoring_mode == "local_to_global":
            result, artifacts = compute_local_to_global_pcs(
                line_set=line_set,
                hypotheses=hypotheses,
                applicability=applicability,
                baseline_scoring_config=config.scoring,
                consensus_config=config.consensus,
                scoring_config=config.scoring_v2,
            )
            extra_payload = {"consensus": artifacts}
        else:
            raise ValueError(f"Unsupported scoring mode: {scoring_mode}")

        image_payload = build_image_result_payload(
            image_path=str(image_path),
            line_set=line_set,
            hypotheses=hypotheses,
            result=result,
            extra_payload=extra_payload,
        )
        result_name = _safe_result_name(input_path, image_path)
        save_json(per_image_dir / f"{result_name}.json", image_payload)

        if scoring_mode == "baseline":
            aggregate_rows.append(
                {
                    "image_path": str(image_path),
                    "scoring_mode": scoring_mode,
                    "pcs_score": result.pcs_score,
                    "applicability_confidence": result.applicability.confidence,
                    "applicability_passed": result.applicability.passed,
                    "local_score": result.local_score,
                    "regional_score": result.regional_score,
                    "num_lines": result.num_lines,
                    "num_patches": result.num_patches,
                    "num_viable_patches": result.metadata["num_viable_patches"],
                }
            )
        else:
            graph = build_region_graph(hypotheses, config.consensus)
            aggregate_rows.append(
                {
                    "image_path": str(image_path),
                    "scoring_mode": scoring_mode,
                    "pcs_score": result.pcs_score,
                    "applicability_confidence": result.applicability_confidence,
                    "local_quality_score": result.local_quality_score,
                    "regional_quality_score": result.regional_quality_score,
                    "global_consensus_score": result.global_consensus_score,
                    "incompatibility_penalty": result.incompatibility_penalty,
                    "num_patches": result.num_patches,
                    "num_supported_patches": result.num_supported_patches,
                    "compared_edges": result.metadata["compared_edges"],
                    "matched_edges": result.metadata["matched_edges"],
                    "inconsistent_edges": result.metadata["inconsistent_edges"],
                    "graph_edges": len(graph.edges),
                }
            )
        aggregate_payload["images"].append(image_payload)

    save_aggregate_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    save_json(output_dir / "aggregate_metrics.json", aggregate_payload)

    if summary_requested and aggregate_rows:
        mean_pcs = sum(float(row["pcs_score"]) for row in aggregate_rows) / len(aggregate_rows)
        mean_confidence = sum(
            float(row["applicability_confidence"]) for row in aggregate_rows
        ) / len(aggregate_rows)
        logger.info(
            "Completed %d images | mode=%s | mean PCS=%.4f | mean applicability=%.4f",
            len(aggregate_rows),
            scoring_mode,
            mean_pcs,
            mean_confidence,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

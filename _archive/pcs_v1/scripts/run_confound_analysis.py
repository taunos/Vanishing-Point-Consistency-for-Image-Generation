"""Phase 3 confound analysis: check whether PCS separation is genuine.

Known confounds checked:
  1. Resolution — normalise all images to common size before scoring.
  2. JPEG artifacts — not checked directly (both sets saved as PNG/JPG).
  3. Detector noise — does raw line count alone separate the classes?
  4. Applicability — does the gate trivially filter one class?

Usage:
    python scripts/run_confound_analysis.py \
        --per-image-csv outputs/phase3/per_image_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from pcs.applicability.gate import evaluate_applicability
from pcs.detectors import create_detector
from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.regional.patching import generate_overlapping_grid_patches
from pcs.scoring.local_to_global import compute_local_to_global_pcs
from pcs.utils.config import load_experiment_config
from pcs.utils.image import load_image
from pcs.utils.seeds import set_deterministic_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_per_image_csv(path: Path) -> list[dict]:
    """Load the CSV produced by run_separation_experiment.py."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ("pcs_score", "pcs_confidence", "local_score", "regional_score",
                        "global_score", "coherence_score", "num_lines",
                        "num_patches", "num_supported_patches"):
                if key in row and row[key] not in (None, "", "None"):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        row[key] = None
            if "applicability_pass" in row:
                row["applicability_pass"] = row["applicability_pass"] in ("True", "true", "1")
            rows.append(row)
    return rows


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except (ImportError, ValueError):
        return None


def _evaluate_resized(
    img_path: str,
    long_edge: int,
    detector,
    config,
) -> dict | None:
    """Load image, resize to common long edge, then run PCS."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        scale = long_edge / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        image = np.asarray(img)

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
        result, _ = compute_local_to_global_pcs(
            line_set=line_set,
            hypotheses=hypotheses,
            applicability=applicability,
            baseline_scoring_config=config.scoring,
            consensus_config=config.consensus,
            scoring_config=config.scoring_v2,
        )
        return {
            "pcs_score": result.pcs_score,
            "num_lines": len(line_set.segments),
            "applicability_pass": applicability.passed,
        }
    except Exception:
        logger.debug("Resized eval failed for %s:\n%s", img_path, traceback.format_exc())
        return None


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-image-csv",
        default="outputs/phase3/per_image_results.csv",
        help="Per-image CSV from the separation experiment.",
    )
    parser.add_argument(
        "--eval-config",
        default="configs/experiments/pcs_eval_local_to_global.yaml",
        help="PCS evaluator config.",
    )
    parser.add_argument("--output-dir", default="outputs/phase3", help="Output directory.")
    parser.add_argument("--long-edge", type=int, default=768, help="Common long-edge for resolution normalization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-resize",
        action="store_true",
        help="Skip the resolution-normalised re-run (faster).",
    )
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)
    config = load_experiment_config(args.eval_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.per_image_csv)
    if not csv_path.exists():
        logger.error("Per-image CSV not found: %s. Run run_separation_experiment.py first.", csv_path)
        return

    rows = _load_per_image_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(rows), csv_path)

    real_rows = [r for r in rows if r["source"] == "real"]
    gen_rows = [r for r in rows if r["source"] == "generated"]
    logger.info("Real: %d, Generated: %d", len(real_rows), len(gen_rows))

    report: dict = {}

    # ------------------------------------------------------------------
    # Confound 1: Line count as separator
    # ------------------------------------------------------------------
    real_lines = [float(r["num_lines"]) for r in real_rows if r["num_lines"] is not None]
    gen_lines = [float(r["num_lines"]) for r in gen_rows if r["num_lines"] is not None]
    if real_lines and gen_lines:
        labels = [1] * len(real_lines) + [0] * len(gen_lines)
        scores = real_lines + gen_lines
        report["line_count_only_auroc"] = _auroc(labels, scores)
        report["mean_line_count_real"] = float(np.mean(real_lines))
        report["mean_line_count_generated"] = float(np.mean(gen_lines))
    else:
        report["line_count_only_auroc"] = None
    logger.info("Line-count-only AUROC: %s", report["line_count_only_auroc"])

    # ------------------------------------------------------------------
    # Confound 2: Applicability pass rates
    # ------------------------------------------------------------------
    report["applicability_pass_rate_real"] = (
        sum(1 for r in real_rows if r.get("applicability_pass")) / max(len(real_rows), 1)
    )
    report["applicability_pass_rate_generated"] = (
        sum(1 for r in gen_rows if r.get("applicability_pass")) / max(len(gen_rows), 1)
    )

    # PCS AUROC on gated images only
    gated_real = [float(r["pcs_score"]) for r in real_rows if r.get("applicability_pass") and r["pcs_score"] is not None]
    gated_gen = [float(r["pcs_score"]) for r in gen_rows if r.get("applicability_pass") and r["pcs_score"] is not None]
    if gated_real and gated_gen:
        labels_g = [1] * len(gated_real) + [0] * len(gated_gen)
        scores_g = gated_real + gated_gen
        report["pcs_auroc_gated_only"] = _auroc(labels_g, scores_g)
    else:
        report["pcs_auroc_gated_only"] = None

    # PCS AUROC on all images
    all_real = [float(r["pcs_score"]) for r in real_rows if r["pcs_score"] is not None]
    all_gen = [float(r["pcs_score"]) for r in gen_rows if r["pcs_score"] is not None]
    if all_real and all_gen:
        labels_a = [1] * len(all_real) + [0] * len(all_gen)
        scores_a = all_real + all_gen
        report["pcs_auroc_all_images"] = _auroc(labels_a, scores_a)
    else:
        report["pcs_auroc_all_images"] = None

    logger.info("Applicability rate (real):  %.1f%%", report["applicability_pass_rate_real"] * 100)
    logger.info("Applicability rate (gen):   %.1f%%", report["applicability_pass_rate_generated"] * 100)
    logger.info("PCS AUROC (gated):          %s", report["pcs_auroc_gated_only"])
    logger.info("PCS AUROC (all):            %s", report["pcs_auroc_all_images"])

    # ------------------------------------------------------------------
    # Confound 3: Resolution-normalised re-run
    # ------------------------------------------------------------------
    if not args.skip_resize:
        logger.info("Re-running PCS with resolution normalised to %dpx long edge…", args.long_edge)
        detector = create_detector(
            config.detector.name,
            min_line_length=config.detector.min_line_length,
            **config.detector.params,
        )

        resized_real_scores: list[float] = []
        resized_gen_scores: list[float] = []

        for source, source_rows, score_list in [
            ("real", real_rows, resized_real_scores),
            ("generated", gen_rows, resized_gen_scores),
        ]:
            for i, r in enumerate(source_rows):
                fpath = r["filename"]
                if not Path(fpath).exists():
                    continue
                res = _evaluate_resized(fpath, args.long_edge, detector, config)
                if res is not None:
                    score_list.append(res["pcs_score"])
                if (i + 1) % 50 == 0:
                    logger.info("  [%s] %d / %d", source, i + 1, len(source_rows))

        if resized_real_scores and resized_gen_scores:
            labels_r = [1] * len(resized_real_scores) + [0] * len(resized_gen_scores)
            scores_r = resized_real_scores + resized_gen_scores
            report["resolution_normalized_auroc"] = _auroc(labels_r, scores_r)
        else:
            report["resolution_normalized_auroc"] = None
        logger.info("Resolution-normalised AUROC: %s", report.get("resolution_normalized_auroc"))
    else:
        report["resolution_normalized_auroc"] = "skipped"

    # ------------------------------------------------------------------
    # Per-component AUROC
    # ------------------------------------------------------------------
    for comp in ("local_score", "regional_score", "global_score", "coherence_score"):
        rv = [float(r[comp]) for r in real_rows if r.get(comp) is not None]
        gv = [float(r[comp]) for r in gen_rows if r.get(comp) is not None]
        if rv and gv:
            labels_c = [1] * len(rv) + [0] * len(gv)
            scores_c = rv + gv
            report[f"{comp}_auroc"] = _auroc(labels_c, scores_c)
        else:
            report[f"{comp}_auroc"] = None
        logger.info("  %s AUROC: %s", comp, report[f"{comp}_auroc"])

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_path = output_dir / "confound_analysis.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Confound analysis saved to %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("CONFOUND ANALYSIS SUMMARY")
    logger.info("=" * 60)
    for k, v in report.items():
        if isinstance(v, float):
            logger.info("  %-40s %.4f", k, v)
        else:
            logger.info("  %-40s %s", k, v)
    logger.info("=" * 60)

    # Diagnostic warnings
    if report.get("line_count_only_auroc") is not None and report.get("pcs_auroc_all_images") is not None:
        if report["line_count_only_auroc"] > report["pcs_auroc_all_images"]:
            logger.warning(
                "WARNING: Line count alone has higher AUROC than PCS. "
                "The metric may be a proxy for detector noise, not projective consistency."
            )
    if report.get("applicability_pass_rate_real", 0) > 0.8 and report.get("applicability_pass_rate_generated", 1) < 0.3:
        logger.warning(
            "WARNING: Large gap in applicability rates. Separation may be "
            "trivially driven by the gate, not structural inconsistency."
        )


if __name__ == "__main__":
    main()

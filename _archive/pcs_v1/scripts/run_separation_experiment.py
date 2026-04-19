"""Phase 3 core separation experiment: real vs generated image PCS comparison.

Runs PCS (local-to-global mode) on real and generated image sets, computes
separation statistics (AUROC, Cohen's d, Mann–Whitney U), and generates
publication-quality plots.

Usage:
    python scripts/run_separation_experiment.py \
        --config configs/experiments/phase3_separation.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import yaml

from pcs.applicability.gate import evaluate_applicability
from pcs.detectors import create_detector
from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.regional.patching import generate_overlapping_grid_patches
from pcs.scoring.local_to_global import compute_local_to_global_pcs
from pcs.utils.config import load_experiment_config
from pcs.utils.image import iter_image_paths, load_image
from pcs.utils.seeds import set_deterministic_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PCS evaluation
# --------------------------------------------------------------------------- #

def _evaluate_image(
    image_path: Path,
    detector,
    config,
) -> dict | None:
    """Run PCS pipeline on one image. Returns result dict or None on failure."""
    try:
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
        result, _artifacts = compute_local_to_global_pcs(
            line_set=line_set,
            hypotheses=hypotheses,
            applicability=applicability,
            baseline_scoring_config=config.scoring,
            consensus_config=config.consensus,
            scoring_config=config.scoring_v2,
        )
        return {
            "pcs_score": result.pcs_score,
            "pcs_confidence": result.applicability_confidence,
            "local_score": result.local_quality_score,
            "regional_score": result.regional_quality_score,
            "global_score": result.global_consensus_score,
            "coherence_score": 1.0 - result.incompatibility_penalty,
            "applicability_pass": applicability.passed,
            "num_lines": len(line_set.segments),
            "num_patches": result.num_patches,
            "num_supported_patches": result.num_supported_patches,
        }
    except Exception:
        logger.warning("Failed on %s:\n%s", image_path, traceback.format_exc())
        return None


def _category_from_path(rel_path: str) -> str:
    """Extract category hint from relative path."""
    parts = Path(rel_path).parts
    # e.g. sdxl/strong_structure/prompt_00_seed_42.png → strong_structure
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _generator_from_path(rel_path: str) -> str:
    """Extract generator hint from relative path."""
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def _compute_stats(real_scores: list[float], gen_scores: list[float]) -> dict:
    """Compute separation statistics."""
    stats: dict = {}

    real = np.array(real_scores)
    gen = np.array(gen_scores)

    stats["n_real"] = len(real)
    stats["n_generated"] = len(gen)
    stats["mean_pcs_real"] = float(np.mean(real))
    stats["std_pcs_real"] = float(np.std(real))
    stats["mean_pcs_generated"] = float(np.mean(gen))
    stats["std_pcs_generated"] = float(np.std(gen))

    # Cohen's d
    pooled_std = np.sqrt((np.var(real) + np.var(gen)) / 2.0)
    if pooled_std > 1e-9:
        stats["cohens_d"] = float((np.mean(real) - np.mean(gen)) / pooled_std)
    else:
        stats["cohens_d"] = 0.0

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        labels = np.concatenate([np.ones(len(real)), np.zeros(len(gen))])
        scores = np.concatenate([real, gen])
        stats["auroc"] = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)
        stats["roc_fpr"] = fpr.tolist()
        stats["roc_tpr"] = tpr.tolist()
    except ImportError:
        logger.warning("sklearn not available — AUROC not computed.")
        stats["auroc"] = None

    # Mann–Whitney U
    try:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(real, gen, alternative="greater")
        stats["mann_whitney_u"] = float(u_stat)
        stats["mann_whitney_p"] = float(p_val)
    except ImportError:
        logger.warning("scipy not available — Mann–Whitney not computed.")
        stats["mann_whitney_u"] = None
        stats["mann_whitney_p"] = None

    return stats


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

_REAL_COLOR = "#2196F3"
_GEN_COLOR = "#F44336"


def _save_plots(
    rows: list[dict],
    stats: dict,
    output_dir: Path,
) -> None:
    """Generate and save publication-quality plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    real_scores = [r["pcs_score"] for r in rows if r["source"] == "real"]
    gen_scores = [r["pcs_score"] for r in rows if r["source"] == "generated"]

    # 1. Score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_scores, bins=30, alpha=0.6, color=_REAL_COLOR, label="Real", density=True)
    ax.hist(gen_scores, bins=30, alpha=0.6, color=_GEN_COLOR, label="Generated", density=True)
    auroc_str = f"AUROC = {stats['auroc']:.3f}" if stats.get("auroc") is not None else ""
    ax.set_xlabel("PCS Score [0, 1]")
    ax.set_ylabel("Density")
    ax.set_title(f"PCS Distribution: Real vs Generated  {auroc_str}")
    ax.legend()
    fig.savefig(output_dir / "pcs_distribution_real_vs_gen.png")
    fig.savefig(output_dir / "pcs_distribution_real_vs_gen.pdf")
    plt.close(fig)

    # 2. ROC curve
    if stats.get("roc_fpr") is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(stats["roc_fpr"], stats["roc_tpr"], color=_REAL_COLOR, lw=2,
                label=f"AUROC = {stats['auroc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve: Real vs Generated")
        ax.legend()
        fig.savefig(output_dir / "roc_curve.png")
        fig.savefig(output_dir / "roc_curve.pdf")
        plt.close(fig)

    # 3. Box plot by category
    categories = sorted({r["category"] for r in rows})
    if categories:
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 5))
        positions = []
        labels = []
        data_groups = []
        pos = 0
        for cat in categories:
            real_cat = [r["pcs_score"] for r in rows if r["category"] == cat and r["source"] == "real"]
            gen_cat = [r["pcs_score"] for r in rows if r["category"] == cat and r["source"] == "generated"]
            if real_cat:
                data_groups.append(real_cat)
                positions.append(pos)
                labels.append(f"{cat}\n(real)")
                pos += 1
            if gen_cat:
                data_groups.append(gen_cat)
                positions.append(pos)
                labels.append(f"{cat}\n(gen)")
                pos += 1
            pos += 0.5
        if data_groups:
            bp = ax.boxplot(data_groups, positions=positions, widths=0.6, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(_REAL_COLOR if "(real)" in labels[i] else _GEN_COLOR)
                patch.set_alpha(0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("PCS Score [0, 1]")
            ax.set_title("PCS by Scene Category")
            fig.savefig(output_dir / "pcs_by_category.png")
            fig.savefig(output_dir / "pcs_by_category.pdf")
        plt.close(fig)

    # 4. Applicability rate by category
    if categories:
        fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.2), 5))
        cat_labels = []
        real_rates = []
        gen_rates = []
        for cat in categories:
            real_cat = [r for r in rows if r["category"] == cat and r["source"] == "real"]
            gen_cat = [r for r in rows if r["category"] == cat and r["source"] == "generated"]
            if real_cat or gen_cat:
                cat_labels.append(cat)
                real_rates.append(
                    sum(1 for r in real_cat if r["applicability_pass"]) / max(len(real_cat), 1)
                )
                gen_rates.append(
                    sum(1 for r in gen_cat if r["applicability_pass"]) / max(len(gen_cat), 1)
                )
        x = np.arange(len(cat_labels))
        width = 0.35
        ax.bar(x - width / 2, real_rates, width, color=_REAL_COLOR, alpha=0.7, label="Real")
        ax.bar(x + width / 2, gen_rates, width, color=_GEN_COLOR, alpha=0.7, label="Generated")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Applicability Pass Rate")
        ax.set_title("Applicability Gate by Category")
        ax.legend()
        fig.savefig(output_dir / "applicability_by_category.png")
        fig.savefig(output_dir / "applicability_by_category.pdf")
        plt.close(fig)

    # 5. Component score comparison
    components = ["local_score", "regional_score", "global_score", "coherence_score"]
    comp_labels = ["Local", "Regional", "Global", "Coherence"]
    real_means = []
    gen_means = []
    real_stds = []
    gen_stds = []
    for comp in components:
        rv = [r[comp] for r in rows if r["source"] == "real" and r[comp] is not None]
        gv = [r[comp] for r in rows if r["source"] == "generated" and r[comp] is not None]
        real_means.append(np.mean(rv) if rv else 0)
        gen_means.append(np.mean(gv) if gv else 0)
        real_stds.append(np.std(rv) if rv else 0)
        gen_stds.append(np.std(gv) if gv else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comp_labels))
    width = 0.35
    ax.bar(x - width / 2, real_means, width, yerr=real_stds, color=_REAL_COLOR,
           alpha=0.7, label="Real", capsize=4)
    ax.bar(x + width / 2, gen_means, width, yerr=gen_stds, color=_GEN_COLOR,
           alpha=0.7, label="Generated", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Score [0, 1]")
    ax.set_title("PCS Component Scores: Real vs Generated (±1 std)")
    ax.legend()
    fig.savefig(output_dir / "component_scores.png")
    fig.savefig(output_dir / "component_scores.pdf")
    plt.close(fig)

    logger.info("Plots saved to %s", output_dir)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/phase3_separation.yaml",
        help="Phase 3 experiment config.",
    )
    parser.add_argument("--real-dir", default=None, help="Override real image directory.")
    parser.add_argument("--gen-dir", default=None, help="Override generated image directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--detector", default=None, help="Override detector name.")
    args = parser.parse_args()

    # Load phase3 config
    with open(args.config, "r", encoding="utf-8") as f:
        phase3_cfg = yaml.safe_load(f)

    real_dir = Path(args.real_dir or phase3_cfg["data"]["real_dir"])
    gen_dir = Path(args.gen_dir or phase3_cfg["data"]["generated_dir"])
    output_dir = Path(args.output_dir or phase3_cfg["output"]["dir"])
    seed = phase3_cfg["experiment"]["seed"]

    # Load PCS evaluator config
    scoring_config_path = phase3_cfg["scoring"]["config"]
    config = load_experiment_config(scoring_config_path)
    set_deterministic_seeds(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    detector_name = args.detector or config.detector.name
    detector = create_detector(
        detector_name,
        min_line_length=config.detector.min_line_length,
        **config.detector.params,
    )
    logger.info("Detector: %s", detector_name)

    # Collect image paths
    real_paths = iter_image_paths(real_dir) if real_dir.exists() else []
    gen_paths = iter_image_paths(gen_dir) if gen_dir.exists() else []
    logger.info("Real images: %d, Generated images: %d", len(real_paths), len(gen_paths))

    if not real_paths and not gen_paths:
        logger.error("No images found. Run prepare_dataset.py and generate_images.py first.")
        return

    # Evaluate all images
    rows: list[dict] = []
    failures = 0

    for source, paths, base_dir in [
        ("real", real_paths, real_dir),
        ("generated", gen_paths, gen_dir),
    ]:
        for i, img_path in enumerate(paths):
            rel = str(img_path.relative_to(base_dir))
            logger.info("[%s %d/%d] %s", source, i + 1, len(paths), rel)

            result = _evaluate_image(img_path, detector, config)
            if result is None:
                failures += 1
                continue

            category = _category_from_path(rel)
            generator = _generator_from_path(rel) if source == "generated" else ""

            rows.append({
                "filename": str(img_path),
                "source": source,
                "category": category,
                "generator": generator,
                "prompt": "",
                "detector": detector_name,
                **result,
            })

    logger.info("Evaluated %d images (%d failures).", len(rows), failures)

    if not rows:
        logger.error("No successful evaluations. Cannot compute statistics.")
        return

    # Save per-image CSV
    csv_path = output_dir / "per_image_results.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Per-image results: %s", csv_path)

    # Compute statistics
    real_scores = [r["pcs_score"] for r in rows if r["source"] == "real"]
    gen_scores = [r["pcs_score"] for r in rows if r["source"] == "generated"]

    if real_scores and gen_scores:
        stats = _compute_stats(real_scores, gen_scores)
    else:
        stats = {"warning": "Insufficient data for both classes."}
        logger.warning("Need both real and generated scores for separation stats.")

    stats["num_failures"] = failures
    stats["detector"] = detector_name

    # Applicability coverage
    real_pass = sum(1 for r in rows if r["source"] == "real" and r["applicability_pass"])
    gen_pass = sum(1 for r in rows if r["source"] == "generated" and r["applicability_pass"])
    stats["applicability_pass_rate_real"] = real_pass / max(len(real_scores), 1)
    stats["applicability_pass_rate_generated"] = gen_pass / max(len(gen_scores), 1)

    # Save aggregate results
    agg_path = output_dir / "aggregate_results.json"
    # Remove non-serializable items for JSON
    json_stats = {k: v for k, v in stats.items() if k not in ("roc_fpr", "roc_tpr")}
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(json_stats, f, indent=2)
    logger.info("Aggregate results: %s", agg_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("PHASE 3 SEPARATION EXPERIMENT RESULTS")
    logger.info("=" * 60)
    if "auroc" in stats and stats["auroc"] is not None:
        logger.info("  AUROC:              %.4f", stats["auroc"])
    if "mean_pcs_real" in stats:
        logger.info("  Mean PCS (real):    %.4f ± %.4f", stats["mean_pcs_real"], stats["std_pcs_real"])
        logger.info("  Mean PCS (gen):     %.4f ± %.4f", stats["mean_pcs_generated"], stats["std_pcs_generated"])
    if "cohens_d" in stats:
        logger.info("  Cohen's d:          %.4f", stats["cohens_d"])
    if stats.get("mann_whitney_p") is not None:
        logger.info("  Mann-Whitney p:     %.6f", stats["mann_whitney_p"])
    logger.info("  Applicability (real): %.1f%%", stats.get("applicability_pass_rate_real", 0) * 100)
    logger.info("  Applicability (gen):  %.1f%%", stats.get("applicability_pass_rate_generated", 0) * 100)
    logger.info("  Failures:           %d", failures)

    # Decision gate
    auroc = stats.get("auroc")
    if auroc is not None:
        if auroc >= 0.70:
            logger.info("  DECISION: STRONG SIGNAL — proceed to Phase 4")
        elif auroc >= 0.65:
            logger.info("  DECISION: CONTINUE — proceed with caution")
        elif auroc >= 0.55:
            logger.info("  DECISION: PIVOT — investigate component ablations")
        else:
            logger.info("  DECISION: WEAK — consider pivoting or negative result writeup")
    logger.info("=" * 60)

    # Generate plots
    _save_plots(rows, stats, output_dir)


if __name__ == "__main__":
    main()

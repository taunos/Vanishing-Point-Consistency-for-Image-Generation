"""Phase 3 sanity check: PCS sensitivity to synthetic geometric corruption.

Validates that PCS assigns lower scores to more severely corrupted images,
independent of the real-vs-generated question.

Usage:
    python scripts/run_sanity_check.py --real-dir data/real --num-images 50
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import numpy as np

from pcs.applicability.gate import evaluate_applicability
from pcs.corruption.synthetic import CorruptionConfig, CorruptionType, apply_corruption
from pcs.detectors import create_detector
from pcs.regional.hypotheses import estimate_regional_hypotheses
from pcs.regional.patching import generate_overlapping_grid_patches
from pcs.scoring.local_to_global import compute_local_to_global_pcs
from pcs.utils.config import load_experiment_config
from pcs.utils.image import iter_image_paths, load_image
from pcs.utils.seeds import set_deterministic_seeds

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEVERITIES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _evaluate_image_array(
    image: np.ndarray, detector, config, scoring_config, consensus_config
) -> float | None:
    """Run PCS pipeline on an in-memory image array. Returns pcs_score or None."""
    try:
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
            consensus_config=consensus_config,
            scoring_config=scoring_config,
        )
        return result.pcs_score
    except Exception:
        logger.debug("Evaluation failed:\n%s", traceback.format_exc())
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", default="data/real", help="Directory of real images.")
    parser.add_argument(
        "--eval-config",
        default="configs/experiments/pcs_eval_local_to_global.yaml",
        help="PCS evaluator config.",
    )
    parser.add_argument("--num-images", type=int, default=50, help="Max images to use.")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/phase3/sanity_<version>).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--scoring-version",
        default="v4",
        choices=["v1", "v2", "v3", "v4"],
        help="Scoring version to use (default: v4).",
    )
    args = parser.parse_args()

    set_deterministic_seeds(args.seed)
    config = load_experiment_config(args.eval_config)
    output_dir = Path(args.output_dir or f"outputs/phase3/sanity_{args.scoring_version}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select scoring config and consensus config based on version flag
    if args.scoring_version == "v4":
        scoring_config = config.scoring_v4
        consensus_config = config.consensus_v4
    elif args.scoring_version == "v3":
        scoring_config = config.scoring_v3
        consensus_config = config.consensus
    elif args.scoring_version == "v2":
        scoring_config = config.scoring_v2
        consensus_config = config.consensus
    else:
        scoring_config = type(config.scoring_v2)(version="v1")
        consensus_config = config.consensus
    logger.info("Using scoring version: %s", scoring_config.version)

    detector = create_detector(
        config.detector.name,
        min_line_length=config.detector.min_line_length,
        **config.detector.params,
    )

    # Select images
    real_dir = Path(args.real_dir)
    all_paths = iter_image_paths(real_dir) if real_dir.exists() else []
    if not all_paths:
        logger.error("No real images found in %s. Run prepare_dataset.py first.", real_dir)
        return

    selected = all_paths[: args.num_images]
    logger.info("Using %d images for sanity check.", len(selected))

    # Collect results: {corruption_type: {severity: [scores]}}
    results: dict[str, dict[float, list[float]]] = {}
    for ctype in CorruptionType:
        results[ctype.value] = {sev: [] for sev in SEVERITIES}

    for img_idx, img_path in enumerate(selected):
        logger.info("[%d/%d] %s", img_idx + 1, len(selected), img_path.name)
        try:
            image = load_image(img_path)
        except Exception as exc:
            logger.warning("  Failed to load: %s", exc)
            continue

        for ctype in CorruptionType:
            for sev in SEVERITIES:
                cfg = CorruptionConfig(corruption_type=ctype, severity=sev)
                corrupted = apply_corruption(image, cfg, seed=args.seed)
                score = _evaluate_image_array(corrupted, detector, config, scoring_config, consensus_config)
                if score is not None:
                    results[ctype.value][sev].append(score)

    # Compute statistics
    summary: dict = {}
    for ctype_name, sev_dict in results.items():
        severity_means = {}
        severity_stds = {}
        all_sevs = []
        all_scores = []
        for sev in SEVERITIES:
            scores = sev_dict[sev]
            if scores:
                severity_means[str(sev)] = float(np.mean(scores))
                severity_stds[str(sev)] = float(np.std(scores))
                all_sevs.extend([sev] * len(scores))
                all_scores.extend(scores)
            else:
                severity_means[str(sev)] = None
                severity_stds[str(sev)] = None

        # Spearman rank correlation
        spearman_rho = None
        if len(all_sevs) >= 6:
            try:
                from scipy.stats import spearmanr
                rho, p_val = spearmanr(all_sevs, all_scores)
                spearman_rho = float(rho)
            except ImportError:
                pass

        # Check monotonicity: score at sev=0 > score at sev=1
        s0 = severity_means.get("0.0")
        s1 = severity_means.get("1.0")
        monotonic_pass = s0 is not None and s1 is not None and s0 > s1

        summary[ctype_name] = {
            "severity_means": severity_means,
            "severity_stds": severity_stds,
            "spearman_rho": spearman_rho,
            "monotonic_pass": monotonic_pass,
            "num_images_evaluated": len(sev_dict.get(0.0, [])),
        }

    # Overall pass criteria
    types_with_negative_rho = sum(
        1
        for v in summary.values()
        if v["spearman_rho"] is not None and v["spearman_rho"] < -0.3
    )
    types_monotonic = sum(1 for v in summary.values() if v["monotonic_pass"])
    overall_pass = types_with_negative_rho >= 3 and types_monotonic == len(CorruptionType)

    report = {
        "per_corruption": summary,
        "types_with_spearman_below_neg03": types_with_negative_rho,
        "types_monotonic": types_monotonic,
        "overall_pass": overall_pass,
    }

    # Save results
    report_path = output_dir / "sanity_check_results.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Sanity check results saved to %s", report_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("=" * 60)
    for ctype_name, data in summary.items():
        rho_str = f"{data['spearman_rho']:.3f}" if data["spearman_rho"] is not None else "N/A"
        mono_str = "PASS" if data["monotonic_pass"] else "FAIL"
        logger.info("  %-25s  ρ=%s  monotonic=%s", ctype_name, rho_str, mono_str)
        means = data["severity_means"]
        for sev in SEVERITIES:
            val = means.get(str(sev))
            logger.info("    sev=%.1f  mean_pcs=%.4f", sev, val if val is not None else float("nan"))
    logger.info("-" * 60)
    logger.info("  Spearman ρ < -0.3:  %d / %d", types_with_negative_rho, len(CorruptionType))
    logger.info("  Monotonic:          %d / %d", types_monotonic, len(CorruptionType))
    logger.info("  OVERALL:            %s", "PASS" if overall_pass else "FAIL")
    logger.info("=" * 60)

    # Generate plots
    _save_plots(summary, output_dir)


def _save_plots(summary: dict, output_dir: Path) -> None:
    """Generate sanity check plots."""
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

    corruption_color = "#FF9800"
    corruption_types = list(summary.keys())

    # Individual plots per corruption type
    for ctype_name, data in summary.items():
        means = data["severity_means"]
        stds = data["severity_stds"]
        sevs = [s for s in SEVERITIES if means.get(str(s)) is not None]
        mean_vals = [means[str(s)] for s in sevs]
        std_vals = [stds[str(s)] for s in sevs]

        if not sevs:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(sevs, mean_vals, yerr=std_vals, color=corruption_color,
                     marker="o", capsize=4, lw=2, label=ctype_name)
        rho = data["spearman_rho"]
        rho_str = f"ρ = {rho:.3f}" if rho is not None else "ρ = N/A"
        ax.set_xlabel("Corruption Severity")
        ax.set_ylabel("PCS Score [0, 1]")
        ax.set_title(f"PCS vs {ctype_name} Severity  ({rho_str})")
        ax.set_xlim(-0.05, 1.05)
        fig.savefig(output_dir / f"sanity_pcs_vs_severity_{ctype_name}.png")
        fig.savefig(output_dir / f"sanity_pcs_vs_severity_{ctype_name}.pdf")
        plt.close(fig)

    # 2x2 summary grid
    if len(corruption_types) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = axes.flatten()
    else:
        fig, axes_flat = plt.subplots(1, len(corruption_types), figsize=(5 * len(corruption_types), 4))
        if len(corruption_types) == 1:
            axes_flat = [axes_flat]

    for idx, ctype_name in enumerate(corruption_types[:4]):
        ax = axes_flat[idx]
        data = summary[ctype_name]
        means = data["severity_means"]
        stds = data["severity_stds"]
        sevs = [s for s in SEVERITIES if means.get(str(s)) is not None]
        mean_vals = [means[str(s)] for s in sevs]
        std_vals = [stds[str(s)] for s in sevs]

        if sevs:
            ax.errorbar(sevs, mean_vals, yerr=std_vals, color=corruption_color,
                         marker="o", capsize=4, lw=2)
        rho = data["spearman_rho"]
        rho_str = f"ρ={rho:.3f}" if rho is not None else "ρ=N/A"
        ax.set_title(f"{ctype_name}  ({rho_str})")
        ax.set_xlabel("Severity")
        ax.set_ylabel("PCS")
        ax.set_xlim(-0.05, 1.05)

    fig.suptitle("Sanity Check: PCS vs Corruption Severity", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "sanity_summary.png")
    fig.savefig(output_dir / "sanity_summary.pdf")
    plt.close(fig)

    logger.info("Sanity check plots saved to %s", output_dir)


if __name__ == "__main__":
    main()

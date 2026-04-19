"""Run PPS on real and generated image directories.

Compute separation statistics, generate plots.

Usage:
    python scripts/run_pps_eval.py
    python scripts/run_pps_eval.py --config configs/experiments/pps_full_eval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pps.fields.perspective_wrapper import PerspectiveFieldsWrapper
from pps.scoring.pps_score import compute_pps


def find_images(directory: Path, extensions: list[str]) -> list[Path]:
    """Recursively find image files."""
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
    return sorted(images)


def load_config(path: str | None) -> dict:
    if path and Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {
        "data": {
            "real_dirs": ["data/real/york_urban"],
            "generated_dirs": ["data/generated/sdxl"],
            "image_extensions": [".jpg", ".jpeg", ".png"],
        },
        "model": {"device": "cuda", "load_paramnet": False},
        "fields": {"grid_size": 4},
        "focal": {"enabled": True, "grid_size": 2},
        "output": {
            "dir": "outputs/evaluation",
            "save_visualizations": True,
            "save_metrics_json": True,
            "save_plots": True,
        },
    }


def run_evaluation(config: dict, root: Path) -> None:
    output_dir = root / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = PerspectiveFieldsWrapper(
        device=config["model"].get("device", "cuda"),
        load_paramnet=config["model"].get("load_paramnet", False),
    )

    grid_fields = config["fields"].get("grid_size", 4)
    grid_focal = config["focal"].get("grid_size", 2)
    use_focal = config["focal"].get("enabled", True)
    extensions = config["data"].get("image_extensions", [".jpg", ".jpeg", ".png"])

    all_results = []

    for label, dirs_key in [("real", "real_dirs"), ("generated", "generated_dirs")]:
        for dir_str in config["data"].get(dirs_key, []):
            img_dir = root / dir_str
            if not img_dir.exists():
                print(f"  WARNING: {img_dir} not found, skipping")
                continue
            images = find_images(img_dir, extensions)
            print(f"\n  [{label}] {img_dir}: {len(images)} images")

            for img_path in images:
                name = img_path.stem
                print(f"    {name}...", end=" ", flush=True)
                from PIL import Image as PILImage
                img = np.array(PILImage.open(img_path))
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[2] == 4:
                    img = img[:, :, :3]

                pps_result = compute_pps(
                    img, wrapper,
                    use_focal_divergence=use_focal,
                    grid_size_fields=grid_fields,
                    grid_size_focal=grid_focal,
                )

                fc = pps_result.field_consistency
                row = {
                    "name": name,
                    "type": label,
                    "source_dir": dir_str,
                    "pps_score": pps_result.pps_score,
                    "pps_confidence": pps_result.pps_confidence,
                    "latitude_std": fc.latitude_std,
                    "gradient_x_std": fc.gradient_x_std,
                    "gradient_y_std": fc.gradient_y_std,
                    "gradient_y_mean": fc.gradient_y_mean,
                    "patch_mean_range": fc.patch_mean_range,
                    "up_angle_mean": fc.up_angle_mean,
                    "up_angle_max": fc.up_angle_max,
                    "up_angle_std": fc.up_angle_std,
                    "field_consistency_score": fc.field_consistency_score,
                }
                if pps_result.focal_divergence:
                    fd = pps_result.focal_divergence
                    row.update({
                        "focal_mean": fd.focal_mean,
                        "focal_std": fd.focal_std,
                        "focal_cv": fd.focal_cv,
                        "focal_range": fd.focal_range,
                        "focal_num_valid": fd.num_valid_regions,
                    })

                all_results.append(row)
                print(f"PPS={pps_result.pps_score:.3f}")

                # Save consistency map visualization
                if config["output"].get("save_visualizations", True) and pps_result.consistency_map is not None:
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    axes[0].imshow(img)
                    axes[0].set_title(f"{name} ({label})")
                    axes[0].axis("off")
                    im = axes[1].imshow(pps_result.consistency_map, cmap="hot")
                    plt.colorbar(im, ax=axes[1], label="Inconsistency")
                    axes[1].set_title(f"PPS={pps_result.pps_score:.3f}")
                    axes[1].axis("off")
                    plt.tight_layout()
                    plt.savefig(viz_dir / f"{label}_{name}_consistency.png", dpi=100)
                    plt.close()

    # Save metrics
    if config["output"].get("save_metrics_json", True):
        with open(output_dir / "pps_metrics.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMetrics saved to {output_dir / 'pps_metrics.json'}")

    # Compute and print summary statistics
    real = [r for r in all_results if r["type"] == "real"]
    gen = [r for r in all_results if r["type"] == "generated"]

    if not real or not gen:
        print("\nNot enough data for comparison.")
        return

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY: {len(real)} real, {len(gen)} generated")
    print(f"{'='*70}")

    metrics_to_compare = [
        "pps_score", "latitude_std", "gradient_x_std", "gradient_y_std",
        "gradient_y_mean", "patch_mean_range", "up_angle_mean",
    ]

    print(f"\n{'Metric':25s} {'Real':>15s} {'Generated':>15s} {'MW p':>10s}")
    print("-" * 70)
    for metric in metrics_to_compare:
        r_vals = [r[metric] for r in real if metric in r]
        g_vals = [r[metric] for r in gen if metric in r]
        if not r_vals or not g_vals:
            continue
        u_stat, p_val = stats.mannwhitneyu(r_vals, g_vals, alternative="two-sided")
        r_str = f"{np.mean(r_vals):.4f}+/-{np.std(r_vals):.4f}"
        g_str = f"{np.mean(g_vals):.4f}+/-{np.std(g_vals):.4f}"
        marker = " **" if p_val < 0.05 else ""
        print(f"{metric:25s} {r_str:>15s} {g_str:>15s} {p_val:>9.4f}{marker}")

    # AUROC
    labels = [0] * len(real) + [1] * len(gen)
    scores = [r["pps_score"] for r in real] + [r["pps_score"] for r in gen]
    # Higher PPS = more consistent = more likely real, so for AUROC real=1
    # We want: real images score higher than generated
    labels_for_auroc = [1] * len(real) + [0] * len(gen)

    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = roc_auc_score(labels_for_auroc, scores)
    print(f"\nAUROC (real vs generated): {auroc:.4f}")

    # Generate plots
    if config["output"].get("save_plots", True):
        _save_plots(real, gen, output_dir, auroc, labels_for_auroc, scores)

    # Save summary
    summary = {
        "num_real": len(real),
        "num_generated": len(gen),
        "auroc": auroc,
        "metrics": {},
    }
    for metric in metrics_to_compare:
        r_vals = [r[metric] for r in real if metric in r]
        g_vals = [r[metric] for r in gen if metric in r]
        if r_vals and g_vals:
            _, p_val = stats.mannwhitneyu(r_vals, g_vals, alternative="two-sided")
            summary["metrics"][metric] = {
                "real_mean": float(np.mean(r_vals)),
                "real_std": float(np.std(r_vals)),
                "gen_mean": float(np.mean(g_vals)),
                "gen_std": float(np.std(g_vals)),
                "p_value": float(p_val),
            }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _save_plots(real, gen, output_dir, auroc, labels, scores):
    """Generate evaluation plots."""
    from sklearn.metrics import roc_curve

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. PPS score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    r_scores = [r["pps_score"] for r in real]
    g_scores = [r["pps_score"] for r in gen]
    bins = np.linspace(
        min(min(r_scores), min(g_scores)) - 0.05,
        max(max(r_scores), max(g_scores)) + 0.05,
        20,
    )
    ax.hist(r_scores, bins=bins, alpha=0.6, label=f"Real (n={len(real)})", color="steelblue")
    ax.hist(g_scores, bins=bins, alpha=0.6, label=f"Generated (n={len(gen)})", color="coral")
    ax.set_xlabel("PPS Score")
    ax.set_ylabel("Count")
    ax.set_title(f"PPS Score Distribution (AUROC={auroc:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "pps_histogram.png", dpi=150)
    plt.savefig(plot_dir / "pps_histogram.pdf")
    plt.close()

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"PPS (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Real vs Generated")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curve.png", dpi=150)
    plt.savefig(plot_dir / "roc_curve.pdf")
    plt.close()

    # 3. Per-metric box plots
    metrics = ["latitude_std", "gradient_x_std", "patch_mean_range", "up_angle_mean"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    for ax, metric in zip(axes, metrics):
        r_vals = [r[metric] for r in real if metric in r]
        g_vals = [r[metric] for r in gen if metric in r]
        bp = ax.boxplot([r_vals, g_vals], tick_labels=["Real", "Generated"], patch_artist=True)
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("coral")
        bp["boxes"][1].set_alpha(0.6)
        ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(plot_dir / "metric_boxplots.png", dpi=150)
    plt.savefig(plot_dir / "metric_boxplots.pdf")
    plt.close()

    print(f"Plots saved to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run PPS evaluation")
    parser.add_argument("--config", type=str, default="configs/experiments/pps_full_eval.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config = load_config(args.config)

    print("PPS Evaluation")
    print(f"Config: {args.config}")
    run_evaluation(config, root)


if __name__ == "__main__":
    main()

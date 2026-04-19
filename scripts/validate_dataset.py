"""Validate the assembled dataset for Phase 3 experiments.

Checks that images load, meet minimum resolution, and prints a summary table.

Usage:
    python scripts/validate_dataset.py --real-dir data/real --gen-dir data/generated
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_SHORT_EDGE = 256


def _scan_images(root: Path) -> list[dict]:
    """Walk *root* and validate every image file."""
    entries: list[dict] = []
    if not root.exists():
        logger.warning("Directory does not exist: %s", root)
        return entries

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        entry: dict = {
            "path": str(path),
            "relative": str(path.relative_to(root)),
            "valid": False,
            "width": 0,
            "height": 0,
            "error": None,
        }
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open after verify
            with Image.open(path) as img:
                w, h = img.size
            entry["width"] = w
            entry["height"] = h
            if min(w, h) < MIN_SHORT_EDGE:
                entry["error"] = f"short_edge={min(w, h)} < {MIN_SHORT_EDGE}"
            else:
                entry["valid"] = True
        except Exception as exc:
            entry["error"] = str(exc)
        entries.append(entry)
    return entries


def _category_from_path(rel_path: str) -> str:
    """Heuristic: extract category from relative path."""
    parts = Path(rel_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def _print_summary(label: str, entries: list[dict]) -> None:
    valid = [e for e in entries if e["valid"]]
    invalid = [e for e in entries if not e["valid"]]
    logger.info("--- %s ---", label)
    logger.info("  Total files scanned:  %d", len(entries))
    logger.info("  Valid:                %d", len(valid))
    logger.info("  Invalid / too small:  %d", len(invalid))

    # Per-category
    cats: dict[str, int] = {}
    for e in valid:
        cat = _category_from_path(e["relative"])
        cats[cat] = cats.get(cat, 0) + 1
    if cats:
        logger.info("  By category:")
        for cat in sorted(cats):
            logger.info("    %-30s %d", cat, cats[cat])

    if invalid:
        logger.info("  Invalid entries:")
        for e in invalid[:10]:
            logger.info("    %s — %s", e["relative"], e["error"])
        if len(invalid) > 10:
            logger.info("    … and %d more", len(invalid) - 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", default="data/real", help="Real images root.")
    parser.add_argument("--gen-dir", default="data/generated", help="Generated images root.")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Where to save validation report.")
    args = parser.parse_args()

    real_entries = _scan_images(Path(args.real_dir))
    gen_entries = _scan_images(Path(args.gen_dir))

    _print_summary("Real Images", real_entries)
    _print_summary("Generated Images", gen_entries)

    # Save report
    metadata_dir = Path(args.metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "real": {
            "total": len(real_entries),
            "valid": sum(1 for e in real_entries if e["valid"]),
            "invalid": sum(1 for e in real_entries if not e["valid"]),
        },
        "generated": {
            "total": len(gen_entries),
            "valid": sum(1 for e in gen_entries if e["valid"]),
            "invalid": sum(1 for e in gen_entries if not e["valid"]),
        },
    }
    report_path = metadata_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to %s", report_path)


if __name__ == "__main__":
    main()

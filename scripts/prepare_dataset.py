"""Download and organise real images for the Phase 3 separation experiment.

Sources tried in order:
  1. York Urban Line Segment Database  (~100 images, small download)
  2. SUN397 architectural subsets via torchvision
  3. Pexels free-licence search (requires PEXELS_API_KEY env var)

Usage:
    python scripts/prepare_dataset.py --output-dir data/real --target-count 200
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import ssl
import zipfile
from pathlib import Path
from urllib import request, error as urlerror

from PIL import Image

# Windows often lacks up-to-date CA bundles for Python's bundled OpenSSL.
# Use an unverified context for dataset downloads (research-only script).
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source 1 — York Urban DB                                                     #
# --------------------------------------------------------------------------- #

_YORK_URBAN_URLS = [
    "https://www.elderlab.yorku.ca/?sdm_process_download=1&download_id=8288",
    "https://www.elderlab.yorku.ca/wp-content/uploads/2020/03/YorkUrbanDB.zip",
    "https://www.elderlab.yorku.ca/wp-content/uploads/2016/02/YorkUrbanDB.zip",
]


def _download_york_urban(output_dir: Path) -> list[dict]:
    """Attempt to download York Urban DB.  Returns manifest entries."""
    dest = output_dir / "york_urban"
    dest.mkdir(parents=True, exist_ok=True)

    for url in _YORK_URBAN_URLS:
        logger.info("Trying York Urban DB from %s …", url)
        try:
            req = request.Request(url, headers={"User-Agent": "PCS-Phase3/1.0"})
            resp = request.urlopen(req, timeout=120, context=_ssl_ctx)
            data = resp.read()
            break
        except (urlerror.URLError, OSError, TimeoutError) as exc:
            logger.warning("  download failed: %s", exc)
            data = None

    if data is None:
        logger.warning("All York Urban mirrors failed.")
        return []

    manifest: list[dict] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for entry in sorted(zf.namelist()):
            low = entry.lower()
            if not any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp")):
                continue
            # Skip thumbnails or very small utility images
            fname = Path(entry).name
            out_path = dest / fname
            with zf.open(entry) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            # Validate
            try:
                with Image.open(out_path) as img:
                    w, h = img.size
                if min(w, h) < 128:
                    out_path.unlink(missing_ok=True)
                    continue
            except Exception:
                out_path.unlink(missing_ok=True)
                continue
            manifest.append({
                "filename": str(out_path.relative_to(output_dir)),
                "source": "york_urban",
                "category": "urban",
                "license": "research",
                "width": w,
                "height": h,
            })

    logger.info("York Urban: extracted %d usable images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Source 2 — SUN397 subsets via torchvision                                    #
# --------------------------------------------------------------------------- #

_SUN397_CATEGORIES = [
    "church/outdoor",
    "corridor",
    "subway_station/platform",
    "office",
    "staircase",
]


def _download_sun397_subsets(output_dir: Path, max_per_cat: int = 50) -> list[dict]:
    """Download SUN397 architectural subsets.  Requires torchvision."""
    try:
        from torchvision.datasets import SUN397
    except ImportError:
        logger.warning("torchvision not installed — skipping SUN397.")
        return []

    dest = output_dir / "sun397_subset"
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_sun397_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    try:
        ds = SUN397(root=str(cache_dir), download=True)
    except Exception as exc:
        logger.warning("SUN397 download failed: %s", exc)
        return []

    # Build label → indices mapping
    for idx in range(len(ds)):
        img_path_str = str(ds._image_files[idx])
        matched_cat = None
        for cat in _SUN397_CATEGORIES:
            if f"/{cat}/" in img_path_str.replace("\\", "/") or f"\\{cat}\\" in img_path_str:
                matched_cat = cat
                break
        if matched_cat is None:
            continue

        cat_dir = dest / matched_cat.replace("/", "_")
        cat_dir.mkdir(parents=True, exist_ok=True)
        existing = list(cat_dir.glob("*"))
        if len(existing) >= max_per_cat:
            continue

        try:
            img, _ = ds[idx]
            fname = f"{matched_cat.replace('/', '_')}_{len(existing):04d}.jpg"
            out_path = cat_dir / fname
            if hasattr(img, "save"):
                img.save(str(out_path), quality=95)
            w, h = img.size
            if min(w, h) < 128:
                out_path.unlink(missing_ok=True)
                continue
            manifest.append({
                "filename": str(out_path.relative_to(output_dir)),
                "source": "sun397",
                "category": matched_cat,
                "license": "research",
                "width": w,
                "height": h,
            })
        except Exception as exc:
            logger.debug("SUN397 image %d failed: %s", idx, exc)

    logger.info("SUN397: extracted %d usable images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Source 3 — Pexels API                                                        #
# --------------------------------------------------------------------------- #

_PEXELS_QUERIES = [
    ("architecture corridor", "corridor"),
    ("cathedral interior", "cathedral"),
    ("office hallway", "hallway"),
    ("urban street perspective", "urban_street"),
    ("subway platform", "subway"),
]


def _download_pexels(output_dir: Path, per_query: int = 40) -> list[dict]:
    """Download images from Pexels free API."""
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        logger.info("PEXELS_API_KEY not set — skipping Pexels source.")
        return []

    dest = output_dir / "pexels"
    dest.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for query, category in _PEXELS_QUERIES:
        cat_dir = dest / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        page = 1
        fetched = 0

        while fetched < per_query:
            url = (
                f"https://api.pexels.com/v1/search?"
                f"query={request.quote(query)}&per_page=40&page={page}"
            )
            req = request.Request(url, headers={"Authorization": api_key})
            try:
                resp = request.urlopen(req, timeout=30, context=_ssl_ctx)
                data = json.loads(resp.read().decode())
            except Exception as exc:
                logger.warning("Pexels query '%s' page %d failed: %s", query, page, exc)
                break

            photos = data.get("photos", [])
            if not photos:
                break

            for photo in photos:
                if fetched >= per_query:
                    break
                img_url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("original")
                if not img_url:
                    continue
                fname = f"{category}_{photo['id']}.jpg"
                out_path = cat_dir / fname
                try:
                    img_req = request.Request(img_url, headers={"User-Agent": "PCS-Phase3/1.0"})
                    img_resp = request.urlopen(img_req, timeout=30, context=_ssl_ctx)
                    with out_path.open("wb") as f:
                        f.write(img_resp.read())
                    with Image.open(out_path) as img:
                        w, h = img.size
                    if min(w, h) < 512:
                        out_path.unlink(missing_ok=True)
                        continue
                    manifest.append({
                        "filename": str(out_path.relative_to(output_dir)),
                        "source": "pexels",
                        "category": category,
                        "license": "pexels_free",
                        "pexels_id": photo["id"],
                        "photographer": photo.get("photographer", ""),
                        "width": w,
                        "height": h,
                    })
                    fetched += 1
                except Exception as exc:
                    logger.debug("Pexels image download failed: %s", exc)
            page += 1

        logger.info("Pexels '%s': fetched %d images.", query, fetched)

    logger.info("Pexels total: %d usable images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/real", help="Root output directory for real images.")
    parser.add_argument("--target-count", type=int, default=200, help="Minimum images to collect.")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Directory for manifest files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    # Try sources in order
    logger.info("=== Source 1: York Urban DB ===")
    manifest.extend(_download_york_urban(output_dir))
    if len(manifest) >= args.target_count:
        logger.info("Reached target (%d images). Stopping.", len(manifest))
    else:
        logger.info("Have %d / %d. Trying next source.", len(manifest), args.target_count)

        logger.info("=== Source 2: SUN397 subsets ===")
        manifest.extend(_download_sun397_subsets(output_dir))
        if len(manifest) >= args.target_count:
            logger.info("Reached target (%d images). Stopping.", len(manifest))
        else:
            logger.info("Have %d / %d. Trying next source.", len(manifest), args.target_count)

            logger.info("=== Source 3: Pexels API ===")
            manifest.extend(_download_pexels(output_dir))

    # Save manifest
    manifest_path = metadata_dir / "real_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest with %d entries to %s", len(manifest), manifest_path)

    if len(manifest) < args.target_count:
        logger.warning(
            "Only collected %d / %d images. Consider adding images manually or "
            "setting PEXELS_API_KEY.",
            len(manifest),
            args.target_count,
        )


if __name__ == "__main__":
    main()

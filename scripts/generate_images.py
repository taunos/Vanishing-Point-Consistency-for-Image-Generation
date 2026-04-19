"""Generate AI images from the Phase 3 prompt pack.

Supports two generator backends:
  1. Stable Diffusion XL (local, via diffusers)
  2. DALL-E 3 (API, requires OPENAI_API_KEY)

Falls back gracefully if a generator is unavailable.

Usage:
    python scripts/generate_images.py --output-dir data/generated --prompt-config configs/datasets/prompt_pack.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Generator 1 — Stable Diffusion XL (local)                                   #
# --------------------------------------------------------------------------- #

def _generate_sdxl(
    prompts_by_category: dict[str, list[str]],
    seeds: list[int],
    output_dir: Path,
    resolution: int = 1024,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> list[dict]:
    """Generate images using SDXL via diffusers."""
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError:
        logger.warning("diffusers / torch not installed — skipping SDXL generation.")
        return []

    manifest: list[dict] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading SDXL pipeline on %s (%s)…", device, dtype)

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            variant="fp16" if dtype == torch.float16 else None,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    except Exception as exc:
        logger.warning("Failed to load SDXL pipeline: %s", exc)
        return []

    for category, prompts in prompts_by_category.items():
        cat_dir = output_dir / "sdxl" / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, prompt in enumerate(prompts):
            for seed in seeds:
                fname = f"prompt_{prompt_idx:02d}_seed_{seed}.png"
                out_path = cat_dir / fname
                if out_path.exists():
                    logger.info("  Skipping existing: %s", out_path)
                    continue

                logger.info("  SDXL | %s | prompt %d | seed %d", category, prompt_idx, seed)
                try:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=resolution,
                        height=resolution,
                        generator=generator,
                    )
                    img = result.images[0]
                    img.save(str(out_path))
                    manifest.append({
                        "filename": str(out_path.relative_to(output_dir.parent)),
                        "generator": "sdxl",
                        "prompt": prompt,
                        "seed": seed,
                        "category": category,
                        "resolution": resolution,
                    })
                except Exception as exc:
                    logger.warning("  SDXL generation failed: %s", exc)

    del pipe
    if "torch" in dir() and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("SDXL: generated %d images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Generator 2 — DALL-E 3 (API)                                                #
# --------------------------------------------------------------------------- #

def _generate_dalle3(
    prompts_by_category: dict[str, list[str]],
    seeds: list[int],
    output_dir: Path,
    resolution: str = "1024x1024",
) -> list[dict]:
    """Generate images using OpenAI DALL-E 3 API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.info("OPENAI_API_KEY not set — skipping DALL-E 3.")
        return []

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed — skipping DALL-E 3.")
        return []

    client = OpenAI(api_key=api_key)
    manifest: list[dict] = []

    for category, prompts in prompts_by_category.items():
        cat_dir = output_dir / "dalle3" / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, prompt in enumerate(prompts):
            # DALL-E 3 doesn't support explicit seeds, so we generate len(seeds) images
            for seed_idx, seed in enumerate(seeds):
                fname = f"prompt_{prompt_idx:02d}_seed_{seed}.png"
                out_path = cat_dir / fname
                if out_path.exists():
                    logger.info("  Skipping existing: %s", out_path)
                    continue

                logger.info("  DALL-E 3 | %s | prompt %d | idx %d", category, prompt_idx, seed_idx)
                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        size=resolution,
                        quality="standard",
                        n=1,
                    )
                    image_url = response.data[0].url
                    # Download the image
                    from urllib import request as urlreq
                    img_data = urlreq.urlopen(image_url, timeout=60).read()
                    img = Image.open(io.BytesIO(img_data))
                    img.save(str(out_path))
                    manifest.append({
                        "filename": str(out_path.relative_to(output_dir.parent)),
                        "generator": "dalle3",
                        "prompt": prompt,
                        "seed": seed,
                        "category": category,
                        "revised_prompt": getattr(response.data[0], "revised_prompt", None),
                    })
                except Exception as exc:
                    logger.warning("  DALL-E 3 generation failed: %s", exc)
                    # Rate limit backoff
                    if "rate" in str(exc).lower():
                        logger.info("  Rate limited — waiting 30 s")
                        time.sleep(30)

    logger.info("DALL-E 3: generated %d images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Generator 3 — Flux-schnell (local fallback via diffusers)                    #
# --------------------------------------------------------------------------- #

def _generate_flux_schnell(
    prompts_by_category: dict[str, list[str]],
    seeds: list[int],
    output_dir: Path,
    resolution: int = 1024,
    num_inference_steps: int = 4,
) -> list[dict]:
    """Generate images using FLUX.1-schnell via diffusers (fast local fallback)."""
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError:
        logger.warning("diffusers / torch not installed — skipping Flux-schnell.")
        return []

    manifest: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    logger.info("Loading Flux-schnell pipeline on %s…", device)
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=dtype,
        )
        pipe = pipe.to(device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    except Exception as exc:
        logger.warning("Failed to load Flux-schnell: %s", exc)
        return []

    for category, prompts in prompts_by_category.items():
        cat_dir = output_dir / "flux_schnell" / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, prompt in enumerate(prompts):
            for seed in seeds:
                fname = f"prompt_{prompt_idx:02d}_seed_{seed}.png"
                out_path = cat_dir / fname
                if out_path.exists():
                    continue

                logger.info("  Flux | %s | prompt %d | seed %d", category, prompt_idx, seed)
                try:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    result = pipe(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        width=resolution,
                        height=resolution,
                        generator=generator,
                    )
                    img = result.images[0]
                    img.save(str(out_path))
                    manifest.append({
                        "filename": str(out_path.relative_to(output_dir.parent)),
                        "generator": "flux_schnell",
                        "prompt": prompt,
                        "seed": seed,
                        "category": category,
                        "resolution": resolution,
                    })
                except Exception as exc:
                    logger.warning("  Flux generation failed: %s", exc)

    del pipe
    if "torch" in dir() and torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Flux-schnell: generated %d images.", len(manifest))
    return manifest


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-config",
        default="configs/datasets/prompt_pack.yaml",
        help="YAML file with prompt categories.",
    )
    parser.add_argument("--output-dir", default="data/generated", help="Root output for generated images.")
    parser.add_argument("--metadata-dir", default="data/metadata", help="Where to save manifest.")
    parser.add_argument(
        "--generators",
        nargs="+",
        default=["sdxl", "dalle3"],
        help="Generators to use. Options: sdxl, dalle3, flux_schnell",
    )
    args = parser.parse_args()

    with open(args.prompt_config, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    prompts_by_category = prompt_config["prompt_categories"]
    gen_cfg = prompt_config.get("generation", {})
    seeds = gen_cfg.get("seeds", [42, 123])
    resolution = gen_cfg.get("resolution", 1024)
    guidance_scale = gen_cfg.get("guidance_scale", 7.5)
    num_inference_steps = gen_cfg.get("num_inference_steps", 30)

    output_dir = Path(args.output_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for gen_name in args.generators:
        if gen_name == "sdxl":
            manifest.extend(_generate_sdxl(
                prompts_by_category, seeds, output_dir,
                resolution=resolution,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ))
        elif gen_name == "dalle3":
            manifest.extend(_generate_dalle3(
                prompts_by_category, seeds, output_dir,
            ))
        elif gen_name == "flux_schnell":
            manifest.extend(_generate_flux_schnell(
                prompts_by_category, seeds, output_dir,
                resolution=resolution,
            ))
        else:
            logger.warning("Unknown generator: %s", gen_name)

    manifest_path = metadata_dir / "gen_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved generation manifest with %d entries to %s", len(manifest), manifest_path)

    if not manifest:
        logger.warning(
            "No images generated. Ensure you have diffusers+torch installed "
            "or set OPENAI_API_KEY for DALL-E 3."
        )


if __name__ == "__main__":
    main()

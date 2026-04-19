# Projective Plausibility Score (PPS)

A research codebase for measuring projective plausibility in AI-generated images using Perspective Fields consistency and analytic camera calibration divergence.

**Core hypothesis:** Real photographs exhibit globally consistent perspective fields (all pixels agree on one camera). AI-generated images with projective errors show discontinuities, inconsistent gradients, or conflicting gravity directions.

## Key Results

- **AUROC = 0.94** separating 102 real (York Urban DB) from 10 SDXL-generated architectural images
- 5 metrics with p < 0.001: `pps_score`, `latitude_std`, `gradient_x_std`, `gradient_y_mean`, `patch_mean_range`
- Best single feature: `gradient_x_std` (horizontal latitude gradient variance) — real photos have near-zero horizontal gradients; generated images break this constraint

## Pipeline

```
Image -> Perspective Fields (PersNet-360Cities) -> Field Consistency Metrics
      -> Line Detection (LSD) -> VP Estimation -> Focal Length Divergence
      -> Combined PPS Score [0, 1]
```

## Installation

```bash
python -m pip install -e .
python -m pip install -e .[dev]
python -m pip install -e .[opencv]
pip install git+https://github.com/jinlinyi/PerspectiveFields.git
```

Requires PyTorch with CUDA support. Tested with PyTorch 2.12+ (nightly cu128) on RTX 5070 Ti.

## Usage

### Run evaluation

```bash
python scripts/run_pps_eval.py --config configs/experiments/pps_full_eval.yaml
```

Outputs: per-image metrics JSON, score histograms, ROC curves, box plots in `outputs/evaluation/`.

### Python API

```python
from pps.fields.perspective_wrapper import PerspectiveFieldsWrapper
from pps.scoring.pps_score import compute_pps
import numpy as np
from PIL import Image

wrapper = PerspectiveFieldsWrapper(device="cuda")
image = np.array(Image.open("photo.jpg"))
result = compute_pps(image, wrapper)
print(f"PPS = {result.pps_score:.3f}")  # higher = more projectively plausible
```

## Data

Image data is not included in this repository (gitignored). To reproduce results:

- **Real images:** Download the [York Urban DB](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/) and place in `data/real/york_urban/`
- **Generated images:** Run `scripts/generate_images.py` or use your own SDXL/Flux/DALL-E generations in `data/generated/`

## Package Layout

```
src/
  pps/                         # Projective Plausibility Score pipeline
    fields/
      perspective_wrapper.py   # PersNet + Paramnet model wrapper
      field_consistency.py     # Latitude/up-vector consistency metrics
    calibration/
      focal_divergence.py      # Per-region focal length divergence
    scoring/
      pps_score.py             # Combined PPS metric
    benchmark/
      dataset.py               # Dataset loading (future)
      evaluation.py            # Evaluation metrics (future)
  pcs/                         # Preserved: line detectors + geometry
    detectors/                 # LSD detector abstraction
    geometry/                  # VP intersection, line helpers
    utils/                     # Config, image utils
scripts/
  run_pps_eval.py              # Main evaluation script
  generate_images.py           # SDXL image generation
configs/
  experiments/
    pps_full_eval.yaml         # Evaluation config
_archive/
  pcs_v1/                      # Archived: original PCS pipeline (Milestones 1-2)
```

## Testing

```bash
pytest tests/ -v
```

20 unit tests covering the wrapper, field consistency, focal divergence, and PPS scoring.

## References

- Jin et al., "Perspective Fields for Single Image Camera Calibration," CVPR 2023 (Highlight)
- Denis et al., "Efficient Edge-Based Methods for Estimating Manhattan Frames in Urban Imagery," ECCV 2008

## Status

**Phase 1 complete.** Next: scale dataset to 50+ images per class across multiple generators (Phase 2).

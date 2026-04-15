# projective-consistency

`projective-consistency` is an evaluator-first research codebase for measuring projective consistency in geometry-rich images such as architecture, corridors, interiors, and urban scenes.

The repository currently includes:

- a swappable line detector interface,
- deterministic multi-scale overlapping patch generation,
- a baseline regional vanishing-point hypothesis estimator,
- a conservative analytic applicability gate,
- a Milestone 1 baseline PCS scorer,
- a Milestone 2 local-to-global consensus evaluator,
- a CLI runner for single images or directories,
- unit tests on synthetic geometry cases.

## What Milestone 2 Adds

Milestone 2 adds an explicit evaluator-side local-to-global path under `src/pcs/consensus`:

- deterministic region graph construction,
- patch-level geometric signatures,
- patch-pair compatibility scoring,
- approximate global consensus fitting,
- incompatibility estimation,
- per-patch inconsistency diagnostics in saved JSON outputs,
- a richer PCS v2 scorer that combines local quality, regional quality, global consensus, and contradiction penalty.

The local-to-global evaluator asks a more structured question than the Milestone 1 baseline:

> Do the regional geometric cues across the image admit a coherent shared projective explanation, or are strong local explanations mutually incompatible?

## What This Repository Still Does Not Implement Yet

This repository still does not implement the final paper method. In particular, it does not yet provide full camera calibration, the final cross-region projective model, learned confidence calibration, refinement/guidance loops, or training losses. The current local-to-global module is an interpretable evaluator-side approximation intended to be testable and replaceable.

The regional estimator is also still a baseline, but it now uses deterministic
intersection-component clustering plus resampling-based stability rather than a
pure one-pass heuristic. That makes Milestone 2 less brittle while still
leaving room for stronger future regional fitting.

## Installation

```bash
python -m pip install -e .
python -m pip install -e .[dev]
python -m pip install -e .[opencv]
```

OpenCV is optional at install time, but the baseline detector adapter expects `opencv-python` if you want to run LSD detection.

## Running The Evaluator

Baseline mode:

```bash
python scripts/run_pcs_eval.py ^
  --input path\to\image.png ^
  --config configs\experiments\pcs_eval_baseline.yaml ^
  --output-dir outputs\baseline_eval
```

Local-to-global mode:

```bash
python scripts/run_pcs_eval.py ^
  --input path\to\image_dir ^
  --config configs\experiments\pcs_eval_local_to_global.yaml ^
  --output-dir outputs\l2g_eval ^
  --summary
```

You can also override the mode from the CLI:

```bash
python scripts/run_pcs_eval.py ^
  --input path\to\image_dir ^
  --config configs\experiments\pcs_eval_baseline.yaml ^
  --output-dir outputs\override_eval ^
  --mode local_to_global ^
  --summary
```

Outputs include:

- per-image JSON metrics,
- aggregate CSV and JSON summaries,
- a saved config snapshot,
- deterministic run metadata including the configured seed,
- richer consensus artifacts in local-to-global mode.

## Testing

```bash
pytest
```

## Package Layout

- `src/pcs/detectors`: detector abstraction and registry
- `src/pcs/geometry`: typed geometry structures, horizon-line helpers, and coarse projective utilities
- `src/pcs/regional`: patch generation and baseline VP hypotheses
- `src/pcs/consensus`: cross-patch compatibility and global consensus modules
- `src/pcs/applicability`: analytic applicability gate
- `src/pcs/scoring`: baseline and local-to-global score composition
- `src/pcs/io`: result serialization helpers
- `src/pcs/utils`: config, image loading, seeds, and logging

## Future Extension Points

The architecture is designed so later milestones can add:

- alternative line detectors such as M-LSD or HAWP,
- stronger multi-hypothesis regional fitting,
- local-to-global camera consensus in `src/pcs/consensus`,
- cross-region incompatibility terms in `src/pcs/consensus`,
- refinement and editing modules.

## Architectural Boundaries

- `src/pcs/regional`: patch generation and patch-local geometric hypothesis estimation only.
- `src/pcs/consensus`: cross-patch graph construction, pairwise compatibility scoring, global consensus fitting, and incompatibility estimation.
- `src/pcs/scoring`: final score composition only.

## Limitations

- The current consensus module is evaluator-side and approximate rather than full physical camera estimation.
- `geometry/camera.py` provides coarse horizon and projective-direction helpers, not a full calibrated camera model.
- The strongest validity claim is still geometry-rich scenes with visible linear perspective evidence.
- Low-evidence scenes should be interpreted through the applicability/confidence outputs rather than the PCS value alone.

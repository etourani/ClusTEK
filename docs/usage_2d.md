# ClusTEK — 2D Usage Guide

This document describes how to use **ClusTEK’s 2D diffusion-enhanced grid clustering pipeline**.
The 2D pipeline is the recommended entry point for new users and provides a complete,
reproducible workflow from raw point data to clustered structures.

---

## Overview

The 2D pipeline consists of two stages:

**Stage-A (Grid selection)**  
Automatically selects grid resolution and density thresholds using either:
- grid search (`TUNING="grid"`), or
- Bayesian optimization (`TUNING="bo"`).

**Stage-B (Diffusion + clustering)**  
Applies diffusion imputation on sparse grid cells and performs
**origin-constrained connected-component analysis (O-CCA)** to obtain final clusters.

---

## Input format

The input must be a CSV file containing at least the following columns:

```
x, y
```

Additional columns (e.g. ground-truth labels) are optional and will be used automatically
for evaluation if present.

---

## Running the 2D pipeline

ClusTEK provides **two equivalent interfaces**:

- a **command-line interface (CLI)** — recommended for reproducibility
- a **Python API** — recommended for scripting and integration

Both interfaces call the same internal pipeline and produce identical results.

---

## Option A: Command-line interface (recommended)

The CLI entry point is:

```
clustek2d
```

To see all available options:

```
clustek2d --help
```

---

### Example: Aggregation dataset (Bayesian optimization)

The following command reproduces the behavior of
`examples/run_aggregation_bo.py`:

```
clustek2d   --input data/aggregation.csv   --outdir examples/out_aggregation_bo   --tuning bo   --w-sil 0.33 --w-dbi 0.34 --w-cov 0.33   --bo-n-calls 50   --h-bounds-rel 0.5,1.25   --R-range 2,20   --k-for-knn 5 --alpha-for-knn 0.8 --target-occ 2.5   --sweep-pct 0.2 --max-bins 200   --beta-list 0.1,0.2,0.25   --cthr-list 0.01,0.02,0.05,0.1   --max-iters 5000 --min-iters 100   --connectivity 4   --make-plots
```

---

### CLI output

The output directory will contain:

- `best_params_summary.json` — final selected parameters and clustering metrics  
- `stageA_pre_diffusion_candidates.csv` — Stage-A candidate grids  
- `stageB_post_diffusion_candidates.csv` — Stage-B diffusion sweep results  
- PDF figures visualizing clusters before and after diffusion (if enabled)

A concise summary of Stage-A, Stage-B, and clustering metrics
is also printed to the terminal.

---

## Option B: Python API

The same pipeline can be executed programmatically using Python.

### Example: Aggregation dataset (Bayesian optimization)

```python
from clustek import run_pipeline_2d

res = run_pipeline_2d(
    points_file="data/aggregation.csv",
    out_dir="out_aggregation_bo",

    TUNING="bo",
    BO_OPT_WEIGHTS=False,
    W_SIL=0.33, W_DBI=0.34, W_COV=0.33,
    BO_N_CALLS=50,
    H_BOUNDS_REL=(0.5, 1.25),
    R_RANGE=(2, 20),

    K_FOR_KNN=5,
    ALPHA_FOR_KNN=0.8,
    TARGET_OCC=2.5,
    SWEEP_PCT=0.2,
    MAX_BINS=200,

    BETA_CANDIDATES=(0.1, 0.2, 0.25),
    CTHR_VALUES=(0.01, 0.02, 0.05, 0.1),
    MAX_ITERS=5000,
    MIN_ITERS=100,

    PERIODIC_CCA=False,
    CONNECTIVITY=4,
    MAKE_PLOTS=True,
)
```

---

## Example scripts

Fully working reference scripts are provided in the `examples/` directory:

- `run_aggregation_grid.py`
- `run_aggregation_bo.py`
- `run_r15_grid.py`
- `run_r15_bo.py`
- `run_sset1_grid.py`
- `run_sset1_bo.py`

---

## Notes

- The 2D pipeline is deterministic given fixed parameters.
- Bayesian optimization requires `scikit-optimize`.
- Diffusion stability is best for `beta ≤ 0.25`.

---

## Next steps

For volumetric snapshots, see the 3D clustering engine (`ClusTEK3D`)
and the corresponding usage documentation.

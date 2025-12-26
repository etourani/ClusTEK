# ClusTEK

**ClusTEK** is a grid-based clustering toolkit built upon grid aggregation, diffusion imputation, and connected-component analysis.
While motivated by molecular simulations, the method is applicable to a wide range of spatially structured datasets.

## Key ideas

1. **Grid aggregation:** ClusTEK discretizes space into a regular grid and assigns each cell a summary value computed from the points it contains (e.g., averages or counts).
2. **Diffusion imputation:** The grid field is stabilized by controlled diffusion, allowing information to propagate between neighboring cells and reducing sparsity and noise.
3. **Origin-Constrained connected components:** Clusters are formed by growing connected components from a user-defined set of origin cells using fast neighborhood connectivity, preventing diffusion from merging unrelated regions.

## Install

From the repository root:

```bash
pip install -e .
```

This installs the core ClusTEK package and its dependencies.

To install additional tools needed for development (testing, linting, benchmarks), use:

```bash
pip install -e ".[dev]"
```

> Requirements are intentionally limited to standard scientific Python packages.  
> Optional 3D post-processing features (e.g., alpha-shape surface reconstruction)  
> may be added in future releases.


## Quickstart

## 2D Clustering Pipeline

**ClusTEK** provides a fully automated 2D diffusion-enhanced grid clustering pipeline.  
This is the recommended entry point for new users.

The pipeline consists of two stages:

- **Stage-A (Grid selection):**  
  Automatic selection of grid resolution and density thresholds using either  
  grid search or Bayesian optimization.

- **Stage-B (Diffusion + clustering):**  
  Diffusion-based imputation on sparse grids followed by  
  origin-constrained connected-component analysis (OC-CCA).

---

### Command-Line Interface

The 2D pipeline can be executed directly from the command line:

```bash
clustek2d \
  --input data/aggregation.csv \
  --outdir out_aggregation \
  --tuning grid \
  --make-plots
```

This runs the complete two-stage pipeline and writes all results  
(JSON summaries, CSV tables, and optional figures) to the output directory.

To see all available options:

```bash
clustek2d --help
```

---

### Python Usage

Programmatic access to the 2D pipeline is available via the Python API.  
We recommend reviewing the fully working reference scripts provided in the `examples/` directory:

- `examples/run_aggregation_grid.py`
- `examples/run_aggregation_bo.py`
- `examples/run_r15_grid.py`, `examples/run_r15_bo.py`
- `examples/run_sset1_grid.py`, `examples/run_sset1_bo.py`

These scripts demonstrate both grid-search and Bayesian-optimization workflows  
and are the recommended starting point for Python users.

---

### Detailed Documentation

For a complete description of the 2D pipeline, including parameter explanations,  
CLI usage, and expected outputs, see:

**2D Usage Guide:** `docs/usage_2d.md`


## Reproducible benchmark scripts

- `examples/run_benchmark_3d.py` — template driver for scanning `(cell_size, C_thr)` and recording runtime/memory.

## Development

Run tests:

```bash
pytest -q
```

Lint (optional):

```bash
ruff check .
```

## Citation

If you use ClusTEK in academic work, please cite:

Tourani, E., Edwards, J. B., Khomami, B. (2025).  
**ClusTEK**: A grid clustering algorithm augmented with diffusion imputation and origin-constrained connected-component analysis:  
Application to polymer crystallization.  
https://doi.org/10.48550/arXiv.2512.16110


## License

Custom MIT-style license — see `LICENSE`.


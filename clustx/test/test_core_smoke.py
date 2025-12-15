import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



import matplotlib
matplotlib.use("Agg")

from clustx import run_pipeline

def test_smoke_runs(tmp_path):
    # --- synth 2-cluster toy data ---
    rng = np.random.default_rng(0)
    a = rng.normal([0.0, 0.0], 0.2, size=(100, 2))
    b = rng.normal([2.0, 2.0], 0.2, size=(100, 2))
    pts = np.vstack([a, b])
    df = pd.DataFrame(pts, columns=["x", "y"])
    csv = tmp_path / "toy.csv"
    df.to_csv(csv, index=False)

    out = tmp_path / "out"

    # --- run tuned mode (grid) with tiny sweeps for speed ---
    res = run_pipeline(
        points_file=str(csv),
        out_dir=str(out),

        # Stage-A (grid) small quantile set
        DENSE_QUANTILES=(0.20, 0.30),

        # Stage-B (diffusion) tiny grid for quick test
        BETA_CANDIDATES=(0.15,),          # single beta
        CTHR_VALUES=(0.01, 0.02),         # two thresholds

        # scoring (keep defaults-ish)
        W_SIL=0.0, W_DBI=0.8, W_COV=0.15, W_REG=0.05,
        K_MIN=2, K_MAX=10,

        # keep plots on (default) and periodic ON (default)
    )

    # --- files exist ---
    assert (out / "stageA_pre_diffusion_candidates.csv").exists()
    assert (out / "stageB_post_diffusion_candidates.csv").exists()
    assert (out / "best_params_summary.json").exists()

    # --- result structure sanity ---
    assert "stageA_best" in res and isinstance(res["stageA_best"], dict)
    assert "stageB_best" in res and isinstance(res["stageB_best"], dict)
    assert res["stageA_best"].get("nx", 0) >= 1
    assert res["stageA_best"].get("ny", 0) >= 1

    # --- plots exist (according to pipeline's keys) ---
    # In tuned mode we export: "before_coarse", "after_std", "after_occa"
    plots = res.get("plots", {})
    for key in ["before_coarse", "after_std", "after_occa"]:
        path = plots.get(key, None)
        assert path is not None, f"Missing plot path for {key}"
        assert os.path.exists(path), f"Plot file not found for {key}: {path}"

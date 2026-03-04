#!/usr/bin/env python3
"""
Unified 2D runner for ClusTEK (Stage-I + Stage-II).

Typical usage:
  python examples/run2d.py --dataset aggregation --tuning grid
  python examples/run2d.py --dataset R15 --tuning bo
  python examples/run2d.py --dataset R15 --tuning bo --bo-opt-weights   # 5D BO variant
  python examples/run2d.py --csv path/to/custom.csv --tuning grid
  or a costum file:
  python examples/run2d.py --csv /path/to/whatever.csv --tuning grid

Outputs (default):
  examples/outputs/2d/<dataset>_<tuning>/
    - best_params_summary.json
    - stageA_pre_diffusion_candidates.csv
    - stageB_post_diffusion_candidates.csv
    - figures (pdf) if MAKE_PLOTS=True
    - stageI_row_summary.json   (always)
    - stageB_metrics_summary.json (if --dump-metrics)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from clustek import run_pipeline_2d as run_pipeline


# -----------------------------
# Dataset resolution
# -----------------------------
def _repo_root_from_examples() -> Path:
    # examples/ is one level below repo root
    return Path(__file__).resolve().parents[1]


def resolve_dataset_csv(dataset: Optional[str], csv_arg: Optional[str]) -> Path:
    """
    If --csv is provided, use it.
    Else resolve --dataset from the repo's data/ layout:
      data/synthetic/<dataset>.csv
      data/md/<dataset>.csv
    with a legacy fallback:
      data/<dataset>.csv
    """
    if csv_arg:
        p = Path(csv_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--csv not found: {p}")
        return p

    if not dataset:
        raise ValueError("Provide either --dataset or --csv.")

    ds = dataset.strip()
    repo = _repo_root_from_examples()

    candidates = [
        repo / "data" / "synthetic" / f"{ds}.csv",
        repo / "data" / "md" / f"{ds}.csv",
        repo / "data" / f"{ds}.csv",  # legacy fallback
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve dataset '{ds}'. Tried:\n" + "\n".join(str(x) for x in candidates)
    )

def default_out_dir(dataset_name: str, tuning: str) -> Path:
    here = Path(__file__).resolve().parent  # examples/
    out = here / "outputs" / "2d" / f"{dataset_name}_{tuning}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------
# Summaries (keeps your “paper table” JSONs)
# -----------------------------
def summarize_stageI(res: Dict[str, Any], *, dataset: str, strategy: str, out_dir: Path) -> Dict[str, Any]:
    """
    Make the small Stage-I JSON row used in the old scripts. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    """
    bestA = res["stageA_best"]
    nx, ny = int(bestA["nx"]), int(bestA["ny"])

    dx_raw = bestA.get("dx")
    dy_raw = bestA.get("dy")
    dx = float(dx_raw[0] if isinstance(dx_raw, (tuple, list)) else dx_raw) if dx_raw is not None else float("nan")
    dy = float(dy_raw[0] if isinstance(dy_raw, (tuple, list)) else dy_raw) if dy_raw is not None else float("nan")

    h = float(max(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else float("nan")
    Q = float(bestA.get("score", float("nan")))

    tiny: Dict[str, Any] = {
        "dataset": dataset,
        "strategy": strategy,
        "nx": nx,
        "ny": ny,
        "h": h,
        "Q": Q,
    }

    # GRID mode reports q + dense_thr :contentReference[oaicite:5]{index=5}
    if str(bestA.get("mode", "")).lower() == "quantile":
        if bestA.get("dense_q") is not None:
            tiny["q"] = float(bestA["dense_q"])
        if bestA.get("dense_thr") is not None:
            tiny["dense_thr"] = float(bestA["dense_thr"])
    else:
        # BO/count mode reports R :contentReference[oaicite:6]{index=6}
        if bestA.get("R") is not None:
            tiny["R"] = int(bestA["R"])

    with open(out_dir / "stageI_row_summary.json", "w") as f:
        json.dump(tiny, f, indent=2)

    # Print a compact console summary (same idea as your scripts)
    print(f"\n[{dataset} | Stage-I = {strategy.upper()}]")
    print(f"(nx, ny) = ({nx}, {ny})")
    if np.isfinite(h):
        print(f"h        = {h:.6g}   (derived as max(dx, dy) with dx={dx:.6g}, dy={dy:.6g})")
    if "q" in tiny:
        print(f"q        = {tiny['q']:.3f}")
    if "dense_thr" in tiny:
        print(f"dense_thr = {tiny['dense_thr']:.6g}")
    if "R" in tiny:
        print(f"R        = {tiny['R']}")
    print(f"Q        = {Q:.6f}")

    return tiny


def dump_stageB_metrics(res: Dict[str, Any], out_dir: Path) -> None:
    """
    Writes metrics JSON like your BO scripts. :contentReference[oaicite:7]{index=7}
    """
    M = res.get("metrics", {})
    with open(out_dir / "stageB_metrics_summary.json", "w") as f:
        json.dump(M, f, indent=2)


# -----------------------------
# Parameter presets (match your old scripts)
# -----------------------------
def build_params(tuning: str, *, bo_opt_weights: bool) -> Dict[str, Any]:
    """
    Defaults are aligned with your existing scripts. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
    """
    common = dict(
        # ---- Grid suggester knobs ----
        K_FOR_KNN=5,
        ALPHA_FOR_KNN=0.8,
        TARGET_OCC=2.5,
        FD_BACKUP=True,
        SWEEP_PCT=0.2,
        MAX_BINS=200,
        # ---- Scoring weights ----
        W_SIL=0.33,
        W_DBI=0.34,
        W_COV=0.33,
        K_MIN=2,
        K_MAX=50,
        # ---- Runtime / topology ----
        PERIODIC_CCA=False,
        CONNECTIVITY=4,
        MAKE_PLOTS=True,
        DO_STD_CCA=True,
    )

    if tuning == "grid":
        # Matches grid scripts :contentReference[oaicite:11]{index=11}
        common.update(
            TUNING="grid",
            DENSE_QUANTILES=(0.20, 0.25, 0.30, 0.35, 0.40, 0.50),
            BETA_CANDIDATES=(0.10, 0.20, 0.25),
            CTHR_VALUES=(0.01, 0.02, 0.05, 0.10),
            MAX_ITERS=5000,
            MIN_ITERS=100,
            TOL=1e-6,
            CHECK_EVERY=10,
        )
        return common

    # tuning == "bo"
    # Matches bo scripts :contentReference[oaicite:12]{index=12}
    common.update(
        TUNING="bo",
        BO_N_CALLS=50,
        H_BOUNDS_REL=(0.5, 1.25),
        R_RANGE=(2, 20),
        BO_OPT_WEIGHTS=bool(bo_opt_weights),
        BETA_CANDIDATES=(0.10, 0.20, 0.25),
        CTHR_VALUES=(0.05, 0.10, 0.20, 0.30),
        MAX_ITERS=5000,
        MIN_ITERS=100,
        TOL=1e-6,
        CHECK_EVERY=10,
    )
    return common


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None, help="aggregation | R15 | s_set1 | ... (resolved under repo_root/data/)")
    p.add_argument("--csv", type=str, default=None, help="Path to a custom CSV with columns x,y (and optional ground-truth labels).")
    p.add_argument("--tuning", type=str, choices=["grid", "bo"], required=True)
    p.add_argument("--out", type=str, default=None, help="Output directory (default: examples/outputs/2d/<dataset>_<tuning>/)")
    p.add_argument("--no-plots", action="store_true", help="Disable MAKE_PLOTS.")
    p.add_argument("--periodic", action="store_true", help="Enable PERIODIC_CCA (default False in your scripts).")
    p.add_argument("--connectivity", type=int, default=4, choices=[4, 8], help="CCA connectivity (4 or 8).")
    p.add_argument("--bo-opt-weights", action="store_true", help="Enable 5D BO weights (W_SIL,W_DBI,W_COV) + (h,R).")
    p.add_argument("--dump-metrics", action="store_true", help="Write stageB_metrics_summary.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_dataset_csv(args.dataset, args.csv)
    dataset_name = args.dataset if args.dataset else csv_path.stem
    tuning = args.tuning.lower()

    out_dir = Path(args.out).expanduser().resolve() if args.out else default_out_dir(dataset_name, tuning)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = build_params(tuning, bo_opt_weights=args.bo_opt_weights)
    if args.no_plots:
        params["MAKE_PLOTS"] = False
    if args.periodic:
        params["PERIODIC_CCA"] = True
    params["CONNECTIVITY"] = int(args.connectivity)

    print(f"\n[run2d] dataset={dataset_name}  tuning={tuning}")
    print(f"[run2d] csv={csv_path}")
    print(f"[run2d] out={out_dir}")

    res = run_pipeline(points_file=str(csv_path), out_dir=str(out_dir), **params)

    summarize_stageI(res, dataset=str(dataset_name), strategy=tuning, out_dir=out_dir)

    # quick pointers to figures (same as old scripts)
    plots = res.get("plots", {})
    if plots:
        print("\nSaved figures:")
        for k, v in plots.items():
            if v:
                print(f"  {k}: {v}")

    print("\n[Stage-II (Diffusion) best]")
    print(f"beta*  = {res.get('best_beta')}")
    print(f"iters* = {res.get('best_iters')}")

    if args.dump_metrics:
        dump_stageB_metrics(res, out_dir)
        print(f"\nWrote: {out_dir / 'stageB_metrics_summary.json'}")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

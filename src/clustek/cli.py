# cli.py — CLI for ClusTEK 2D pipeline (grid/BO; optional 5D BO with weight tuning)
from __future__ import annotations

import argparse
import json
import sys
import os
from typing import List, Tuple, Optional

import numpy as np

from .core2d import run_pipeline_2d


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_range(vals, lo: float, hi: float, name: str) -> None:
    for v in vals:
        if not (lo <= v <= hi):
            raise ValueError(f"{name}: value {v} is out of range [{lo}, {hi}]")


def _fmt(x) -> str:
    if x is None:
        return "nan"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "nan"
        return f"{float(x):.4f}"
    return str(x)


def _print_summary(res: dict) -> None:
    # --- Stage-I summary ---
    bestA = res.get("stageA_best", {})
    if bestA:
        nx = int(bestA.get("nx", -1))
        ny = int(bestA.get("ny", -1))

        dx_raw = bestA.get("dx")
        dy_raw = bestA.get("dy")
        dx = float(dx_raw[0] if isinstance(dx_raw, (tuple, list)) else dx_raw) if dx_raw is not None else float("nan")
        dy = float(dy_raw[0] if isinstance(dy_raw, (tuple, list)) else dy_raw) if dy_raw is not None else float("nan")
        h = max(dx, dy) if np.isfinite(dx) and np.isfinite(dy) else float("nan")

        R = bestA.get("R", None)
        Q = bestA.get("score", None)

        print("\n[Stage-I best]")
        print(f"(nx, ny) = ({nx}, {ny})")
        if np.isfinite(h):
            print(f"h        = {h:.6g}   (derived as max(dx, dy) with dx={dx:.6g}, dy={dy:.6g})")
        else:
            print("h        = nan")
        if R is not None:
            print(f"R        = {int(R)}")
        if Q is not None:
            print(f"Q        = {float(Q):.6f}")

    # --- Where are the figures? ---
    plots = res.get("plots", {}) or {}
    if plots:
        print("\nSaved figures:")
        for k, v in plots.items():
            if v:
                print(f"  {k}: {v}")

    # --- Stage-II winners ---
    if "best_beta" in res or "best_iters" in res:
        print("\n[Stage-II (Diffusion) best]")
        print(f"beta*     = {res.get('best_beta')}")
        print(f"iters*    = {res.get('best_iters')}")

    # --- Metrics ---
    M = res.get("metrics", {}) or {}
    if M:
        rows = []
        for tag in ["before", "after_std", "after_occa"]:
            mt = M.get(tag, {}) or {}
            rows.append({
                "phase": tag,
                "k": mt.get("k"),
                "coverage": _fmt(mt.get("coverage")),
                "silhouette": _fmt(mt.get("silhouette")),
                "DBI": _fmt(mt.get("dbi")),
                "ARI": _fmt(mt.get("ARI")),
                "NMI": _fmt(mt.get("NMI")),
                "V": _fmt(mt.get("V_measure")),
                "FM": _fmt(mt.get("FM")),
                "purity": _fmt(mt.get("purity")),
            })

        print("\n[Clustering metrics]")
        print("phase        k   coverage  silhouette     DBI      ARI      NMI       V        FM    purity")
        for r in rows:
            print(f"{r['phase']:<11} {r['k']!s:>2}   {r['coverage']:>8}  {r['silhouette']:>10}  {r['DBI']:>7}  "
                  f"{r['ARI']:>7}  {r['NMI']:>7}  {r['V']:>7}  {r['FM']:>7}  {r['purity']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="clustek2d",
        description="ClusTEK 2D: Stage-A (grid/BO) + diffusion + origin-constrained CCA (O-CCA)."
    )

    ap.add_argument("--input", required=True, help="CSV with columns x,y")
    ap.add_argument("--outdir", default="./clustek2d_out", help="Output directory")

    # -------- Fixed mode (fast path) --------
    ap.add_argument("--fixed-grid", type=str, default=None,
                    help="nx,ny (e.g., 120,90). If set, skips tuning and uses fixed params.")
    ap.add_argument("--fixed-dense-thr", type=float, default=None,
                    help="Normalized [0,1] threshold defining pre-imputation dense cells.")
    ap.add_argument("--fixed-beta", type=float, default=None,
                    help="Diffusion beta (recommend ≤ 0.25 for stability).")
    ap.add_argument("--fixed-cthr", type=float, default=None,
                    help="Post-diffusion selection threshold C_sel in [0,1].")

    # -------- Tuned mode (used if fixed-grid is NOT set) --------
    ap.add_argument("--tuning", choices=["grid", "bo"], default="grid",
                    help="Stage-A parameter selection mode.")

    # Grid-mode Stage-A options
    ap.add_argument("--dense-qs", type=str, default="0.20,0.25,0.30,0.35,0.40,0.50",
                    help="Comma-separated quantiles for Stage-A when tuning=grid (each in [0,1]).")

    # BO-mode Stage-A options
    ap.add_argument("--bo-n-calls", type=int, default=35, help="BO evaluation budget.")
    ap.add_argument("--h-bounds-rel", type=str, default="0.5,1.8",
                    help="Relative bounds around h0 for BO, e.g., '0.5,1.8'.")
    ap.add_argument("--R-range", type=str, default="1,30",
                    help="Integer R range for BO, e.g., '1,30'.")

    # Optional: BO weight tuning (5D BO)
    try:
        from argparse import BooleanOptionalAction  # py>=3.9
        bool_action = BooleanOptionalAction
    except Exception:
        bool_action = None

    if bool_action:
        ap.add_argument("--bo-opt-weights", default=True, action=bool_action,
                        help="If true, BO also optimizes (W_SIL, W_DBI, W_COV) with h and R.")
        ap.add_argument("--fd-backup", default=True, action=bool_action,
                        help="Enable/disable Freedman–Diaconis backup for h (default: enabled).")
    else:
        ap.add_argument("--bo-opt-weights", action="store_true", default=True,
                        help="Enable BO weight optimization (cannot disable if Python<3.9).")
        ap.add_argument("--fd-backup", action="store_true", default=True,
                        help="Use FD backup (default: enabled). (Cannot disable if Python<3.9)")

    ap.add_argument("--bo-weight-floor", type=float, default=0.10,
                    help="Lower bound for each BO weight component before normalization (default 0.10).")
    ap.add_argument("--bo-weight-ceil", type=float, default=0.90,
                    help="Upper bound for each BO weight component before normalization (default 0.90).")

    # Grid suggester knobs (only used when tuning)
    ap.add_argument("--k-for-knn", type=int, default=5)
    ap.add_argument("--alpha-for-knn", type=float, default=0.8)
    ap.add_argument("--target-occ", type=float, default=4.0)
    ap.add_argument("--sweep-pct", type=float, default=0.4)
    ap.add_argument("--max-bins", type=int, default=300)

    # Stage-B sweeps / iteration controls
    ap.add_argument("--beta-list", type=str, default="0.02,0.05,0.10,0.20,0.50",
                    help="Comma-separated betas for diffusion (tuned mode).")
    ap.add_argument("--cthr-list", type=str,
                    default="0.01,0.03,0.05,0.07,0.09,0.11,0.13,0.15,0.17,0.19,0.21",
                    help="Comma-separated thresholds applied after diffusion (tuned mode).")
    ap.add_argument("--max-iters", type=int, default=50000)
    ap.add_argument("--min-iters", type=int, default=60)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--check-every", type=int, default=10)

    # Scoring weights (used if BO weight tuning is disabled/unavailable)
    ap.add_argument("--w-sil", type=float, default=0.0)
    ap.add_argument("--w-dbi", type=float, default=0.80)
    ap.add_argument("--w-cov", type=float, default=0.15)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=50)

    # Runtime options
    ap.add_argument("--periodic-cca", action="store_true", default=False,
                    help="Enable periodic BCs in diffusion/CCA (default: off).")
    ap.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                    help="Neighborhood connectivity for CCA/O-CCA (default 4).")
    ap.add_argument("--make-plots", action="store_true", default=True,
                    help="Write PDF diagnostic plots (default: enabled).")
    ap.add_argument("--no-std-cca-plot", action="store_true", default=False,
                    help="Skip plotting the standard CCA figure (only O-CCA).")

    # Output control
    ap.add_argument("--print-json", action="store_true", default=False,
                    help="Also print full JSON result to stdout (can be large).")

    args = ap.parse_args()

    try:
        os.makedirs(args.outdir, exist_ok=True)

        # Parse lists / tuples
        BETA_CANDIDATES = tuple(_parse_float_list(args.beta_list))
        CTHR_VALUES     = tuple(_parse_float_list(args.cthr_list))
        DENSE_QS        = tuple(_parse_float_list(args.dense_qs))
        H_BOUNDS_REL    = tuple(_parse_float_list(args.h_bounds_rel))
        R_RANGE_TUP     = tuple(_parse_int_list(args.R_range))

        # Validations
        if len(H_BOUNDS_REL) != 2 or not (H_BOUNDS_REL[0] > 0 and H_BOUNDS_REL[1] >= H_BOUNDS_REL[0]):
            raise ValueError("--h-bounds-rel must be two positive numbers like '0.5,1.8' with hi>=lo.")
        if len(R_RANGE_TUP) != 2 or not (R_RANGE_TUP[0] >= 1 and R_RANGE_TUP[1] >= R_RANGE_TUP[0]):
            raise ValueError("--R-range must be two integers like '1,30' with hi>=lo and lo>=1.")
        _ensure_range(DENSE_QS, 0.0, 1.0, "dense-qs")
        _ensure_range(CTHR_VALUES, 0.0, 1.0, "cthr-list")
        if any(b <= 0 for b in BETA_CANDIDATES):
            raise ValueError("beta-list: all betas must be > 0.")
        if not (0.0 <= args.bo_weight_floor <= 1.0):
            raise ValueError("--bo-weight-floor must be in [0,1].")
        if not (0.0 <= args.bo_weight_ceil <= 1.0):
            raise ValueError("--bo-weight-ceil must be in [0,1].")
        if args.bo_weight_floor > args.bo_weight_ceil:
            raise ValueError("--bo-weight-floor must be <= --bo-weight-ceil.")
        if args.sweep_pct <= 0 or args.sweep_pct > 1.0:
            raise ValueError("--sweep-pct should be in (0,1].")
        if args.max_bins < 1:
            raise ValueError("--max-bins must be >= 1.")
        if args.k_min < 1 or args.k_max < args.k_min:
            raise ValueError("--k-min must be >=1 and --k-max >= --k-min.")

        # Parse fixed-grid
        FIXED_GRID: Optional[Tuple[int, int]] = None
        if args.fixed_grid:
            parts = [p.strip() for p in args.fixed_grid.split(",")]
            if len(parts) != 2:
                raise ValueError("--fixed-grid expects 'nx,ny'")
            FIXED_GRID = (int(parts[0]), int(parts[1]))
            if FIXED_GRID[0] < 1 or FIXED_GRID[1] < 1:
                raise ValueError("--fixed-grid values must be >= 1")

        # Fixed-mode requirements
        if FIXED_GRID is not None:
            for name, val in [("fixed-dense-thr", args.fixed_dense_thr),
                              ("fixed-cthr", args.fixed_cthr)]:
                if val is None:
                    raise ValueError(f"--{name} is required in fixed mode.")
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"--{name} must be in [0,1].")
            if args.fixed_beta is None:
                raise ValueError("--fixed-beta is required in fixed mode.")
            if args.fixed_beta <= 0:
                raise ValueError("--fixed-beta must be > 0.")
            if args.fixed_beta > 0.25:
                print("[warn] fixed-beta > 0.25 may be unstable for explicit 5-point diffusion.", file=sys.stderr)

        # Execute
        res = run_pipeline_2d(
            points_file=args.input,
            out_dir=args.outdir,

            # Fixed fast path
            FIXED_GRID=FIXED_GRID,
            FIXED_DENSE_THR=args.fixed_dense_thr,
            FIXED_BETA=args.fixed_beta,
            FIXED_CTHR=args.fixed_cthr,

            # Tuned path (Stage-A)
            TUNING=args.tuning,
            H_BOUNDS_REL=H_BOUNDS_REL,
            R_RANGE=R_RANGE_TUP,
            BO_N_CALLS=args.bo_n_calls,
            BO_OPT_WEIGHTS=bool(args.bo_opt_weights),
            BO_WEIGHT_FLOOR=args.bo_weight_floor,
            BO_WEIGHT_CEIL=args.bo_weight_ceil,

            # Grid suggester
            K_FOR_KNN=args.k_for_knn,
            ALPHA_FOR_KNN=args.alpha_for_knn,
            TARGET_OCC=args.target_occ,
            FD_BACKUP=bool(args.fd_backup),
            SWEEP_PCT=args.sweep_pct,
            MAX_BINS=args.max_bins,

            # Stage-A (grid)
            DENSE_QUANTILES=DENSE_QS,

            # Stage-B sweeps + iterations
            BETA_CANDIDATES=BETA_CANDIDATES,
            CTHR_VALUES=CTHR_VALUES,
            MAX_ITERS=args.max_iters,
            MIN_ITERS=args.min_iters,
            TOL=args.tol,
            CHECK_EVERY=args.check_every,

            # scoring defaults
            W_SIL=args.w_sil,
            W_DBI=args.w_dbi,
            W_COV=args.w_cov,
            K_MIN=args.k_min,
            K_MAX=args.k_max,

            # runtime
            PERIODIC_CCA=bool(args.periodic_cca),
            CONNECTIVITY=args.connectivity,
            MAKE_PLOTS=bool(args.make_plots),
            DO_STD_CCA=not bool(args.no_std_cca_plot),
        )

        # Print the same kind of summary as your example scripts
        _print_summary(res)

        if args.print_json:
            print("\n[Full result JSON]")
            print(json.dumps(res, indent=2))

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

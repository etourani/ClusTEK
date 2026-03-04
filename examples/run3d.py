#!/usr/bin/env python3
"""
3D benchmark driver for ClusTEK.

Usage:
  python examples/run3d_benchmark.py --dataset 9k
  python examples/run3d_benchmark.py --dataset 180k_tmid2
  python examples/run3d_benchmark.py --csv /path/to/snapshot.csv
  python examples/run3d_benchmark.py --csv /path/to/snapshot.csv --run-name myrun

Outputs (default):
  examples/outputs/3d/<run_name>/
    - benchmark_summary.csv
    - plots/cluster3d_*png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from clustek import ClusTEK3D
from clustek.core3d import DiffusionParams


# -----------------------------
# Repo helpers
# -----------------------------
def _repo_root_from_examples() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_md_csv(dataset: Optional[str], csv_arg: Optional[str]) -> Path:
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
        repo / "data" / "md" / f"{ds}.csv",
        repo / "data" / f"{ds}.csv",  # legacy fallback
    ]
    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not resolve MD dataset '{ds}'. Tried:\n" + "\n".join(str(x) for x in candidates)
    )


def default_out_dir(run_name: str) -> Path:
    here = Path(__file__).resolve().parent  # examples/
    out = here / "outputs" / "3d" / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


# -----------------------------
# Plotting
# -----------------------------
def plot_clusters_3d(
    df_sel: pd.DataFrame,
    clusters: dict[int, list[int]],
    *,
    xlo: float,
    xhi: float,
    ylo: float,
    yhi: float,
    zlo: float,
    zhi: float,
    outpath: str,
    title: str = "",
    max_clusters: Optional[int] = None,
    max_points_per_cluster: int = 8000,
    point_size: float = 3.0,
) -> None:
    if df_sel.empty or not clusters:
        return

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    items = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    if max_clusters is not None:
        items = items[:max_clusters]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab20", len(items))

    for idx, (cid, row_ids) in enumerate(items):
        sub = df_sel.iloc[row_ids]
        if len(sub) > max_points_per_cluster:
            sub = sub.sample(max_points_per_cluster, random_state=0)

        ax.scatter(
            sub["x"], sub["y"], sub["z"],
            s=point_size, alpha=0.85,
            color=cmap(idx),
            label=f"c{cid} (n={len(row_ids)})",
        )

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect([Lx, Ly, Lz])

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    if title:
        ax.set_title(title, fontsize=18)

    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None, help="MD dataset name in data/md/, e.g. 9k or 180k_tmid2")
    p.add_argument("--csv", type=str, default=None, help="Path to MD snapshot CSV")
    p.add_argument("--run-name", type=str, default=None, help="Output folder name under examples/outputs/3d/")
    p.add_argument("--out", type=str, default=None, help="Explicit output directory (overrides --run-name)")
    p.add_argument("--max-clusters", type=int, default=None, help="Plot only the largest N clusters (default: all)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = resolve_md_csv(args.dataset, args.csv)
    run_name = args.run_name or (args.dataset if args.dataset else csv_path.stem)

    out_dir = Path(args.out).expanduser().resolve() if args.out else default_out_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Sweep configs ---
    cell_sizes = [(0.8, 0.8, 0.8), (1.0, 1.0, 1.0), (1.2, 1.2, 1.2)]
    c_thrs = [0.3, 0.4, 0.5]

    diffusion = DiffusionParams(beta=0.1, iters=500, csel=0.10)

    # --- Atom baseline (once) ---
    base = ClusTEK3D(df, cell_size=(1.0, 1.0, 1.0), label_thr=0.4, label_col="c_label")
    t0 = time.time()
    atom_labels = base.cluster_atoms_connected_components(cutoff=1.5)
    t_atom = time.time() - t0
    print(f"Atom baseline runtime: {t_atom:.3f} s")
    n_atom_clusters = int(len(set(atom_labels)) - (1 if 0 in set(atom_labels) else 0))

    # bounds
    xlo = float(df["xlo"].iloc[0]); xhi = float(df["xhi"].iloc[0])
    ylo = float(df["ylo"].iloc[0]); yhi = float(df["yhi"].iloc[0])
    zlo = float(df["zlo"].iloc[0]); zhi = float(df["zhi"].iloc[0])

    rows = []
    for cs in cell_sizes:
        for thr in c_thrs:
            print(f"\ncell_size={cs}, C_thr={thr}")

            engine = ClusTEK3D(df, cell_size=cs, label_thr=thr, label_col="c_label")
            engine.particles_to_meshes()

            # ---- grid-only ----
            t0 = time.time()
            sel0 = engine.compute_filtered_cells(use_diffusion=False)
            cl0 = engine.cluster_cells(sel0) if len(sel0) else {}
            t_grid = time.time() - t0

            # ---- grid + diffusion ----
            t0 = time.time()
            sel1 = engine.compute_filtered_cells(use_diffusion=True, diffusion=diffusion)
            cl1 = engine.cluster_cells(sel1) if len(sel1) else {}
            t_diff = time.time() - t0

            if len(sel1) and len(cl1):
                outpng = out_dir / "plots" / f"cluster3d_diff_cs{cs[0]:.2f}_thr{thr:.2f}.png"
                plot_clusters_3d(
                    sel1,
                    cl1,
                    xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, zlo=zlo, zhi=zhi,
                    outpath=str(outpng),
                    title=f"Diff clusters | cs={cs}, C_thr={thr} | n_sel={len(sel1)} | n_cl={len(cl1)}",
                    max_clusters=args.max_clusters,
                )

            rows.append(
                {
                    "cell_size": cs,
                    "C_thr": thr,
                    "diffusion": asdict(diffusion),
                    "atom_time_s": t_atom,
                    "atom_n_clusters": n_atom_clusters,
                    "grid_time_s": t_grid,
                    "grid_n_selected": int(len(sel0)),
                    "grid_n_clusters": int(len(cl0)),
                    "diff_time_s": t_diff,
                    "diff_n_selected": int(len(sel1)),
                    "diff_n_clusters": int(len(cl1)),
                }
            )

    out = pd.DataFrame(rows)
    out_csv = out_dir / "benchmark_summary.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(f"Wrote 3D plots into {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DiffusionParams:
    """Parameters for diffusion imputation on 3D mesh grids.

    beta:
        Explicit Euler diffusion coefficient (6-neighbor Laplacian). Conservative guideline:
        beta <= 1/6 for stability (unit spacing). Using smaller values (0.02--0.10) is typical.
    iters:
        Number of explicit diffusion iterations.
    csel:
        Post-diffusion selection threshold for admitting sparse/unsampled cells.
        Dense pre-diffusion cells are always retained.
    """
    beta: float = 0.10
    iters: int = 500
    csel: float = 0.10


class ClusTEK3D:
    """
    ClusTEK 3D grid clustering engine.

    Per snapshot:
      1) Bin atoms into a regular 3D grid.
      2) Compute cell-averaged scalar field C^(0) (e.g., mean crystallinity per cell).
      3) Build masks (MD strategy):
           - dense occupied cells: C^(0) >= C_thr
           - sparse occupied cells: 0 < C^(0) < C_thr
           - true-melt occupied cells: C^(0) == 0 (within tol)
           - unsampled cells: no atoms in cell
      4) Diffuse from dense into (sparse + unsampled) only.
         Dense cells are clamped to C^(0); true-melt occupied cells clamped to 0.
      5) Select cells for clustering:
           - keep all dense occupied cells
           - admit sparse/unsampled cells only if C^(diff) >= C_sel
      6) Cluster selected cells via periodic KDTree connectivity in *physical* space.

    Notes
    -----
    - Input dataframe must include columns: x, y, z, label_col,
      plus xlo/xhi/ylo/yhi/zlo/zhi for periodic box bounds.
    - Assumes per-atom scalar is bounded in [0,1] (true for 0/1 crystallinity labels).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        cell_size: Tuple[float, float, float],
        label_thr: float = 0.40,
        label_col: str = "c_label",
        bounds_cols: Tuple[str, str, str, str, str, str] = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"),
        zero_tol: float = 1e-12,
    ) -> None:
        self.data = data.copy()
        self.cell_size = tuple(float(v) for v in cell_size)
        self.cthr = float(label_thr)
        self.label_col = label_col
        self.bounds_cols = bounds_cols
        self.zero_tol = float(zero_tol)

        required = {"x", "y", "z", self.label_col, *self.bounds_cols}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        xlo, xhi, ylo, yhi, zlo, zhi = (float(self.data[c].iloc[0]) for c in self.bounds_cols)
        self.xlo, self.xhi = xlo, xhi
        self.ylo, self.yhi = ylo, yhi
        self.zlo, self.zhi = zlo, zhi
        self.Lx, self.Ly, self.Lz = (xhi - xlo), (yhi - ylo), (zhi - zlo)

        self.mesh_df: Optional[pd.DataFrame] = None
        self.grid_shape: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def particles_to_meshes(self) -> pd.DataFrame:
        """Assign each particle to a mesh cell and compute per-cell mean label."""
        dx, dy, dz = self.cell_size
        nx = int(np.ceil(self.Lx / dx))
        ny = int(np.ceil(self.Ly / dy))
        nz = int(np.ceil(self.Lz / dz))
        self.grid_shape = (nx, ny, nz)

        # periodic indexing into [0, n)
        xi = np.floor((self.data["x"].to_numpy() - self.xlo) / dx).astype(int) % nx
        yi = np.floor((self.data["y"].to_numpy() - self.ylo) / dy).astype(int) % ny
        zi = np.floor((self.data["z"].to_numpy() - self.zlo) / dz).astype(int) % nz

        d = self.data.copy()
        d["xi"], d["yi"], d["zi"] = xi, yi, zi

        cell_means = (
            d.groupby(["xi", "yi", "zi"], as_index=False)[self.label_col]
            .mean()
            .rename(columns={self.label_col: "label_mean"})
        )

        # cell centers in physical coordinates
        cell_means["x"] = self.xlo + (cell_means["xi"] + 0.5) * dx
        cell_means["y"] = self.ylo + (cell_means["yi"] + 0.5) * dy
        cell_means["z"] = self.zlo + (cell_means["zi"] + 0.5) * dz

        self.mesh_df = cell_means
        return cell_means

    # ------------------------------------------------------------------
    # Internal helpers: field + masks
    # ------------------------------------------------------------------
    def _build_field_and_masks(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the pre-diffusion field C^(0) and masks.

        Returns
        -------
        field0:
            (nx, ny, nz) array. Occupied cells filled by label_mean; unsampled are 0 placeholder.
        dense_mask:
            occupied & (field0 >= cthr)
        sparse_mask:
            occupied & (0 < field0 < cthr)
        melt_mask:
            occupied & (field0 == 0 within tol)
        unsampled_mask:
            ~occupied
        """
        if self.mesh_df is None or self.grid_shape is None:
            raise RuntimeError("Call particles_to_meshes() before building fields.")

        nx, ny, nz = self.grid_shape
        field0 = np.zeros((nx, ny, nz), dtype=float)
        occupied = np.zeros((nx, ny, nz), dtype=bool)

        for row in self.mesh_df.itertuples(index=False):
            i, j, k = int(row.xi), int(row.yi), int(row.zi)
            val = float(row.label_mean)
            # enforce bounded assumption
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            field0[i, j, k] = val
            occupied[i, j, k] = True

        unsampled_mask = ~occupied
        melt_mask = occupied & (field0 <= self.zero_tol)
        dense_mask = occupied & (field0 >= self.cthr)
        sparse_mask = occupied & (~dense_mask) & (~melt_mask)

        return field0, dense_mask, sparse_mask, melt_mask, unsampled_mask

    # ------------------------------------------------------------------
    # Diffusion / imputation (masked)
    # ------------------------------------------------------------------
    def diffuse_grid(self, diffusion: DiffusionParams) -> Dict[str, np.ndarray]:
        """
        Run masked diffusion on the 3D field with periodic boundary conditions.

        Update only:
            sparse occupied cells + unsampled cells

        Clamp each iteration:
            dense occupied cells -> fixed to field0
            true-melt occupied cells -> fixed to 0
            overall -> clipped to [0,1]
        """
        if self.mesh_df is None or self.grid_shape is None:
            raise RuntimeError("Call particles_to_meshes() before diffuse_grid().")

        field0, dense_mask, sparse_mask, melt_mask, unsampled_mask = self._build_field_and_masks()
        field = field0.copy()

        beta = float(diffusion.beta)
        iters = int(diffusion.iters)
        if beta <= 0:
            raise ValueError("Diffusion beta must be > 0.")
        # allow expert use; no hard error here

        update_mask = sparse_mask | unsampled_mask

        for _ in range(iters):
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                + np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
                - 6.0 * field
            )

            field[update_mask] = field[update_mask] + beta * lap[update_mask]
            np.clip(field, 0.0, 1.0, out=field)

            # re-impose constraints
            field[dense_mask] = field0[dense_mask]
            field[melt_mask] = 0.0

        return {
            "field0": field0,
            "field_final": field,
            "dense_mask": dense_mask,
            "sparse_mask": sparse_mask,
            "melt_mask": melt_mask,
            "unsampled_mask": unsampled_mask,
            "update_mask": update_mask,
        }

    # ------------------------------------------------------------------
    # Cell selection
    # ------------------------------------------------------------------
    def compute_filtered_cells(
        self,
        *,
        use_diffusion: bool = True,
        diffusion: DiffusionParams = DiffusionParams(),
    ) -> pd.DataFrame:
        """
        Return selected cells for clustering.

        - no diffusion: keep occupied cells with label_mean >= C_thr
        - with diffusion:
            keep all dense occupied cells,
            plus (sparse OR unsampled) cells if diffused field >= C_sel
        """
        if self.mesh_df is None:
            self.particles_to_meshes()
        assert self.mesh_df is not None
        if self.grid_shape is None:
            raise RuntimeError("Internal error: grid_shape missing after particles_to_meshes().")

        if not use_diffusion:
            sel = self.mesh_df[self.mesh_df["label_mean"] >= self.cthr].copy()
            sel.rename(columns={"label_mean": "label_used"}, inplace=True)
            sel["selected_from"] = "dense"
            return sel

        out = self.diffuse_grid(diffusion)
        field_final = out["field_final"]
        dense_mask = out["dense_mask"]
        sparse_mask = out["sparse_mask"]
        unsampled_mask = out["unsampled_mask"]

        csel = float(diffusion.csel)

        selected_mask = dense_mask.copy()
        admit_mask = (sparse_mask | unsampled_mask) & (field_final >= csel)
        selected_mask |= admit_mask

        idx = np.argwhere(selected_mask)
        if idx.size == 0:
            return pd.DataFrame(columns=["xi", "yi", "zi", "x", "y", "z", "label_used", "label_raw", "selected_from"])

        dx, dy, dz = self.cell_size

        # for label_raw lookup (occupied only)
        occupied_lookup = {
            (int(r.xi), int(r.yi), int(r.zi)): float(r.label_mean)
            for r in self.mesh_df.itertuples(index=False)
        }

        rows: List[Dict[str, float]] = []
        for i, j, k in idx:
            i, j, k = int(i), int(j), int(k)
            x = self.xlo + (i + 0.5) * dx
            y = self.ylo + (j + 0.5) * dy
            z = self.zlo + (k + 0.5) * dz

            if dense_mask[i, j, k]:
                src = "dense"
            elif sparse_mask[i, j, k]:
                src = "sparse_imputed"
            elif unsampled_mask[i, j, k]:
                src = "unsampled_imputed"
            else:
                src = "other"

            rows.append(
                {
                    "xi": i,
                    "yi": j,
                    "zi": k,
                    "x": x,
                    "y": y,
                    "z": z,
                    "label_used": float(field_final[i, j, k]),
                    "label_raw": float(occupied_lookup.get((i, j, k), np.nan)),
                    "selected_from": src,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Clustering on selected cells (PERIODIC)
    # ------------------------------------------------------------------
    def cluster_cells(
        self,
        selected_cells: pd.DataFrame,
        *,
        radius: Optional[float] = None,
    ) -> Dict[int, List[int]]:
        """
        Cluster selected cells via periodic connectivity in physical space.

        We cluster using the physical cell centers (x,y,z) with a periodic KDTree:
            cKDTree(coords_shifted, boxsize=(Lx,Ly,Lz))

        IMPORTANT:
        SciPy requires coords to be in [0,L) when boxsize is used,
        so we shift by (xlo,ylo,zlo) before building the tree.

        Returns
        -------
        dict cluster_id -> list of row indices (into selected_cells)
        """
        if selected_cells.empty:
            return {}

        if radius is None:
            # default: face-adjacent at ~1 cell in physical space.
            # For cubic cells, radius ~ cell_size works; we use min(dx,dy,dz)*1.01
            dx, dy, dz = self.cell_size
            radius = 1.01 * min(dx, dy, dz)

        coords = selected_cells[["x", "y", "z"]].to_numpy(dtype=float)

        # ---- shift into [0,L) for periodic KDTree ----
        coords[:, 0] -= self.xlo
        coords[:, 1] -= self.ylo
        coords[:, 2] -= self.zlo

        # Numerical safety: wrap into [0,L)
        coords[:, 0] = np.mod(coords[:, 0], self.Lx)
        coords[:, 1] = np.mod(coords[:, 1], self.Ly)
        coords[:, 2] = np.mod(coords[:, 2], self.Lz)

        tree = cKDTree(coords, boxsize=(self.Lx, self.Ly, self.Lz))
        pairs = tree.query_pairs(r=float(radius))

        n = len(coords)
        parent = np.arange(n, dtype=int)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in pairs:
            union(i, j)

        clusters: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        # reindex 1..K
        remap = {old: new for new, old in enumerate(sorted(clusters.keys()), start=1)}
        return {remap[k]: v for k, v in clusters.items()}

    # ------------------------------------------------------------------
    # Atom-based reference clustering (PERIODIC)
    # ------------------------------------------------------------------
    def cluster_atoms_connected_components(
        self,
        *,
        cutoff: float = 1.5,
        label_filter_thr: Optional[float] = None,
    ) -> np.ndarray:
        """
        Atom-based connected components under periodic boundary conditions.

        Returns
        -------
        labels : (N,) int
            0 -> background
            1..K -> component id among atoms with label >= label_filter_thr
        """
        if label_filter_thr is None:
            label_filter_thr = self.cthr

        pts = self.data[["x", "y", "z"]].to_numpy(dtype=float)
        labels = np.zeros(len(pts), dtype=int)

        mask = self.data[self.label_col].to_numpy(dtype=float) >= float(label_filter_thr)
        if not np.any(mask):
            return labels

        pts_f = pts[mask].copy()

        # ---- shift into [0,L) for periodic KDTree ----
        pts_f[:, 0] -= self.xlo
        pts_f[:, 1] -= self.ylo
        pts_f[:, 2] -= self.zlo

        pts_f[:, 0] = np.mod(pts_f[:, 0], self.Lx)
        pts_f[:, 1] = np.mod(pts_f[:, 1], self.Ly)
        pts_f[:, 2] = np.mod(pts_f[:, 2], self.Lz)

        tree = cKDTree(pts_f, boxsize=(self.Lx, self.Ly, self.Lz))
        pairs = tree.query_pairs(r=float(cutoff))

        n = len(pts_f)
        parent = np.arange(n, dtype=int)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in pairs:
            union(i, j)

        comp = np.zeros(n, dtype=int)
        roots: Dict[int, int] = {}
        next_id = 1
        for i in range(n):
            r = find(i)
            if r not in roots:
                roots[r] = next_id
                next_id += 1
            comp[i] = roots[r]

        labels[np.where(mask)[0]] = comp
        return labels
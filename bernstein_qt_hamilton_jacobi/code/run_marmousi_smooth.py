#!/usr/bin/env python3
"""Reproduce the Marmousi first-arrival benchmark in the manuscript.

The script deliberately separates branch selection from high-order correction:
1. A first-order causal fast sweep on the native Marmousi grid is used only as
   the error reference.
2. A much coarser causal sweep selects the first-arrival branch.
3. A global C0 Bernstein p=7 correction is continued through six meshes.

The Marmousi velocity arrays are not redistributed with the source package.
Pass the 376 x 1151 CSV/TXT file explicitly.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from plot_style import ELEGANT_CMAP, LINE_BLUE, LINE_RED, MESH_GRAY, LIGHT_GRID

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from marmousi_bb_core import (  # noqa: E402
    MarmProblem,
    RadialFactor,
    RectFastFactored,
    bernstein_values,
    fast_sweep,
    make_seed,
    multiindices2,
)

DOMAIN_X_KM = 9.2
DOMAIN_Z_KM = 3.0
SOURCE = np.array([4.6, 0.0], dtype=float)
MESH_SEQUENCE = [(6, 2), (12, 4), (18, 6), (24, 8), (30, 10), (36, 12)]


def load_velocity(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.loadtxt(path, delimiter=",") / 1000.0  # m/s -> km/s
    if v.shape != (376, 1151):
        raise ValueError(f"expected a 376 x 1151 Marmousi array, got {v.shape}")
    z = np.linspace(0.0, DOMAIN_Z_KM, v.shape[0])
    x = np.linspace(0.0, DOMAIN_X_KM, v.shape[1])
    return v, x, z


def causal_reference(v: np.ndarray, x: np.ndarray, z: np.ndarray):
    hx = float(x[1] - x[0])
    hz = float(z[1] - z[0])
    ix = int(np.argmin(np.abs(x - SOURCE[0])))
    iz = int(np.argmin(np.abs(z - SOURCE[1])))
    T, ncycles, maxchange = fast_sweep(1.0 / v, hx, hz, iz, ix, 100, 1e-12)
    return T, ncycles, maxchange


def eval_pts(model: RectFastFactored, coeff: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    (xmin, xmax), (zmin, zmax) = model.bounds
    hx = (xmax - xmin) / model.nx
    hz = (zmax - zmin) / model.nz
    ii = np.minimum(np.maximum(((pts[:, 0] - xmin) / hx).astype(int), 0), model.nx - 1)
    jj = np.minimum(np.maximum(((pts[:, 1] - zmin) / hz).astype(int), 0), model.nz - 1)
    rr = (pts[:, 0] - (xmin + ii * hx)) / hx
    tt = (pts[:, 1] - (zmin + jj * hz)) / hz
    lower = tt <= rr
    K = 2 * (jj * model.nx + ii) + (~lower).astype(int)
    lam = np.empty((len(pts), 3), dtype=float)
    lam[lower] = np.column_stack(
        [1.0 - rr[lower], rr[lower] - tt[lower], tt[lower]]
    )
    upper = ~lower
    lam[upper] = np.column_stack(
        [1.0 - tt[upper], rr[upper], tt[upper] - rr[upper]]
    )
    out = np.empty(len(pts), dtype=float)
    for kk in np.unique(K):
        mask = K == kk
        out[mask] = bernstein_values(model.p, lam[mask]) @ coeff[model.asm.l2g[kk]]
    return out + model.factor.u(pts)


def transfer(old: RectFastFactored, old_coeff: np.ndarray, new: RectFastFactored) -> np.ndarray:
    """Transfer by exact interpolation at the degree-p Bernstein domain points."""
    inds = multiindices2(new.p)
    lam = np.asarray(inds, dtype=float) / new.p
    A = bernstein_values(new.p, lam)
    out = np.zeros(len(new.asm.keys), dtype=float)
    cnt = np.zeros(len(new.asm.keys), dtype=float)
    for K, tri in enumerate(new.tris):
        V = new.verts[tri]
        pts = lam @ V
        values = eval_pts(old, old_coeff, pts) - new.factor.u(pts)
        ck = la.solve(A, values, check_finite=False)
        gids = new.asm.l2g[K]
        np.add.at(out, gids, ck)
        np.add.at(cnt, gids, 1.0)
    out /= np.maximum(cnt, 1.0)
    out[new.fixed] = new.fixed_vals
    return out


def run_sweep(v: np.ndarray, x: np.ndarray, z: np.ndarray, Tref: np.ndarray, p: int = 7):
    # Coarse causal selector: 73 x 25 points.
    xc, zc, vc, Tseed, seed_cycles = make_seed(v, x, z, 72, 24, tuple(SOURCE))
    problem = MarmProblem(v, x, z, Tseed, xc, zc)
    s0 = 1.0 / float(v[0, np.argmin(np.abs(x - SOURCE[0]))])
    factor = RadialFactor(SOURCE, s0)
    den = float(np.sqrt(np.mean(Tref * Tref)))

    # Error of the branch selector itself, interpolated to the native grid.
    from scipy.interpolate import RegularGridInterpolator

    seed_interp = RegularGridInterpolator((zc, xc), Tseed, bounds_error=False, fill_value=None)
    X, Z = np.meshgrid(x, z)
    Tseed_fine = seed_interp(np.column_stack([Z.ravel(), X.ravel()])).reshape(Tref.shape)
    seed_err = Tseed_fine - Tref
    selector = {
        "relrmse": float(np.sqrt(np.mean(seed_err * seed_err)) / den),
        "mae": float(np.mean(np.abs(seed_err))),
        "linf": float(np.max(np.abs(seed_err))),
        "cycles": int(seed_cycles),
    }

    rows = []
    old = None
    old_coeff = None
    best = None
    best_model = None
    best_coeff = None
    best_field = None
    for nx, nz in MESH_SEQUENCE:
        model = RectFastFactored(nx, nz, p, problem, factor, qorder=10)
        initial = None if old is None else transfer(old, old_coeff, model)
        coeff, sol, seconds = model.solve(initial, maxit=5, verbose=0, lsmr_tol=2e-5)
        field = model.eval_grid(coeff, x, z)
        err = field - Tref
        R, _ = model.state(coeff[model.free])
        row = {
            "nx": nx,
            "nz": nz,
            "ntri": len(model.tris),
            "dofs": len(coeff),
            "relrmse": float(np.sqrt(np.mean(err * err)) / den),
            "mae": float(np.mean(np.abs(err))),
            "linf": float(np.max(np.abs(err))),
            "residual_rms": float(np.linalg.norm(R) / math.sqrt(len(R))),
            "time": float(seconds),
            "nfev": int(sol.nfev),
        }
        rows.append(row)
        if best is None or row["relrmse"] < best["relrmse"]:
            best = row
            best_model = model
            best_coeff = coeff.copy()
            best_field = field.copy()
        print(row, flush=True)
        old, old_coeff = model, coeff
    return selector, rows, best, best_model, best_coeff, best_field


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_figures(v, x, z, Tref, best, best_field, rows, figure_dir: Path):
    figure_dir.mkdir(parents=True, exist_ok=True)
    extent = [x[0], x[-1], z[-1], z[0]]
    err_ms = 1000.0 * (best_field - Tref)

    fig, axs = plt.subplots(2, 2, figsize=(10.6, 5.5), constrained_layout=True)
    vmid = 0.5 * (float(np.nanmin(v)) + float(np.nanmax(v)))
    im = axs[0, 0].imshow(v, extent=extent, aspect="auto", cmap=ELEGANT_CMAP, vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)))
    axs[0, 0].plot(SOURCE[0], SOURCE[1], "rx", ms=6, mew=1.3)
    axs[0, 0].set_title("Smoothed Marmousi velocity")
    axs[0, 0].set_ylabel("Depth (km)")
    fig.colorbar(im, ax=axs[0, 0], label="km/s")

    im = axs[0, 1].imshow(Tref, extent=extent, aspect="auto", cmap=ELEGANT_CMAP, vmin=float(np.nanmin(Tref)), vmax=float(np.nanmax(Tref)))
    axs[0, 1].plot(SOURCE[0], SOURCE[1], "rx", ms=6, mew=1.3)
    axs[0, 1].set_title("Native causal reference")
    fig.colorbar(im, ax=axs[0, 1], label="Travel time (s)")

    im = axs[1, 0].imshow(best_field, extent=extent, aspect="auto", cmap=ELEGANT_CMAP, vmin=float(np.nanmin(Tref)), vmax=float(np.nanmax(Tref)))
    axs[1, 0].plot(SOURCE[0], SOURCE[1], "rx", ms=6, mew=1.3)
    axs[1, 0].set_title(f"BB correction: p=7, {best['ntri']} triangles")
    axs[1, 0].set_xlabel("Position (km)")
    axs[1, 0].set_ylabel("Depth (km)")
    fig.colorbar(im, ax=axs[1, 0], label="Travel time (s)")

    lim = max(abs(float(np.nanmin(err_ms))), abs(float(np.nanmax(err_ms))))
    im = axs[1, 1].imshow(err_ms, extent=extent, aspect="auto", cmap=ELEGANT_CMAP, vmin=-lim, vmax=lim)
    axs[1, 1].plot(SOURCE[0], SOURCE[1], "kx", ms=6, mew=1.2)
    axs[1, 1].set_title("BB - causal reference")
    axs[1, 1].set_xlabel("Position (km)")
    fig.colorbar(im, ax=axs[1, 1], label="Error (ms)")
    fig.savefig(figure_dir / "marmousi_smooth_bb_panels.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "marmousi_smooth_bb_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ntri = np.asarray([r["ntri"] for r in rows], float)
    rel = np.asarray([r["relrmse"] for r in rows], float)
    res = np.asarray([r["residual_rms"] for r in rows], float)

    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    ax.semilogy(ntri, rel, "o-", color=LINE_BLUE)
    ax.set_xlabel("Triangles")
    ax.set_ylabel("Relative RMS travel-time error")
    ax.grid(True, which="both", alpha=LIGHT_GRID)
    fig.tight_layout()
    fig.savefig(figure_dir / "marmousi_h_convergence.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "marmousi_h_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.25))
    ax.semilogy(ntri, res, "o-", color=LINE_RED)
    ax.set_xlabel("Triangles")
    ax.set_ylabel("Hamiltonian moment residual RMS")
    ax.grid(True, which="both", alpha=LIGHT_GRID)
    fig.tight_layout()
    fig.savefig(figure_dir / "marmousi_residual_convergence.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "marmousi_residual_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocity", required=True, type=Path,
                    help="smoothed 376x1151 Marmousi CSV/TXT (values in m/s)")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_marmousi")
    ap.add_argument("--raw-velocity", type=Path, default=None,
                    help="optional unsmoothed 376x1151 Marmousi array for the robustness sweep")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"

    v, x, z = load_velocity(args.velocity)
    Tref, cycles, maxchange = causal_reference(v, x, z)
    print(f"native causal reference: cycles={cycles}, maxchange={maxchange:.3e}, Tmax={Tref.max():.6f} s")
    selector, rows, best, _, _, best_field = run_sweep(v, x, z, Tref, p=7)
    write_csv(rows, args.output_dir / "marmousi_smooth_h_sweep_p7.csv")
    np.savez_compressed(args.output_dir / "marmousi_native_reference_smooth.npz",
                        T=Tref, v=v, x=x, z=z, source=SOURCE)
    make_figures(v, x, z, Tref, best, best_field, rows, figure_dir)

    summary = args.output_dir / "marmousi_summary.txt"
    summary.write_text(
        "Marmousi smooth benchmark\n"
        f"coarse selector rel RMS: {selector['relrmse']:.8e}\n"
        f"coarse selector Linf (s): {selector['linf']:.8e}\n"
        f"best triangles: {best['ntri']}\n"
        f"best DOFs: {best['dofs']}\n"
        f"best rel RMS: {best['relrmse']:.8e}\n"
        f"best MAE (s): {best['mae']:.8e}\n"
        f"best Linf (s): {best['linf']:.8e}\n"
    )

    if args.raw_velocity is not None:
        vr, xr, zr = load_velocity(args.raw_velocity)
        Tr, _, _ = causal_reference(vr, xr, zr)
        _, raw_rows, _, _, _, _ = run_sweep(vr, xr, zr, Tr, p=7)
        write_csv(raw_rows, args.output_dir / "marmousi_raw_h_sweep_p7.csv")

    print(summary.read_text())


if __name__ == "__main__":
    main()

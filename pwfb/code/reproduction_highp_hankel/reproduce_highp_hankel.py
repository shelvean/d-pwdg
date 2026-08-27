#!/usr/bin/env python3
"""
Reproduce the high-p plane-wave approximation experiment for the centered
outgoing Hankel field in the Adaptive Local Representations manuscript.

Experiment
----------
u(x) = H_0^{(1)}(k |x|),   k = 16,
on one of the eight congruent annular sectors 0.5 < r < 1.
The local trace disk has center x_K=(0.75,0) and radius h=0.44.

For p equispaced plane waves the Cauchy-trace least-squares problem is solved
analytically by modal residue classes.  No ill-conditioned p-by-p normal
equations are solved.

Three conditioning quantities are treated distinctly:
  1. kappa_tr_raw:
     cond_2 of the raw plane-wave Cauchy-trace Gram matrix.
  2. kappa_tr_orth:
     cond_2 after trace-Riesz orthonormalization.  It is independently validated by
     1024-point periodic boundary quadrature at 90 decimal digits.
  3. This script does NOT compute the global graph-Riesz PWDG condition
     number kappa_GR; that belongs to the full DG solve.

The broken L2 error is evaluated from the *residual modal expansion*, not by
subtracting two nearly equal O(1) fields.  This is essential near 1e-16.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import jv, hankel1


P_LIST = [49, 57, 65, 73, 81, 89, 97]
K = mp.mpf("16")
H = mp.mpf("0.44")
D = mp.mpf("0.75")
M_MODAL = 220
N_GAUSS = 160
N_BOUNDARY = 1024


def J(m: int, x):
    return mp.besselj(m, x)


def Jp(m: int, x):
    return (J(m - 1, x) - J(m + 1, x)) / 2


def H1(m: int, x):
    return mp.hankel1(m, x)


def build_modal_data(dps: int = 70):
    mp.mp.dps = dps
    k, h, D = mp.mpf("16"), mp.mpf("0.44"), mp.mpf("0.75")
    kh = k * h
    ms = list(range(-M_MODAL, M_MODAL + 1))

    tau2 = {
        m: 2 * mp.pi * h * k * (J(m, kh) ** 2 + Jp(m, kh) ** 2)
        for m in ms
    }

    # Local center x_K=(D,0), source at the origin.
    # The source-center vector has polar angle pi.
    a = {m: H1(m, k * D) * ((-1) ** m) for m in ms}

    # Remove the i^m factor in the Jacobi-Anger expansion.
    y = {m: (mp.j ** (-m)) * a[m] for m in ms}
    total_trace_energy = mp.fsum(tau2[m] * abs(a[m]) ** 2 for m in ms)
    return ms, tau2, a, y, total_trace_energy


def projection_for_p(p, ms, tau2, a, y, total_trace_energy):
    C = []
    pred = {}
    for s in range(p):
        inds = [m for m in ms if m % p == s]
        den = mp.fsum(tau2[m] for m in inds)
        num = mp.fsum(tau2[m] * y[m] for m in inds)
        Cs = num / den
        C.append(Cs)
        for m in inds:
            pred[m] = (mp.j ** m) * Cs

    err2 = mp.fsum(
        tau2[m] * abs(a[m] - pred[m]) ** 2 for m in ms
    )
    trace_error = mp.sqrt(err2 / total_trace_energy)

    # The circulant trace-Gram eigenvalues from the exact symbol.
    lam = []
    for s in range(p):
        inds = [m for m in ms if (-m) % p == s]
        lam.append(p * mp.fsum(tau2[m] for m in inds))
    raw_cond = max(lam) / min(lam)

    return C, pred, trace_error, lam, raw_cond


def broken_l2_from_residual(p, pred, a, ms):
    """
    Relative L2 error over one annular sector centered at angle zero.
    By rotational symmetry this equals the broken relative L2 error over all
    eight sectors.

    The error field is evaluated directly from its local Fourier-Bessel
    residual coefficients.  This avoids catastrophic cancellation at p=97.
    """
    gx, gw = leggauss(N_GAUSS)
    r = 0.75 + 0.25 * gx
    rw = 0.25 * gw
    theta = (np.pi / 8.0) * gx
    tw = (np.pi / 8.0) * gw

    RR, TT = np.meshgrid(r, theta, indexing="ij")
    W = np.outer(rw, tw) * RR

    X = RR * np.cos(TT)
    Y = RR * np.sin(TT)
    DX = X - 0.75
    DY = Y
    rho = np.hypot(DX, DY)
    psi = np.arctan2(DY, DX)

    exact = hankel1(0, 16.0 * RR)
    denominator = np.sum(W * np.abs(exact) ** 2)

    # Modes |m|>160 are numerically irrelevant on this local disk; retaining
    # the full M_MODAL range is harmless but scipy may under/overflow in
    # extreme negative-order Bessel evaluations.  The residual coefficients
    # beyond |m|=150 are far below the requested accuracy.
    use_ms = [m for m in ms if abs(m) <= 150]
    residual = np.array(
        [complex(a[m] - pred[m]) for m in use_ms], dtype=np.complex128
    )
    basis = np.array(
        [jv(m, 16.0 * rho) * np.exp(1j * m * psi) for m in use_ms]
    )
    error_field = np.tensordot(residual, basis, axes=(0, 0))
    numerator = np.sum(W * np.abs(error_field) ** 2)
    return float(np.sqrt(numerator / denominator))


def independent_orth_condition(p, lam_construct, boundary_points=N_BOUNDARY):
    """
    Independently validate the trace orthonormalization.

    The trace-Riesz orthonormalization uses modal/circulant eigenvalues computed at 70 digits.
    The Gram eigenvalues are then recomputed from the physical boundary
    integral using an unrelated 1024-point periodic quadrature at 90 digits.
    The ratios lambda_quad/lambda_construct are the eigenvalues of the
    independently recomputed orthonormalized Gram matrix.
    """
    # Freeze the construction values at 70-digit precision before raising dps.
    lam70 = [mp.mpf(mp.nstr(x, 75)) for x in lam_construct]

    mp.mp.dps = 90
    k, h = mp.mpf("16"), mp.mpf("0.44")
    kh = k * h
    Q = int(boundary_points)
    tq = [2 * mp.pi * q / Q for q in range(Q)]
    ct = [mp.cos(t) for t in tq]
    st = [mp.sin(t) for t in tq]

    # First row of the raw Gram matrix, evaluated directly in physical trace
    # coordinates. For theta_0=0 and theta_l=delta:
    # G_0l = kh ∫[1+cos(t)cos(t-delta)]
    #          exp(i kh[cos(t)-cos(t-delta)]) dt.
    row = []
    for ell in range(p):
        delta = 2 * mp.pi * ell / p
        cd, sd = mp.cos(delta), mp.sin(delta)
        s = mp.mpc(0)
        for q in range(Q):
            c0 = ct[q]
            c1 = ct[q] * cd + st[q] * sd
            s += (1 + c0 * c1) * mp.e ** (mp.j * kh * (c0 - c1))
        row.append(k * h * (2 * mp.pi / Q) * s)

    lam_quad = []
    for s in range(p):
        v = mp.fsum(
            row[ell] * mp.e ** (-2 * mp.pi * mp.j * s * ell / p)
            for ell in range(p)
        )
        lam_quad.append(mp.re(v))

    ratios = [lam_quad[s] / lam70[s] for s in range(p)]
    orth_cond = max(ratios) / min(ratios)
    return orth_cond


def run(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ms, tau2, a, y, Etr = build_modal_data(70)

    rows = []
    for p in P_LIST:
        # Reset construction precision for every row.
        mp.mp.dps = 70
        C, pred, trace_err, lam, raw_cond = projection_for_p(
            p, ms, tau2, a, y, Etr
        )
        l2_err = broken_l2_from_residual(p, pred, a, ms)
        orth_cond = independent_orth_condition(p, lam, N_BOUNDARY)

        rows.append({
            "p": p,
            "relative_trace_error": float(trace_err),
            "broken_relative_L2_error": l2_err,
            "raw_trace_gram_condition": float(raw_cond),
            "orth_trace_gram_condition_minus_1":
                float(orth_cond - 1),
        })

        print(
            f"p={p:3d}  "
            f"Etr={mp.nstr(trace_err, 10):>13s}  "
            f"EL2={l2_err:.10e}  "
            f"kraw={mp.nstr(raw_cond, 10):>13s}  "
            f"korth=1+{mp.nstr(orth_cond-1, 10)}"
        )

    df = pd.DataFrame(rows)
    csv_path = outdir / "highp_hankel_reproduced.csv"
    df.to_csv(csv_path, index=False)

    # Manuscript-ready LaTeX rows.
    tex = []
    for r in rows:
        tex.append(
            f"{r['p']} & "
            f"${r['relative_trace_error']:.2e}$ & "
            f"${r['broken_relative_L2_error']:.2e}$ & "
            f"${r['raw_trace_gram_condition']:.2e}$ & "
            f"$1+{r['orth_trace_gram_condition_minus_1']:.2e}$ \\\\"
        )
    (outdir / "table_rows.tex").write_text("\n".join(tex) + "\n")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    run(args.out)

"""Parallel scaling of the elementwise ESPRIT selection stage.
Single-file version for Google Colab: paste into one cell and run, or
upload and execute with  %run esprit_parallel_colab.py

What it does.  On three refinement levels of the annulus 0.5 < r < 1
(8, 32, 128 elements) with the Herglotz-DtN reference field at k = 16,
every element independently ESPRIT-initializes candidate ray counts
q_PW = 0,...,4 at local budget p = 21, polishes the angles by
Nelder-Mead in the weighted Cauchy-trace tail norm, and scores the
(q_PW, q_FB) split.  Elements are uncoupled, so the stage is
embarrassingly parallel; wall time is measured against worker count
with one BLAS thread per worker, and every parallel run is asserted to
reproduce the serial selections exactly.  At the end the script prints
the LaTeX rows for tab:esprit-parallel and the filled-in sentence for
the manuscript paragraph, and writes esprit_parallel.csv.

Colab note: report the printed cpu_count with the numbers.  Free-tier
Colab usually exposes 2 vCPUs; a high-RAM/paid runtime or your own
workstation gives more cores and a more informative table.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.optimize import minimize

# ------------------------------------------------------------- parameters
k = 16.0
a, R = 0.5, 1.0
NdtN = 70
P_BUDGET = 21          # local budget p
Q_MAX = 4              # candidate ray counts q_PW = 0..Q_MAX
NS_ESPRIT = 24         # consecutive positive-tail modal samples
LEVELS = [(8, 1), (16, 2), (32, 4)]
WORKERS = None         # e.g. [1, 2, 4, 8]; None = auto up to cpu_count()

# ------------------------------------------------------------ exact field
def herglotz_dtn_coeffs():
    nq = 4096
    phi = 2 * np.pi * np.arange(nq) / nq
    g = np.exp(1.55 * np.cos(phi - 0.48) + 0.22 * np.cos(2 * phi + 0.35)) \
        * np.exp(1j * (0.28 * np.sin(3 * phi + 0.20)
                       + 0.10 * np.cos(5 * phi - 0.40)))
    ghat = np.fft.fft(g) / nq
    ms = np.arange(-NdtN, NdtN + 1)
    gm = ghat[ms % nq]
    return ms, 2 * np.pi * (1j ** ms) * jv(ms, k * a) * gm / hankel1(ms, k * a)

MS, CM = herglotz_dtn_coeffs()

def u_exact(x):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    return (hankel1(MS[None, :], k * r[:, None])
            * np.exp(1j * np.outer(th, MS))) @ CM

def grad_exact(x):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    E = np.exp(1j * np.outer(th, MS))
    ur = (k * h1vp(MS[None, :], k * r[:, None], 1) * E) @ CM
    ut = (hankel1(MS[None, :], k * r[:, None]) * E) @ (1j * MS * CM) / r
    er = np.c_[np.cos(th), np.sin(th)]; et = np.c_[-np.sin(th), np.cos(th)]
    return ur[:, None] * er + ut[:, None] * et

# ------------------------------------------- per-element modal trace data
def local_modal_data(nt, nr):
    """Equilibrated Cauchy-trace modal coefficients b_m on each element's
    containing disk.  Setup cost; not part of the timed selection stage."""
    dr = (R - a) / nr
    nq = 256; t = 2 * np.pi * np.arange(nq) / nq; wq = 2 * np.pi / nq
    loc = []
    for i in range(nr):
        r0, r1 = a + i * dr, a + (i + 1) * dr
        for j in range(nt):
            tm = 2 * np.pi * j / nt + np.pi / nt
            cen = 0.5 * (r0 + r1) * np.array([np.cos(tm), np.sin(tm)])
            h = 0.55 * np.hypot(r1 - r0, r1 * 2 * np.pi / nt)
            M = min(45, int(np.ceil(k * h)) + 18)
            mm = np.arange(-M, M + 1)
            xq = cen + h * np.c_[np.cos(t), np.sin(t)]
            nrm = np.c_[np.cos(t), np.sin(t)]
            uv = u_exact(xq)
            un = np.einsum('qd,qd->q', grad_exact(xq), nrm)
            tau = np.sqrt(2 * np.pi * h * k
                          * (jv(mm, k * h) ** 2 + jvp(mm, k * h) ** 2))
            Ph = np.exp(1j * np.outer(t, mm))
            phi_v = jv(mm, k * h) * Ph
            phi_n = k * jvp(mm, k * h) * Ph
            ip = wq * h * (k * (phi_v.conj().T @ uv)
                           + (1 / k) * (phi_n.conj().T @ un))
            loc.append((mm, tau, ip / tau))
    return loc

# ------------------------------------------------------ ESPRIT selection
_LOC = None

def _init(loc):
    global _LOC
    _LOC = loc

def esprit_nodes(gamma, q):
    N = len(gamma)
    L = N // 2 + 1
    H = np.array([[gamma[r + s] for s in range(N - L + 1)] for r in range(L)])
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    Uq = U[:, :q]
    S, *_ = np.linalg.lstsq(Uq[:-1, :], Uq[1:, :], rcond=None)
    z = np.linalg.eigvals(S)
    return np.mod(-np.angle(z), 2 * np.pi)

def tail_residual(thetas, mm, tau, gamma, M):
    tail = np.abs(mm) > M
    if len(thetas) == 0:
        return float(np.sum(np.abs(tau[tail] * gamma[tail]) ** 2))
    A = tau[tail, None] * np.exp(-1j * np.outer(mm[tail], thetas))
    b = tau[tail] * gamma[tail]
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    return float(np.real(np.vdot(r, r)))

def select_element(K):
    mm, tau, b = _LOC[K]
    gamma = (1j ** (-mm)) * (b / tau)
    best = None
    for q in range(Q_MAX + 1):
        M = (P_BUDGET - q - 1) // 2
        if q == 0:
            cand = (tail_residual(np.array([]), mm, tau, gamma, M), q, M, ())
        else:
            i0 = np.searchsorted(mm, M + 1)
            g = gamma[i0:i0 + NS_ESPRIT]
            if len(g) < 2 * q + 1:
                continue
            th0 = esprit_nodes(g, q)
            out = minimize(tail_residual, th0,
                           args=(mm, tau, gamma, M),
                           method='Nelder-Mead',
                           options=dict(xatol=1e-6, fatol=1e-14,
                                        maxfev=200 * q))
            cand = (out.fun, q, M,
                    tuple(np.sort(np.mod(out.x, 2 * np.pi))))
        if best is None or cand[0] < 0.99 * best[0]:
            best = cand
    return (K,) + best

# ---------------------------------------------------------------- driver
def run_level(nt, nr, workers_list):
    loc = local_modal_data(nt, nr)
    ne = nt * nr
    _init(loc)
    t0 = time.perf_counter()
    serial = [select_element(K) for K in range(ne)]
    t_serial = time.perf_counter() - t0
    rows = [dict(nt=nt, nr=nr, ne=ne, workers=0, mode='serial',
                 t_wall=t_serial, per_elem_ms=1e3 * t_serial / ne,
                 speedup=1.0, efficiency=1.0)]
    for W in workers_list:
        with Pool(W, initializer=_init, initargs=(loc,)) as pool:
            t0 = time.perf_counter()
            par = pool.map(select_element, range(ne),
                           chunksize=max(1, ne // (4 * W)))
            t_par = time.perf_counter() - t0
        assert all(np.isclose(s_[1], p_[1]) and s_[2] == p_[2]
                   and np.allclose(s_[4], p_[4], atol=1e-12)
                   for s_, p_ in zip(serial, par)), "parallel != serial"
        rows.append(dict(nt=nt, nr=nr, ne=ne, workers=W, mode='pool',
                         t_wall=t_par, per_elem_ms=1e3 * t_par / ne,
                         speedup=t_serial / t_par,
                         efficiency=t_serial / t_par / W))
    return rows, serial

def print_latex(df):
    print('\n% ---- rows for tab:esprit-parallel ----')
    groups = list(df.groupby(['nt', 'nr'], sort=False))
    for gi, ((nt, nr), g) in enumerate(groups):
        first = True
        for _, r in g[g['mode'] == 'pool'].iterrows():
            lead = f"{int(r['ne'])}" if first else ''
            print(f"{lead} & {int(r['workers'])} & {r['t_wall']:.3f} & "
                  f"{r['per_elem_ms']:.1f} & {r['speedup']:.2f} & "
                  f"{r['efficiency']:.2f}\\\\")
            first = False
        if gi < len(groups) - 1:
            print('\\midrule')
    print('% ---- end rows ----')
    fin = df[df['nt'] == df['nt'].max()]
    ser = fin[fin['mode'] == 'serial'].iloc[0]
    best = fin[fin['mode'] == 'pool'].sort_values('speedup').iloc[-1]
    print('\n% numbers for the manuscript paragraph:')
    print(f"% finest level: {int(ser['ne'])} elements, serial "
          f"{ser['t_wall']:.2f} s ({ser['per_elem_ms']:.1f} ms per element); "
          f"W={int(best['workers'])}: {best['t_wall']:.2f} s, speedup "
          f"{best['speedup']:.1f}, efficiency {best['efficiency']:.2f}; "
          f"cpu_count={cpu_count()}.")

def main():
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass                                    # already set (notebook rerun)
    workers = WORKERS or sorted(
        {w for w in (1, 2, 4, 8, 16, 32, cpu_count()) if w <= cpu_count()})
    print(f'worker counts: {workers}  (cpu_count={cpu_count()})', flush=True)
    all_rows = []
    for nt, nr in LEVELS:
        rows, sel = run_level(nt, nr, workers)
        all_rows += rows
        qs = [s[2] for s in sel]
        print(f'level ({nt},{nr}): ne={nt*nr}, serial '
              f'{rows[0]["t_wall"]:.3f}s '
              f'({rows[0]["per_elem_ms"]:.2f} ms/elem), '
              f'q_PW histogram {np.bincount(qs, minlength=Q_MAX+1)}',
              flush=True)
        for r in rows[1:]:
            print(f'  W={r["workers"]}: {r["t_wall"]:.3f}s  '
                  f'speedup {r["speedup"]:.2f}  eff {r["efficiency"]:.2f}',
                  flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv('esprit_parallel.csv', index=False)
    print('\nwrote esprit_parallel.csv')
    print_latex(df)

if __name__ == '__main__':
    main()

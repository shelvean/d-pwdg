"""Parallel scaling of the elementwise ESPRIT selection stage.

Each element independently: ESPRIT-initializes candidate ray counts
q = 0,...,4 from its own equilibrated modal tail, polishes the angles by
Nelder-Mead in the weighted tail norm, and scores the (q_PW, q_FB) split
at fixed budget p = 21.  Elements are uncoupled, so the stage is
embarrassingly parallel; this script measures wall time against worker
count with multiprocessing over elements.

Run on the target machine:
    python3 run_esprit_parallel.py            # workers = 1,2,4,...,ncores
    python3 run_esprit_parallel.py 1 4 16     # explicit worker counts
Then regenerate the manuscript rows:
    python3 make_parallel_table.py

Output: data/esprit_parallel.csv.  Workers are pinned to one BLAS thread
each (OMP_NUM_THREADS=1) so the scaling measures process parallelism,
not thread oversubscription.  Correctness: the script asserts that every
parallel run reproduces the serial selections and residuals exactly.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_timing_levels import Mesh, LEVELS   # mesh + exact-field modal data

P_BUDGET = 21
Q_MAX = 4
NS_ESPRIT = 24        # consecutive positive-tail modal samples
ROOT = Path(__file__).resolve().parents[1]

_LOC = None           # per-process copy of the element modal data

def _init(loc):
    global _LOC
    _LOC = loc

def esprit_nodes(gamma, q):
    """Direction angles from the exponential sequence gamma_m ~ sum c_j e^{-im theta_j}."""
    N = len(gamma)
    L = N // 2 + 1
    H = np.array([[gamma[r + s] for s in range(N - L + 1)] for r in range(L)])
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    Uq = U[:, :q]
    S, *_ = np.linalg.lstsq(Uq[:-1, :], Uq[1:, :], rcond=None)
    z = np.linalg.eigvals(S)
    return np.mod(-np.angle(z), 2 * np.pi)

def tail_residual(thetas, mm, tau, gamma, M):
    """E(q, M)^2 of the hybrid identity: weighted LS fit of the modal tail."""
    tail = np.abs(mm) > M
    if len(thetas) == 0:
        return float(np.sum(np.abs(tau[tail] * gamma[tail]) ** 2))
    A = tau[tail, None] * np.exp(-1j * np.outer(mm[tail], thetas))
    b = tau[tail] * gamma[tail]
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ c
    return float(np.real(np.vdot(r, r)))

def select_element(K):
    """Full ESPRIT-initialized (q_PW, q_FB) selection for one element."""
    mm, tau, b = _LOC[K]
    gamma = (1j ** (-mm)) * (b / tau)          # de-phased modal coefficients
    best = None
    for q in range(Q_MAX + 1):
        M = (P_BUDGET - q - 1) // 2            # FB modes |m| <= M
        if q == 0:
            res = tail_residual(np.array([]), mm, tau, gamma, M)
            cand = (res, q, M, ())
        else:
            i0 = np.searchsorted(mm, M + 1)
            g = gamma[i0:i0 + NS_ESPRIT]
            if len(g) < 2 * q + 1:
                continue
            th0 = esprit_nodes(g, q)
            out = minimize(tail_residual, th0,
                           args=(mm, tau, gamma, M),
                           method='Nelder-Mead',
                           options=dict(xatol=1e-6, fatol=1e-14, maxfev=200 * q))
            cand = (out.fun, q, M, tuple(np.sort(np.mod(out.x, 2 * np.pi))))
        if best is None or cand[0] < 0.99 * best[0]:   # smallest active space wins ties
            best = cand
    return (K,) + best

def run_level(nt, nr, workers_list):
    mesh = Mesh(nt, nr)
    ne = mesh.ne
    _init(mesh.loc)
    t0 = time.perf_counter()
    serial = [select_element(K) for K in range(ne)]
    t_serial = time.perf_counter() - t0
    rows = [dict(nt=nt, nr=nr, ne=ne, workers=0, mode='serial',
                 t_wall=t_serial, per_elem_ms=1e3 * t_serial / ne,
                 speedup=1.0, efficiency=1.0)]
    for W in workers_list:
        with Pool(W, initializer=_init, initargs=(mesh.loc,)) as pool:
            t0 = time.perf_counter()
            par = pool.map(select_element, range(ne),
                           chunksize=max(1, ne // (4 * W)))
            t_par = time.perf_counter() - t0
        assert all(np.isclose(a[1], b_[1]) and a[2] == b_[2] and
                   np.allclose(a[4], b_[4], atol=1e-12)
                   for a, b_ in zip(serial, par)), "parallel != serial"
        rows.append(dict(nt=nt, nr=nr, ne=ne, workers=W, mode='pool',
                         t_wall=t_par, per_elem_ms=1e3 * t_par / ne,
                         speedup=t_serial / t_par,
                         efficiency=t_serial / t_par / W))
    return rows, serial

if __name__ == '__main__':
    if len(sys.argv) > 1:
        workers = [int(w) for w in sys.argv[1:]]
    else:
        nc = cpu_count()
        workers = sorted({w for w in (1, 2, 4, 8, 16, 32, nc) if w <= nc})
    print(f'worker counts: {workers}  (cpu_count={cpu_count()})', flush=True)
    all_rows = []
    for nt, nr in LEVELS:
        rows, sel = run_level(nt, nr, workers)
        all_rows += rows
        qs = [s[2] for s in sel]
        print(f'level ({nt},{nr}): ne={nt*nr}, serial '
              f'{rows[0]["t_wall"]:.3f}s ({rows[0]["per_elem_ms"]:.2f} ms/elem), '
              f'selected q_PW histogram {np.bincount(qs, minlength=5)}', flush=True)
        for r in rows[1:]:
            print(f'  W={r["workers"]}: {r["t_wall"]:.3f}s  '
                  f'speedup {r["speedup"]:.2f}  eff {r["efficiency"]:.2f}', flush=True)
        pd.DataFrame(all_rows).to_csv(ROOT / 'data' / 'esprit_parallel.csv',
                                      index=False)
    print('wrote data/esprit_parallel.csv')

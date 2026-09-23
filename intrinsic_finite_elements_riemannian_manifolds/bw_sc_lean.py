"""Seifert-Weber coexact spectrum on the Bernstein-Whitney complex, memory-lean.

No global 2-form mass and no global sparse triple product.  Each cell forms, in the
coordinates of the global functions it sees (local copies collapsed),

    M1_c = P1^T M1_loc P1,   M2_c = P2^T M2_loc P2,   A_c = D_c^T M2_c D_c,
    K_c  = A_c - sigma M1_c,

eliminates its interior functions (supported in the cell only), and adds its Schur
contribution to the boundary system.  Shift-invert then uses the factored boundary
Schur complement and the stored cell blocks; the mass products are cell-wise.

usage: python3 bw_sc_lean.py r [level] [nev] [sigma]
"""
import sys, time, gc, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import scipy.linalg as la
import hypforms
from sw_forms import seifert_weber
from bw_complex import BWComplex, coefficients, mass, cell_d

r = int(sys.argv[1]); level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
nev = int(sys.argv[3]) if len(sys.argv) > 3 else 6
sig = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
t0 = time.perf_counter()
cells, pairings, _ = seifert_weber(level)
X = BWComplex(cells, pairings, r, hypforms.Q, forms=(1, 2))
idx1, idx2 = X.index[1], X.index[2]
N = X.dims[1]
isB = np.ones(N, bool)
for key, g in idx1.items():
    if key[0] == 3:
        isB[g] = False
Bidx = np.where(isB)[0]
posB = -np.ones(N, np.int64); posB[Bidx] = np.arange(len(Bidx))

store = []                       # per cell: (global 1-form ids, M1_c, K_c, interior pos, boundary pos, lu)
srows, scols, svals = [], [], []
worst_copy = 0.0
for c in range(len(cells)):
    b1, b2 = X.basis[1][c], X.basis[2][c]
    k1 = list(dict.fromkeys(e[0] for e in b1)); p1 = {k: i for i, k in enumerate(k1)}
    k2 = list(dict.fromkeys(e[0] for e in b2)); p2 = {k: i for i, k in enumerate(k2)}
    P1 = np.zeros((len(b1), len(k1))); P1[np.arange(len(b1)), [p1[e[0]] for e in b1]] = 1
    P2 = np.zeros((len(b2), len(k2))); P2[np.arange(len(b2)), [p2[e[0]] for e in b2]] = 1
    M1c = P1.T @ mass(coefficients(b1, r, 1), cells[c], hypforms.Q, r, 1, r + 4) @ P1
    M2c = P2.T @ mass(coefficients(b2, r, 2), cells[c], hypforms.Q, r, 2, r + 4) @ P2
    cd, cw = cell_d(c, X.classes, b1, r, 1)
    worst_copy = max(worst_copy, cw)
    Dc = np.zeros((len(k2), len(k1)))
    for s, tg in cd.items():
        for t, v in tg.items():
            Dc[p2[t], p1[s]] = v
    Kc = Dc.T @ M2c @ Dc - sig * M1c
    g = np.array([idx1[k] for k in k1])
    Ip = np.where(~isB[g])[0]; Bp = np.where(isB[g])[0]
    lu = la.lu_factor(Kc[np.ix_(Ip, Ip)])
    KIB = Kc[np.ix_(Ip, Bp)]
    Sc = Kc[np.ix_(Bp, Bp)] - KIB.T @ la.lu_solve(lu, KIB)
    gb = posB[g[Bp]]
    srows.append(np.repeat(gb, len(gb)).astype(np.int32))
    scols.append(np.tile(gb, len(gb)).astype(np.int32))
    svals.append(Sc.ravel())
    store.append((g, M1c, Kc, Ip, Bp, lu))
    del M2c, Dc, P1, P2
t1 = time.perf_counter()
S = sp.csc_matrix((np.concatenate(svals), (np.concatenate(srows), np.concatenate(scols))),
                  shape=(len(Bidx), len(Bidx)))
del srows, scols, svals; gc.collect()
luS = sla.splu(S, permc_spec='MMD_AT_PLUS_A')
nnzLU = luS.L.nnz + luS.U.nnz
del S; gc.collect()
t2 = time.perf_counter()


def opinv(b):
    b = np.asarray(b).ravel()
    bB = b[Bidx].copy()
    for g, M1c, Kc, Ip, Bp, lu in store:
        yi = la.lu_solve(lu, b[g[Ip]])
        bB[posB[g[Bp]]] -= Kc[np.ix_(Bp, Ip)] @ yi
    xB = luS.solve(bB)
    x = np.zeros(N); x[Bidx] = xB
    for g, M1c, Kc, Ip, Bp, lu in store:
        x[g[Ip]] = la.lu_solve(lu, b[g[Ip]] - Kc[np.ix_(Ip, Bp)] @ xB[posB[g[Bp]]])
    return x


def mmul(x):
    x = np.asarray(x).ravel(); y = np.zeros(N)
    for g, M1c, Kc, Ip, Bp, lu in store:
        y[g] += M1c @ x[g]
    return y


def amul(x):
    x = np.asarray(x).ravel(); y = np.zeros(N)
    for g, M1c, Kc, Ip, Bp, lu in store:
        y[g] += (Kc + sig * M1c) @ x[g]
    return y


OP = sla.LinearOperator((N, N), matvec=opinv, dtype=float)
Mop = sla.LinearOperator((N, N), matvec=mmul, dtype=float)
Aop = sla.LinearOperator((N, N), matvec=amul, dtype=float)
lam = np.sort(sla.eigsh(Aop, M=Mop, k=nev, sigma=sig, OPinv=OP, which='LM', tol=1e-13,
                        return_eigenvectors=False))
t3 = time.perf_counter()
co = lam[lam > 1.0]
print(f'r={r} level={level} 1-forms {N} (boundary {len(Bidx)}) copies-consistency {worst_copy:.1e} '
      f'cells {t1-t0:.0f}s factor {t2-t1:.0f}s solve {t3-t2:.0f}s nnz(L+U) {nnzLU}')
print('coexact:', np.array2string(co, precision=12, max_line_width=170))
print(f'mean of lowest six {co[:6].mean():.12f}')
np.save(f'run/bwlean_r{r}_L{level}.npy', lam)

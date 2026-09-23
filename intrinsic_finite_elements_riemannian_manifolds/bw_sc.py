"""Seifert-Weber coexact spectrum on the Bernstein-Whitney complex, with the shift-invert
solve done by static condensation.

Interior functions (subsimplex = the cell itself) are supported in one cell, so in
K = A - sigma M1 the interior-interior block is block diagonal, one dense block per
cell.  Eliminating them leaves the Schur complement

    S = K_BB - sum_cells K_Bi K_ii^{-1} K_iB

on the edge and face functions only.  Each K_ii is nonsingular for sigma = 2: on an
interior block the eigenvalues of the pencil (A_ii, M_ii) are either 0 (gradients of
interior 0-form bubbles) or large, since the cells are small.

usage: python3 bw_sc.py r [level] [nev] [sigma]
"""
import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import scipy.linalg as la
import hypforms
from sw_forms import seifert_weber
from bw_complex import BWComplex

r = int(sys.argv[1]); level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
nev = int(sys.argv[3]) if len(sys.argv) > 3 else 6
sig = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
t0 = time.perf_counter()
cells, pairings, _ = seifert_weber(level)
X = BWComplex(cells, pairings, r, hypforms.Q, forms=(0, 1, 2))
D0, D1 = X.assemble_d(0), X.assemble_d(1)
t1 = time.perf_counter()
M1, M2 = X.assemble_mass(1), X.assemble_mass(2)
A = (D1.T @ M2 @ D1).tocsr()
del M2
t2 = time.perf_counter()
dd = abs(D1 @ D0).max()

# interior / boundary split, interior grouped by cell
keys = list(X.index[1].keys())
interior = {}
for key, g in X.index[1].items():
    if key[0] == 3:
        interior.setdefault(key[1], []).append(g)
blocks = [np.array(sorted(v)) for v in interior.values()]
Iset = np.concatenate(blocks)
N = X.dims[1]
isB = np.ones(N, bool); isB[Iset] = False
Bidx = np.where(isB)[0]
posB = -np.ones(N, int); posB[Bidx] = np.arange(len(Bidx))
K = (A - sig * M1).tocsr()

# factor interior blocks and form the Schur complement on the boundary functions
facts, Kib = [], []
rows, cols, vals = [], [], []
for Ib in blocks:
    Kii = K[Ib][:, Ib].toarray()
    KiB = K[Ib][:, Bidx]
    nzc = np.unique(KiB.nonzero()[1])               # boundary functions touching this cell
    KiBd = KiB[:, nzc].toarray()
    lu = la.lu_factor(Kii)
    W = la.lu_solve(lu, KiBd)                       # K_ii^{-1} K_iB
    corr = KiBd.T @ W                               # K_Bi K_ii^{-1} K_iB  (K symmetric)
    rows.append(np.repeat(nzc, len(nzc))); cols.append(np.tile(nzc, len(nzc)))
    vals.append(-corr.ravel())
    facts.append((Ib, lu, nzc))
KBB = K[Bidx][:, Bidx].tocoo()
S = sp.csr_matrix((np.concatenate([KBB.data] + vals),
                   (np.concatenate([KBB.row] + rows), np.concatenate([KBB.col] + cols))),
                  shape=(len(Bidx), len(Bidx))).tocsc()
luS = sla.splu(S, permc_spec='MMD_AT_PLUS_A')
t3 = time.perf_counter()


def opinv(b):
    b = np.asarray(b).ravel()
    bB = b[Bidx].copy()
    for Ib, lu, nzc in facts:
        yi = la.lu_solve(lu, b[Ib])
        bB[nzc] -= K[Ib][:, Bidx[nzc]].T @ yi if False else (K[Bidx[nzc]][:, Ib] @ yi)
    xB = luS.solve(bB)
    x = np.zeros(N)
    x[Bidx] = xB
    for Ib, lu, nzc in facts:
        x[Ib] = la.lu_solve(lu, b[Ib] - K[Ib][:, Bidx[nzc]] @ xB[nzc])
    return x


# precompute the cell coupling slices once (they are reused in every application)
coupl = []
for Ib, lu, nzc in facts:
    coupl.append((Ib, lu, nzc, K[Ib][:, Bidx[nzc]].toarray()))


def opinv_fast(b):
    b = np.asarray(b).ravel()
    bB = b[Bidx].copy()
    ys = []
    for Ib, lu, nzc, C in coupl:
        yi = la.lu_solve(lu, b[Ib])
        bB[nzc] -= C.T @ yi
    xB = luS.solve(bB)
    x = np.zeros(N)
    x[Bidx] = xB
    for Ib, lu, nzc, C in coupl:
        x[Ib] = la.lu_solve(lu, b[Ib] - C @ xB[nzc])
    return x


OP = sla.LinearOperator((N, N), matvec=opinv_fast, dtype=float)
lam = np.sort(sla.eigsh(A, M=M1, k=nev, sigma=sig, OPinv=OP, which='LM', tol=1e-13,
                        return_eigenvectors=False))
t4 = time.perf_counter()
co = lam[lam > 1.0]
print(f'r={r} level={level} 1-forms {N} (interior {len(Iset)}, boundary {len(Bidx)}) '
      f'|D1D0|={dd:.1e} complex {t1-t0:.0f}s masses {t2-t1:.0f}s condense {t3-t2:.0f}s '
      f'solve {t4-t3:.0f}s  nnz(L+U) {luS.L.nnz + luS.U.nnz}')
print('coexact:', np.array2string(co, precision=12, max_line_width=170))
print(f'mean of lowest six {co[:6].mean():.12f}')
np.save(f'run/bwsc_r{r}_L{level}.npy', lam)

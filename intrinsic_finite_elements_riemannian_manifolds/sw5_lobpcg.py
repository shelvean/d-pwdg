"""Degree-5 coexact spectrum on Seifert-Weber without factoring the 36k system.

Solve (A + alpha G) x = lambda M1 x with G = M1 D0 (L0 + eps I)^{-1} D0^T M1 and
L0 = D0^T M1 D0.  On coexact vectors (D0^T M1 x = 0) G vanishes, so the coexact
eigenpairs are unchanged; on exact vectors x = D0 phi, G x = M1 x up to eps, so the
exact part is lifted to about alpha.  Only the 0-form system (about 10k unknowns) is
factored; the 1-form operator is applied matrix-free inside LOBPCG.
"""
import sys, time, pickle, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from sw_forms import seifert_weber, SparseEntityComplex
r, level = 5, 1
alpha, nev, maxit = 40.0, int(sys.argv[1]) if len(sys.argv) > 1 else 16, 4000
t0 = time.perf_counter()
cells, pairings, metrics = seifert_weber(level)
C = SparseEntityComplex(len(cells), r, pairings)
A = sp.load_npz('run/sw5_A.npz').tocsr(); M1 = sp.load_npz('run/sw5_M1.npz').tocsr()
D0 = C.D[0].tocsr()
L0 = (D0.T @ M1 @ D0).tocsc()
eps = 1e-12 * L0.diagonal().mean()
lu0 = sla.splu((L0 + eps * sp.identity(L0.shape[0])).tocsc())
print('sizes', A.shape, 'L0', L0.shape, 'L0 nnz', L0.nnz, 'LU0 nnz', lu0.L.nnz + lu0.U.nnz,
      round(time.perf_counter() - t0), 's', flush=True)
def Bmul(X):
    X = np.asarray(X)
    Y = M1 @ X
    Z = lu0.solve(np.ascontiguousarray(D0.T @ Y))
    return A @ X + alpha * (M1 @ (D0 @ Z))
N = A.shape[0]
Bop = sla.LinearOperator((N, N), matvec=Bmul, matmat=Bmul, dtype=float)
dg = A.diagonal() + M1.diagonal()
P = sla.LinearOperator((N, N), matvec=lambda x: x / dg if x.ndim == 1 else x / dg[:, None],
                       matmat=lambda X: X / dg[:, None], dtype=float)
rng = np.random.default_rng(1)
X0 = rng.standard_normal((N, nev))
hist = []
w, V, resid = sla.lobpcg(Bop, X0, B=M1, M=P, tol=1e-9, maxiter=maxit, largest=False,
                         retResidualNormsHistory=True, verbosityLevel=0)
w = np.sort(w)
print('iterations', len(resid), 'final max residual', float(np.max(resid[-1])), flush=True)
print('eigenvalues', np.array2string(w, precision=10, max_line_width=160), flush=True)
np.save('run/swlam_r5_L1_lobpcg.npy', w)
print('seconds', round(time.perf_counter() - t0), flush=True)

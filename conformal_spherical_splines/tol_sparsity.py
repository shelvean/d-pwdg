"""Rank-tolerance sensitivity and sparsity growth for the null-space matrix
(C^1, d=6): rank J, dim, first eigenvalues of the round sphere for
tol = 1e-6..1e-12 at level 1; and N, n_h, nnz(Z), nnz(Z^T K Z) for levels 1-4."""
import json, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from sphsplines import icosphere, smoothness_sphere, assemble_sphere
from prolong import column_order, build_prolongation
d, r = 6, 1; m = 28
out = {}
v, t = icosphere(1); N = m * t.shape[0]
J = smoothness_sphere(v, t, d, r)
M, K = assemble_sphere(v, t, d, q=12, weight=None)
rows = []
for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
    Z, D, Fr = build_prolongation(J, N, column_order(v, t, d), tol=tol)
    Kz = (Z.T @ K @ Z).tocsc(); Mz = (Z.T @ M @ Z).tocsc()
    vals = np.sort(spla.eigsh(Kz, k=8, M=Mz, sigma=-1.0, which='LM', return_eigenvectors=False))
    rows.append(dict(tol=tol, rank=int(N - Z.shape[1]), dim=int(Z.shape[1]), eig=[float(x) for x in vals[:8]]))
    print(json.dumps(rows[-1]))
out['tol'] = rows
sp_rows = []
for lev in [1, 2, 3, 4]:
    v, t = icosphere(lev); N = m * t.shape[0]
    t0 = time.time()
    J = smoothness_sphere(v, t, d, r)
    Z, D, Fr = build_prolongation(J, N, column_order(v, t, d))
    tz = time.time() - t0
    M, K = assemble_sphere(v, t, d, q=12, weight=None)
    t0 = time.time(); Kz = (Z.T @ K @ Z).tocsr(); Kz.eliminate_zeros(); tk = time.time() - t0
    row = dict(lev=lev, N=int(N), nh=int(Z.shape[1]), nnzJ=int(J.nnz), nnzZ=int(Z.nnz), nnzK=int(K.nnz), nnzKz=int(Kz.nnz),
               nnzZ_per_col=float(Z.nnz / Z.shape[1]), nnzKz_per_row=float(Kz.nnz / Kz.shape[0]), t_Z=tz, t_Kz=tk)
    sp_rows.append(row); print(json.dumps(row))
out['sparsity'] = sp_rows
json.dump(out, open('tol_sparsity.json', 'w'), indent=1)

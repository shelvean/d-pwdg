"""Table 'tab:cond' (conditioning and sparsity on the torus) and the dimension
checks of Table 'tab:dim' for the flat quotients."""
import numpy as np, scipy.sparse.linalg as spla
from flatquot import Quotient
for kind in ['torus', 'klein', 'mobius']:
    for n in [4, 8]:
        for d, r in [(2, 0), (4, 0), (6, 1)]:
            Q = Quotient(kind, n, n, d, r); Z = Q.nullspace()
            print(f"dim {kind:7s} n={n} ({d},{r}) N={Q.N:6d} dim={Z.shape[1]:6d}")
for d, r in [(2, 0), (6, 1)]:
    for n in [4, 8, 16]:
        Q = Quotient('torus', n, n, d, r); Z = Q.nullspace()
        K, M, F_, c1 = Q.assemble(q=d+3)
        Kz = (Z.T@K@Z).tocsr(); Mz = (Z.T@M@Z).tocsr(); Kz.eliminate_zeros(); Mz.eliminate_zeros()
        A = (Kz+Mz).tocsc()
        lmax = spla.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]; lmin = spla.eigsh(A, k=1, sigma=0.0, which='LM', return_eigenvectors=False)[0]
        mmax = spla.eigsh(Mz.tocsc(), k=1, which='LM', return_eigenvectors=False)[0]; mmin = spla.eigsh(Mz.tocsc(), k=1, sigma=0.0, which='LM', return_eigenvectors=False)[0]
        print(f"cond ({d},{r}) n={n:2d} dim={Z.shape[1]:6d} nnzZ/col={Z.nnz/Z.shape[1]:.2f} nnzKz/row={Kz.nnz/Kz.shape[0]:.1f} cond(K+M)={lmax/lmin:.3e} cond(M)={mmax/mmin:.3e}", flush=True)

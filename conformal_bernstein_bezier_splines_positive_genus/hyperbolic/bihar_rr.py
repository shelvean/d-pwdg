"""Biharmonic on Bolza using the rank-revealing local prolongation, sparse throughout."""
import numpy as np, time
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from hypmesh import octagon_mesh, build_dofs
from hypbb import bern_indices, lap_basis, quad_points
from hypc1 import smoothness_rows
from prolong import rr_prolongation

def run(L, d, r=1, nquad=18, k=6):
    tris, V0, gens = octagon_mesh(L)
    dof, _ = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d)); ndof = int(dof.max()) + 1
    R, C, av, mv = [], [], [], []
    for t, T in enumerate(tris):
        X, w = quad_points(T, nquad)
        phi, lap = lap_basis(T, d, X)
        A = np.einsum("p,pm,pl->ml", w, lap, lap)
        M = np.einsum("p,pm,pl->ml", w, phi, phi)
        gl = dof[t*nloc:(t+1)*nloc]
        R.append(np.repeat(gl, nloc)); C.append(np.tile(gl, nloc))
        av.append(A.ravel()); mv.append(M.ravel())
    R, C = np.concatenate(R), np.concatenate(C)
    Ag = coo_matrix((np.concatenate(av), (R, C)), shape=(ndof, ndof)).tocsr()
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).tocsr()
    S = csr_matrix(smoothness_rows(tris, dof, d, r, V0, gens))
    t0 = time.time(); Z, rank, free, piv = rr_prolongation(S); tz = time.time() - t0
    Ar = (Z.T @ Ag @ Z).tocsc(); Mr = (Z.T @ Mg @ Z).tocsc()
    vals = np.sort(eigsh(Ar, k=k, M=Mr, sigma=-1.0, which="LM",
                         return_eigenvectors=False).real)
    return vals, ndof, Z.shape[1], len(tris), tz, Z.nnz

REF = 3.8388872588421995185866224504354645970819150157**2
print(f"reference lambda_1 = {REF:.14f}")
print(f"{'d':>2} {'L':>2} {'ntri':>5} {'ndof':>6} {'dim':>6} {'Z nnz':>7} "
      f"{'t_Z(s)':>7} {'lam_0':>10} {'lam_1':>16} {'rel err':>10}")
errs = {}
for d in (4, 6):
    for L in (1, 2, 3):
        vals, ndof, dim, nt, tz, znnz = run(L, d)
        rel = abs(vals[1]-REF)/REF
        errs.setdefault(d, []).append(rel)
        print(f"{d:>2} {L:>2} {nt:>5} {ndof:>6} {dim:>6} {znnz:>7} {tz:>7.2f} "
              f"{vals[0]:>10.2e} {vals[1]:>16.10f} {rel:>10.2e}")
    print()
for d, e in errs.items():
    print(f"    d={d} rates: " + ", ".join(f"{np.log2(a/b):.2f}" for a, b in zip(e, e[1:]))
          + f"    (predicted 2(d-1) = {2*(d-1)})")

"""
hypc1.py -- C^r smoothness constraints on the hyperbolic mesh, and a fourth-order test.

The C^r conditions have exactly the planar Bernstein form.  For triangles
T = (v0,v1,v2) and its neighbour with opposite vertex v3 across the edge (v1,v2),
write v3 = beta0 v0 + beta1 v1 + beta2 v2 as a LINEAR relation in R^{2,1} (no
normalisation: the polynomials are homogeneous, so no affine condition is needed).
Then for m = 0..r,
    ctilde_{(m,j,k)} = sum_{nu} c_{(nu0, j+nu1, k+nu2)} B^m_nu(beta).
Across a paired edge the neighbour is pulled back by the side pairing; barycentric
coordinates are invariant under SO^+(2,1), so its coefficients are unchanged and only
the vertex positions move.

Fourth-order test problem: Delta^2 u = lambda u on the Bolza surface.  On a closed
manifold Delta^2 and Delta share eigenfunctions, so the biharmonic eigenvalues are the
SQUARES of the Laplace eigenvalues.  That makes the Strohmaier-Uski value a published
reference for a fourth-order computation, and it is a conforming problem only in the
C^1 space.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from hypgeom import J, bolza_octagon, side_maps, inv_lorentz, dist
from hypbb import (bern_indices, multinomial, bary, bern_eval, lap_basis,
                   constant_coeffs, quad_points, vertex_matrix, gram)
from hypmesh import octagon_mesh, build_dofs, on_side


def _perm_to(target, Vm, tol=1e-8):
    """Permutation p with Vm[:, p[a]] == target[a]."""
    p = []
    for a in range(3):
        dd = [np.max(np.abs(Vm[:, b] - target[a])) for b in range(3)]
        b = int(np.argmin(dd))
        if dd[b] > tol:
            raise RuntimeError(f"vertex match failed, gap {dd[b]:.2e}")
        p.append(b)
    return p


def edge_partners(tris, V0, gens, tol=1e-8):
    """List of (t, a, tprime, gamma) : edge opposite vertex a of triangle t is glued to
    triangle tprime, whose vertices must be transformed by gamma before use."""
    mids, owner = [], []
    for t, T in enumerate(tris):
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            mids.append(0.5 * (T[:, b] + T[:, c]))
            owner.append((t, a))
    Mid = np.array(mids)
    tree = cKDTree(Mid)
    pairs = {}
    used = set()
    for i, jdx in tree.query_pairs(tol):
        pairs[i] = jdx
        pairs[jdx] = i
        used.add(i)
        used.add(jdx)

    g = side_maps(gens)
    free = [i for i in range(len(Mid)) if i not in used]
    for i in free:
        t, a = owner[i]
        T = tris[t]
        b, c = (a + 1) % 3, (a + 2) % 3
        for jsd in range(8):
            if on_side(np.array([T[:, b], T[:, c]]), V0, jsd).all():
                break
        else:
            raise RuntimeError("unmatched edge not on any side")
        img = g[jsd] @ Mid[i]
        dd, k = cKDTree(Mid[free]).query(img)
        if dd > 1e-7:
            raise RuntimeError(f"no partner for boundary edge, gap {dd:.2e}")
        pairs[i] = free[k]

    out = []
    for i, (t, a) in enumerate(owner):
        jdx = pairs[i]
        tp, ap = owner[jdx]
        if tp < t:
            continue
        T = tris[t]
        b, c = (a + 1) % 3, (a + 2) % 3
        gamma = np.eye(3)
        if np.max(np.abs(Mid[i] - Mid[jdx])) > tol:
            for jsd in range(8):
                if on_side(np.array([T[:, b], T[:, c]]), V0, jsd).all():
                    gamma = inv_lorentz(g[jsd])
                    break
        out.append((t, a, tp, gamma))
    return out


def smoothness_rows(tris, dof, d, r, V0, gens):
    """Rows of S imposing C^m, m = 1..r, on the merged (C^0) coefficient vector."""
    idx = bern_indices(d)
    pos = {e: m for m, e in enumerate(idx)}
    nloc = len(idx)
    rows, cols, vals = [], [], []
    nrow = 0
    for (t, a, tp, gamma) in edge_partners(tris, V0, gens):
        T = tris[t]
        b, c = (a + 1) % 3, (a + 2) % 3
        # canonical order for T: (opposite, shared1, shared2)
        pT = _perm_to([T[:, a], T[:, b], T[:, c]], T)
        Np = gamma @ tris[tp]
        # find the neighbour's opposite vertex
        shared = [T[:, b], T[:, c]]
        opp = None
        for q in range(3):
            if all(np.max(np.abs(Np[:, q] - s)) > 1e-8 for s in shared):
                opp = q
        pN = _perm_to([Np[:, opp], T[:, b], T[:, c]], Np)
        V = np.column_stack([T[:, a], T[:, b], T[:, c]])
        beta = np.linalg.solve(V, Np[:, opp])

        glT = dof[t * nloc:(t + 1) * nloc]
        glN = dof[tp * nloc:(tp + 1) * nloc]

        def locT(e):
            f = [0, 0, 0]
            for q in range(3):
                f[pT[q]] = e[q]
            return glT[pos[tuple(f)]]

        def locN(e):
            f = [0, 0, 0]
            for q in range(3):
                f[pN[q]] = e[q]
            return glN[pos[tuple(f)]]

        for m in range(1, r + 1):
            nus = [(i, j, m - i - j) for i in range(m, -1, -1) for j in range(m - i, -1, -1)]
            for jj in range(d - m + 1):
                kk = d - m - jj
                rows.append(nrow)
                cols.append(locN((m, jj, kk)))
                vals.append(1.0)
                for nu in nus:
                    rows.append(nrow)
                    cols.append(locT((nu[0], jj + nu[1], kk + nu[2])))
                    vals.append(-multinomial(m, *nu) * beta[0] ** nu[0]
                                * beta[1] ** nu[1] * beta[2] ** nu[2])
                nrow += 1
    ndof = int(dof.max()) + 1
    return coo_matrix((vals, (rows, cols)), shape=(nrow, ndof)).toarray()


def nullspace(S, tol=1e-10):
    U, s, Vt = np.linalg.svd(S, full_matrices=True)
    rank = int((s > tol * max(S.shape) * s[0]).sum()) if s.size else 0
    return Vt[rank:].T, rank


def biharmonic(level, d, r, nquad=20):
    """Assemble the C^r-constrained biharmonic and mass pencils."""
    tris, V0, gens = octagon_mesh(level)
    dof, P = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d))
    ndof = int(dof.max()) + 1
    rows, cols, av, mv = [], [], [], []
    for t, T in enumerate(tris):
        X, w = quad_points(T, nquad)
        phi, lap = lap_basis(T, d, X)
        A = np.einsum("p,pm,pl->ml", w, lap, lap)
        M = np.einsum("p,pm,pl->ml", w, phi, phi)
        gl = dof[t * nloc:(t + 1) * nloc]
        rows.append(np.repeat(gl, nloc))
        cols.append(np.tile(gl, nloc))
        av.append(A.ravel())
        mv.append(M.ravel())
    R, C = np.concatenate(rows), np.concatenate(cols)
    Ag = coo_matrix((np.concatenate(av), (R, C)), shape=(ndof, ndof)).toarray()
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).toarray()
    if r == 0:
        Z = np.eye(ndof)
        rank = 0
    else:
        S = smoothness_rows(tris, dof, d, r, V0, gens)
        Z, rank = nullspace(S)
    return Z.T @ Ag @ Z, Z.T @ Mg @ Z, Z.shape[1], ndof, len(tris)


if __name__ == "__main__":
    from scipy.linalg import eigh
    LAM1 = 3.8388872588421995185866224504354645970819150157
    REF = LAM1 ** 2
    print("=" * 78)
    print("Biharmonic  Delta^2 u = lambda u  on the Bolza surface")
    print(f"reference lambda_1 = lambda_1(Delta)^2 = {REF:.14f}")
    print("=" * 78)
    print(f"{'d':>2} {'r':>2} {'L':>2} {'ndof':>7} {'dim S^r_d':>10} "
          f"{'lam_0':>11} {'lam_1':>17} {'rel err':>10}")
    hist = {}
    for d, r in ((6, 1), (5, 1), (6, 0)):
        for L in (1, 2):
            A, M, nz, ndof, nt = biharmonic(L, d, r)
            vals = np.sort(eigh(A, M, eigvals_only=True).real)
            l0, l1 = vals[0], vals[1]
            rel = abs(l1 - REF) / REF
            hist.setdefault((d, r), []).append(rel)
            print(f"{d:>2} {r:>2} {L:>2} {ndof:>7} {nz:>10} "
                  f"{l0:>11.2e} {l1:>17.10f} {rel:>10.2e}")
        print()
    for k, v in hist.items():
        if len(v) > 1:
            print(f"    d={k[0]} r={k[1]}: rate {np.log2(v[0]/v[1]):.2f}")

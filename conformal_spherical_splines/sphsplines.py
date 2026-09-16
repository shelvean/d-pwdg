"""
sphsplines.py
=============

Spherical splines S^r_d on triangulations of the unit sphere, built on the
existing bivariate infrastructure (barynets, prolong). Follows Lai-Schumaker
Ch. 13-14: spherical Bernstein basis polynomials are restrictions to S^2 of
homogeneous trivariate polynomials H^d_ijk = (d!/i!j!k!) h1^i h2^j h3^k,
where h solves [v1 v2 v3] h = v.

What is reused verbatim from the planar code:
  - barynets.indices, barynets.locate  (B-net enumeration, degree shifts)
  - barynets.crcellarrays              (smoothness row combinatorics)
  - prolong.column_order, prolong.build_prolongation (null space of S)
The smoothness condition across an edge (Thm 13.30) has the SAME algebraic
form as the planar one; the only change is that the barycentric coordinates
of the fourth vertex are spherical (a 3x3 solve) instead of planar (2x2
affine). smoothness_sphere below is smoothfast.smoothness_fast with that one
geometric substitution.

Coefficient layout: discontinuous per-triangle blocks of size
m = (d+1)(d+2)/2 at offset kk*m, as everywhere else in the codebase.
Continuity (i = 0 rows) is part of the smoothness matrix S, so S^r_d is
ker S and Z from build_prolongation parametrizes it.

Quadrature on a spherical triangle T = <v1,v2,v3>: conical Gauss product on
the FLAT triangle in R^3 followed by radial projection u -> u/|u|. If u has
planar barycentrics lam in the flat triangle, then the spherical barycentrics
of v = u/|u| are b = lam/|u| (since u = V lam and V b = v), and the surface
measure is dA_S = (h_n/|u|^3) dA_flat with h_n the distance from the origin
to the plane of the triangle. Gradients of the homogeneous extension use
D_g H^d_ijk = d [ H^{d-1}_{i-1,j,k} h1(g) + H^{d-1}_{i,j-1,k} h2(g)
                 + H^{d-1}_{i,j,k-1} h3(g) ],  h(g) = V^{-1} g,
(Thm 13.28), projected tangentially: grad_S = (I - v v^T) grad.

Shelvean Kapita's spline-obstacle codebase + spherical extension, Aug 2026.
"""

import math
import numpy as np
import numpy.linalg as la
from scipy.special import gamma, comb, roots_jacobi
from scipy.sparse import csr_matrix, coo_matrix, block_diag
from scipy.sparse.linalg import eigsh

from barynets import indices, locate, crcellarrays
from prolong import column_order, build_prolongation


# ---------------------------------------------------------------------------
# spherical triangulations
# ---------------------------------------------------------------------------

def icosahedron():
    """Vertices (12x3, unit) and faces (20x3, det[v1,v2,v3] > 0)."""
    p = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
                  [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]],
                 dtype=float)
    v /= la.norm(v, axis=1)[:, None]
    from scipy.spatial import ConvexHull
    t = ConvexHull(v).simplices.astype(np.int64)
    return v, orient_out(v, t)


def orient_out(v, t):
    """Orient every triangle so that det[v1 v2 v3] > 0 (outward)."""
    t = t.copy()
    for k in range(t.shape[0]):
        if la.det(v[t[k, :], :].T) < 0:
            t[k, [1, 2]] = t[k, [2, 1]]
    return t


def refine_sphere(v, t):
    """One uniform 4-split with midpoints projected to the sphere."""
    nv = v.shape[0]
    mid = {}
    verts = [v]

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in mid:
            return mid[key]
        p = v[a, :] + v[b, :]
        p = p / la.norm(p)
        idx = nv + len(mid)
        mid[key] = idx
        verts.append(p[None, :])
        return idx

    newt = []
    for tri in t:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        newt += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    vnew = np.vstack(verts)
    tnew = np.array(newt, dtype=np.int64)
    return vnew, orient_out(vnew, tnew)


def icosphere(nref):
    v, t = icosahedron()
    for _ in range(nref):
        v, t = refine_sphere(v, t)
    return v, t


def meshsize_sphere(v, t):
    """Longest geodesic edge."""
    h = 0.0
    for tri in t:
        for a in range(3):
            i, j = int(tri[a]), int(tri[(a + 1) % 3])
            h = max(h, math.acos(np.clip(v[i] @ v[j], -1.0, 1.0)))
    return h


def fast_meshdata(t, nvert):
    """
    Combinatorial replacement for smesh.meshdata (same return signature),
    avoiding the O(N*E) edge scan. b is empty: closed surface, no boundary.
    """
    edge_id = {}
    e_list = []
    ti, tj = [], []
    for kk, tri in enumerate(t):
        for a in range(3):
            key = (min(int(tri[a]), int(tri[(a + 1) % 3])),
                   max(int(tri[a]), int(tri[(a + 1) % 3])))
            if key not in edge_id:
                edge_id[key] = len(e_list)
                e_list.append(key)
            ti.append(kk)
            tj.append(edge_id[key])
    e = np.array(e_list, dtype=np.int64)
    ne = e.shape[0]
    te = coo_matrix((np.ones(len(ti), dtype=np.int64), (ti, tj)),
                    shape=(t.shape[0], ne)).tocsr()
    tv = coo_matrix((np.ones(3 * t.shape[0], dtype=np.int64),
                     (np.repeat(np.arange(t.shape[0]), 3), t.ravel())),
                    shape=(t.shape[0], nvert)).tocsr()
    ev = coo_matrix((np.ones(2 * ne, dtype=np.int64),
                     (np.repeat(np.arange(ne), 2), e.ravel())),
                    shape=(ne, nvert)).tocsr()
    b = np.array([], dtype=np.int64)
    return e, b, te, tv, ev


# ---------------------------------------------------------------------------
# smoothness matrix (Thm 13.30): smoothfast with spherical barycentrics
# ---------------------------------------------------------------------------

_A = np.array([[0, 1], [1, 2], [2, 0], [1, 0], [2, 1], [0, 2]])
_AKEY = {(int(a), int(b)): i for i, (a, b) in enumerate(_A)}


def sphbary(V, x):
    """Spherical barycentrics of point(s) x wrt triangle columns V (3x3)."""
    return la.solve(V, np.asarray(x, dtype=float))


def smoothness_sphere(v, t, d, r, mesh=None):
    """
    C^0..C^r smoothness matrix S with S c = 0 on the discontinuous
    representation. Identical to smoothfast.smoothness_fast except that the
    barycentric coordinates of the fourth vertex are spherical.
    """
    e, b, te, tv, ev = mesh if mesh is not None else fast_meshdata(t, v.shape[0])
    deg = np.asarray(te.sum(axis=0)).ravel()
    inedges = np.nonzero(deg > 1)[0]
    n = inedges.size
    m = (d + 1) * (d + 2) // 2
    Neq = sum(d + 1 - j for j in range(r + 1))

    tec = te.tocsc()
    adj = [tec.indices[tec.indptr[k]:tec.indptr[k + 1]] for k in inedges]

    pre = []
    for i in range(r + 1):
        I, J, K = indices(i)
        coef = math.factorial(i) / (gamma(I + 1) * gamma(J + 1) * gamma(K + 1))
        I1, J1 = crcellarrays(d, i)
        pre.append((I, J, K, coef, I1, J1))

    ROW, COL, VAL = [], [], []

    for j in range(n):
        k = inedges[j]
        v1, v2 = e[k, 0], e[k, 1]
        t1, t2 = int(adj[j][0]), int(adj[j][1])
        T1, T2 = t[t1, :], t[t2, :]
        a = (int(np.argwhere(T1 == v1)[0][0]), int(np.argwhere(T1 == v2)[0][0]))
        bb = (int(np.argwhere(T2 == v1)[0][0]), int(np.argwhere(T2 == v2)[0][0]))
        e1, e2 = _AKEY[a], _AKEY[bb]
        if e1 > 2:
            e1 -= 3
            T1, T2 = T2, T1
            e1, e2 = e2, e1
            t1, t2 = t2, t1
        else:
            e2 -= 3

        v4 = np.setdiff1d(T2, np.array([v1, v2]))
        # --- the single geometric change: spherical barycentrics of v4 ---
        V1 = v[T1, :].T                      # 3x3, columns are vertices of T1
        lam = sphbary(V1, v[v4, :].ravel())
        if e1 == 1:
            lam = lam[[1, 2, 0]]
        elif e1 == 2:
            lam = np.array([lam[0], lam[2], lam[1]])

        EqCt = 0
        for i in range(r + 1):
            I, J, K, coef, I1, J1 = pre[i]
            Lambd = coef * (lam[0] ** I) * (lam[1] ** J) * (lam[2] ** K)
            T1mat = I1[:, :, e1]
            T2vec = J1[:, e2]
            numeq, ncol = T1mat.shape
            ve1 = (t1 * m + T1mat.T).ravel()
            ve2 = t2 * m + T2vec
            ve3 = np.tile(Lambd, (numeq, 1)).T.ravel()
            ve4 = -np.ones(T2vec.size)
            rows = (j * Neq + EqCt + np.arange(1, numeq + 1)[:, None]
                    * np.ones((numeq, ncol + 1))).astype(int)
            ROW.append((rows - 1).T.ravel())
            COL.append(np.hstack((ve1, ve2)))
            VAL.append(np.hstack((ve3, ve4)))
            EqCt += numeq

    if not ROW:
        return csr_matrix((0, m * t.shape[0]))
    return csr_matrix((np.concatenate(VAL),
                       (np.concatenate(ROW).astype(int),
                        np.concatenate(COL).astype(int))),
                      shape=(n * Neq, m * t.shape[0]))


# ---------------------------------------------------------------------------
# quadrature on spherical triangles
# ---------------------------------------------------------------------------

def refquad(q):
    """Conical Gauss product rule on the reference triangle in barycentrics.
    Returns lam (3 x q^2) and weights w (q^2,), sum(w) = 1/2."""
    x1, w1 = roots_jacobi(q, 1, 0)          # weight (1-x) absorbed
    x2, w2 = roots_jacobi(q, 0, 0)
    x1 = 0.5 * (x1 + 1.0); w1 = 0.25 * w1   # (1-x) on [0,1]: scale 1/4
    x2 = 0.5 * (x2 + 1.0); w2 = 0.5 * w2
    X1, X2 = np.meshgrid(x1, x2, indexing='ij')
    L1 = X1.ravel()
    L2 = (X2 * (1.0 - X1)).ravel()
    L3 = 1.0 - L1 - L2
    W = np.outer(w1, w2).ravel()
    return np.vstack([L1, L2, L3]), W


def triquad_sphere(v1, v2, v3, lam, w):
    """
    Map the reference rule to the spherical triangle <v1,v2,v3>.
    Returns: pts (3 x nq) on the sphere, sb (3 x nq) spherical barycentrics,
    W (nq,) surface-measure weights.
    """
    V = np.column_stack([v1, v2, v3])       # 3x3
    u = V @ lam                             # 3 x nq, points on flat triangle
    r = la.norm(u, axis=0)
    pts = u / r
    sb = lam / r                            # spherical barycentrics
    nrm = np.cross(v2 - v1, v3 - v1)
    A2 = la.norm(nrm)                        # 2 * flat area
    hn = abs(np.dot(v1, nrm)) / A2           # distance origin -> plane
    W = w * A2 * hn / r**3                   # (2A_flat) * h_n / |u|^3
    return pts, sb, W


# ---------------------------------------------------------------------------
# basis evaluation and gradients
# ---------------------------------------------------------------------------

def bern_eval(d, b):
    """All spherical Bernstein basis values: (m x nq) from barycentrics
    b (3 x nq). Valid for any b (not summing to 1)."""
    I, J, K = indices(d)
    mult = comb(d, I) * comb(d - I, J)       # d!/(i!j!k!)
    return mult[:, None] * (b[0][None, :] ** I[:, None]) \
        * (b[1][None, :] ** J[:, None]) * (b[2][None, :] ** K[:, None])


def grad_eval(d, b, Vinv):
    """
    Euclidean gradients of the homogeneous extensions of all basis functions
    at sphere points with spherical barycentrics b (3 x nq).
    Returns g (3, m, nq). Thm 13.28 with h(e_c) = Vinv[:, c].
    """
    I, J, K = indices(d)
    m, nq = I.size, b.shape[1]
    Bm1 = bern_eval(d - 1, b)                # (m1 x nq)
    i1, j1, k1 = indices(d - 1)

    def shifted(di, dj, dk):
        ii, jj, kk = I - di, J - dj, K - dk
        ok = (ii >= 0) & (jj >= 0) & (kk >= 0)
        out = np.zeros((m, nq))
        pos = locate(ii[ok], jj[ok], kk[ok], i1, j1, k1)
        out[ok, :] = Bm1[pos, :]
        return out

    T1, T2, T3 = shifted(1, 0, 0), shifted(0, 1, 0), shifted(0, 0, 1)
    g = np.empty((3, m, nq))
    for c in range(3):                       # e_x, e_y, e_z
        g[c] = d * (Vinv[0, c] * T1 + Vinv[1, c] * T2 + Vinv[2, c] * T3)
    return g


# ---------------------------------------------------------------------------
# assembly: block-diagonal mass and stiffness on the discontinuous space
# ---------------------------------------------------------------------------

def assemble_sphere(v, t, d, q=12, weight=None):
    """
    Global (block-diagonal) mass and stiffness matrices on S^0-discontinuous
    coefficients (size m*ntri). weight: optional callable w(pts) with pts
    (3 x nq) -> (nq,), the conformal factor e^{2 lambda}; default 1.
    K uses tangential gradients: grad_S = (I - x x^T) grad.
    """
    lam, wref = refquad(q)
    Mblocks, Kblocks = [], []
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, W = triquad_sphere(v1, v2, v3, lam, wref)
        B = bern_eval(d, sb)                            # m x nq
        Ww = W * weight(pts) if weight is not None else W
        Mblocks.append((B * Ww[None, :]) @ B.T)
        Vinv = la.inv(np.column_stack([v1, v2, v3]))
        g = grad_eval(d, sb, Vinv)                      # 3 x m x nq
        dot = np.einsum('cq,cmq->mq', pts, g)           # x . grad
        gt = g - pts[:, None, :] * dot[None, :, :]      # tangential
        Kloc = np.einsum('cmq,q,cnq->mn', gt, W, gt)
        Kblocks.append(Kloc)
    M = block_diag(Mblocks, format='csr')
    K = block_diag(Kblocks, format='csr')
    return M, K


# ---------------------------------------------------------------------------
# interpolation into S^0_d (per-triangle Vandermonde at domain points)
# ---------------------------------------------------------------------------

def domain_points_sphere(v1, v2, v3, d):
    """Spherical domain points and their spherical barycentrics."""
    I, J, K = indices(d)
    lam = np.vstack([I, J, K]).astype(float) / d
    V = np.column_stack([v1, v2, v3])
    u = V @ lam
    r = la.norm(u, axis=0)
    return u / r, lam / r


def interp_sphere(v, t, d, f):
    """Per-triangle interpolation of f(pts) at the spherical domain points.
    The result is C^infty-consistent for smooth f: shared edge/vertex domain
    points get identical coefficients from both sides only in the limit, but
    an f that IS a global spherical polynomial of degree d is reproduced
    exactly, coefficients agreeing across triangles (Thm 14.29)."""
    m = (d + 1) * (d + 2) // 2
    c = np.zeros(m * t.shape[0])
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb = domain_points_sphere(v1, v2, v3, d)
        A = bern_eval(d, sb).T                          # points x basis
        c[kk * m:(kk + 1) * m] = la.solve(A, f(pts))
    return c


def eval_on_triangles(v, t, d, c, q=8):
    """Evaluate the spline at quadrature points of every triangle; returns
    (values, exact-evaluation points) for error checks."""
    lam, wref = refquad(q)
    m = (d + 1) * (d + 2) // 2
    vals, ptss = [], []
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, _ = triquad_sphere(v1, v2, v3, lam, wref)
        B = bern_eval(d, sb)
        vals.append(c[kk * m:(kk + 1) * m] @ B)
        ptss.append(pts)
    return np.concatenate(vals), np.concatenate(ptss, axis=1)


# ---------------------------------------------------------------------------
# the eigenvalue pencil on S^r_d
# ---------------------------------------------------------------------------

def sphere_pencil(v, t, d, r, q=12, weight=None):
    """Constrained pencil: returns Z, Kz, Mz with columns of Z spanning
    ker S (= S^r_d(triangulation)), Kz = Z^T K Z, Mz = Z^T M Z."""
    m = (d + 1) * (d + 2) // 2
    S = smoothness_sphere(v, t, d, r)
    pos = column_order(v, t, d)
    Z, D, F = build_prolongation(S, m * t.shape[0], pos)
    M, K = assemble_sphere(v, t, d, q=q, weight=weight)
    Kz = (Z.T @ K @ Z).tocsr()
    Mz = (Z.T @ M @ Z).tocsr()
    return Z, Kz, Mz, S, M, K


def laplace_eigs(Kz, Mz, k=16):
    """Smallest k eigenvalues of the symmetric pencil, via shift-invert at
    sigma = -1 (K + M is SPD)."""
    Ks = ((Kz + Kz.T) * 0.5).tocsc()
    Ms = ((Mz + Mz.T) * 0.5).tocsc()
    vals = eigsh(Ks, k=k, M=Ms, sigma=-1.0, which='LM',
                 return_eigenvectors=False)
    return np.sort(vals)


def exact_sphere_spectrum(k):
    """First k Laplace-Beltrami eigenvalues of the round sphere with
    multiplicity: l(l+1), each 2l+1 times."""
    out = []
    l = 0
    while len(out) < k:
        out += [l * (l + 1)] * (2 * l + 1)
        l += 1
    return np.array(out[:k], dtype=float)

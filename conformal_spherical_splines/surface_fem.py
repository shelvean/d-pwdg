"""
surface_fem.py
==============

Parametric surface finite elements (Dziuk 1988; higher order as in Demlow
2009) for the Poisson equation -Delta_M u = f and the Laplace-Beltrami
eigenproblem on a closed genus-zero surface M given by a map
Phi : S^2 -> M.  Used only for comparison with the conformal spherical
spline method of poisson_surface.py.

Discrete surface Gamma_h: the icosahedral triangulation of S^2 at level
`lev` is pushed to M by Phi.  For polynomial degree k the geometry is
isoparametric: each triangle carries the (k+1)(k+2)/2 Lagrange nodes, the
spherical node positions are the normalized barycentric lattice points of
the spherical triangle, and the physical nodes are their images under Phi,
so every node lies on the exact surface M and Gamma_h is the degree-k
Lagrange interpolant of M (k = 1 is Dziuk's polyhedral surface).

Element computation on the reference triangle: X(xi) = sum_i N_i(xi) X_i,
J = dX/dxi (3 x 2), G = J^T J, dGamma = sqrt(det G) dxi, and the
tangential gradient of a basis function is grad_Gamma N_i = J G^{-1}
grad_xi N_i.

Data on Gamma_h: a quadrature point with reference coordinates xi is
associated with the spherical point s(xi) = normalize(sum_i N_i(xi) s_i);
f and u are evaluated there (i.e. at Phi(s(xi)) in M), which is a lift of
Gamma_h to M with O(h^{k+1}) displacement.  Errors are measured on
Gamma_h: L^2 with the mean removed, and the H^1 seminorm against the exact
surface gradient grad_M u = w^{-1} dPhi(grad_S u~), with dPhi by central
differences.
"""
import math, time
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sphsplines import icosphere, meshsize_sphere, refquad


# ---------------------------------------------------------------------------
# Lagrange basis on the reference triangle (barycentric lattice nodes)
# ---------------------------------------------------------------------------
def lattice(k):
    """Lattice nodes (i,j,l) with i+j+l=k, ordered: vertices, edges, interior."""
    nodes = []
    # vertices in the order of local vertex 0,1,2 : (k,0,0),(0,k,0),(0,0,k)
    nodes += [(k, 0, 0), (0, k, 0), (0, 0, k)]
    # edges: e0 = (v0,v1), e1 = (v1,v2), e2 = (v2,v0), interior points in
    # order from the first vertex to the second
    for a in range(1, k):
        nodes.append((k - a, a, 0))
    for a in range(1, k):
        nodes.append((0, k - a, a))
    for a in range(1, k):
        nodes.append((a, 0, k - a))
    for i in range(1, k):
        for j in range(1, k - i):
            l = k - i - j
            if l >= 1:
                nodes.append((i, j, l))
    return np.array(nodes, dtype=int)


def monomials(k, lam):
    """All bivariate monomials xi^a eta^b with a+b<=k at points with
    barycentrics lam (3 x n): xi = lam[1], eta = lam[2].  Returns values
    (n x m) and gradients (2 x n x m)."""
    xi, eta = lam[1], lam[2]
    cols, dxi, deta = [], [], []
    for a in range(k + 1):
        for b in range(k + 1 - a):
            cols.append(xi ** a * eta ** b)
            dxi.append(a * xi ** (a - 1) * eta ** b if a > 0 else np.zeros_like(xi))
            deta.append(b * xi ** a * eta ** (b - 1) if b > 0 else np.zeros_like(xi))
    return np.column_stack(cols), np.stack([np.column_stack(dxi), np.column_stack(deta)])


class RefElement:
    def __init__(self, k, q):
        self.k = k
        self.nodes = lattice(k)
        self.nn = len(self.nodes)
        lamn = self.nodes.T / k                              # 3 x nn barycentrics
        V, _ = monomials(k, lamn)                            # nn x nn Vandermonde
        self.Vinv = la.inv(V)
        self.lamq, self.wq = refquad(q)                      # 3 x nq, (nq,)
        P, dP = monomials(k, self.lamq)
        self.N = P @ self.Vinv                               # nq x nn basis values
        self.dN = np.stack([dP[0] @ self.Vinv, dP[1] @ self.Vinv])   # 2 x nq x nn


# ---------------------------------------------------------------------------
# mesh: global node numbering for degree k on the icosahedral triangulation
# ---------------------------------------------------------------------------
def build_mesh(lev, k, embed):
    v, t = icosphere(lev)                                    # v: NV x 3 on S^2
    NV, NT = v.shape[0], t.shape[0]
    ref = lattice(k)
    # edges
    edges = {}
    def edge_id(a, b):
        key = (min(a, b), max(a, b))
        if key not in edges:
            edges[key] = len(edges)
        return edges[key], (a < b)
    conn = np.zeros((NT, len(ref)), dtype=int)
    snodes = [v.copy()]                                     # spherical node coords
    edge_nodes = {}
    ne = k - 1
    nint = (k - 1) * (k - 2) // 2
    n_edge_total = 0
    # first pass: number edges
    for kk in range(NT):
        for (a, b) in [(t[kk, 0], t[kk, 1]), (t[kk, 1], t[kk, 2]), (t[kk, 2], t[kk, 0])]:
            edge_id(a, b)
    NE = len(edges)
    offset_edge = NV
    offset_int = NV + NE * ne
    coords = np.zeros((NV + NE * ne + NT * nint, 3))
    coords[:NV] = v
    for kk in range(NT):
        vi = t[kk]
        P = v[vi]                                            # 3 x 3 local vertices
        # vertices
        conn[kk, :3] = vi
        col = 3
        for ei, (a, b) in enumerate([(vi[0], vi[1]), (vi[1], vi[2]), (vi[2], vi[0])]):
            eid, fwd = edge_id(a, b)
            ids = offset_edge + eid * ne + np.arange(ne)
            if not fwd:
                ids = ids[::-1]
            conn[kk, col:col + ne] = ids
            # node coordinates from the lattice (only written once; same either way)
            for a_ in range(1, k):
                lamx = ref[col + a_ - 1] / k
                s = lamx @ P
                s /= la.norm(s)
                coords[conn[kk, col + a_ - 1]] = s
            col += ne
        for m in range(nint):
            gid = offset_int + kk * nint + m
            conn[kk, col + m] = gid
            lamx = ref[col + m] / k
            s = lamx @ P; s /= la.norm(s)
            coords[gid] = s
    X = embed(coords.T).T if embed is not None else coords
    return dict(v=v, t=t, conn=conn, S=coords, X=X, NV=NV, NE=NE, NT=NT, h=meshsize_sphere(v, t))


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def assemble(mesh, ref, f=None):
    """Stiffness K, mass M (sparse, global), load vector F (if f given, f is
    a callable of spherical points 3 x n), plus per-element quadrature
    data for error evaluation."""
    conn, X, S = mesh['conn'], mesh['X'], mesh['S']
    NT, nn = conn.shape
    nq = ref.wq.size
    rows = np.repeat(conn, nn, axis=1).ravel()
    cols = np.tile(conn, (1, nn)).ravel()
    Kv = np.zeros((NT, nn, nn)); Mv = np.zeros((NT, nn, nn))
    Fv = np.zeros(conn.shape)
    qdata = []
    for kk in range(NT):
        Xe = X[conn[kk]]                                     # nn x 3
        Se = S[conn[kk]]
        J = np.einsum('dqi,ic->qdc', ref.dN, Xe)             # nq x 2 x 3
        G = np.einsum('qdc,qec->qde', J, J)                  # nq x 2 x 2
        detG = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] ** 2
        dS = np.sqrt(detG) * ref.wq                          # surface weights
        Ginv = np.linalg.inv(G)
        # grad_Gamma N_i = J^T Ginv grad_xi N_i  -> nq x nn x 3
        gxi = np.transpose(ref.dN, (1, 2, 0))                # nq x nn x 2
        gG = np.einsum('qie,qed,qdc->qic', gxi, Ginv, J)
        Kv[kk] = np.einsum('q,qic,qjc->ij', dS, gG, gG)
        Mv[kk] = np.einsum('q,qi,qj->ij', dS, ref.N, ref.N)
        sq = ref.N @ Se                                      # nq x 3 spherical pts
        sq /= la.norm(sq, axis=1)[:, None]
        if f is not None:
            fq = f(sq.T)
            Fv[kk] = ref.N.T @ (dS * fq)
        qdata.append((dS, sq, gG))
    n = X.shape[0]
    K = sp.csr_matrix((Kv.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((Mv.ravel(), (rows, cols)), shape=(n, n))
    F = np.zeros(n); np.add.at(F, conn.ravel(), Fv.ravel())
    return K, M, F, qdata


def solve_poisson(mesh, ref, f):
    t0 = time.time()
    K, M, F, qdata = assemble(mesh, ref, f)
    n = K.shape[0]
    g = M @ np.ones(n)
    Aug = sp.bmat([[K, sp.csr_matrix(g[:, None])], [sp.csr_matrix(g[None, :]), None]], format='csc')
    sol = spla.spsolve(Aug, np.concatenate([F, [0.0]]))
    return sol[:n], dict(dim=n, time=time.time() - t0), (K, M, qdata)


def surface_gradient_exact(embed, grad_ut, weight, sq, eps=1e-5):
    """grad_M u at Phi(s) = w^{-1} dPhi_s(grad_S u~(s)), dPhi by central
    differences along the tangent vector grad_S u~ (sq: n x 3)."""
    s = sq.T                                                 # 3 x n
    g = grad_ut(s)                                           # 3 x n tangent
    gn = la.norm(g, axis=0)
    tvec = np.where(gn > 1e-300, g / np.maximum(gn, 1e-300), 0.0)
    sp_ = s + eps * tvec; sp_ /= la.norm(sp_, axis=0)
    sm_ = s - eps * tvec; sm_ /= la.norm(sm_, axis=0)
    dPhi = (embed(sp_) - embed(sm_)) / (2 * eps)             # 3 x n
    w = weight(s) if weight is not None else np.ones(s.shape[1])
    return dPhi * (gn / w)                                    # 3 x n


def errors(mesh, ref, uh, qdata, ut, grad_ut, weight, embed):
    conn = mesh['conn']
    num_e = num_h = den = 0.0
    for kk, (dS, sq, gG) in enumerate(qdata):
        uq = ref.N @ uh[conn[kk]]
        num_e += np.sum(dS * ut(sq.T)); num_h += np.sum(dS * uq); den += np.sum(dS)
    me, mh = num_e / den, num_h / den
    e0 = e1 = 0.0
    for kk, (dS, sq, gG) in enumerate(qdata):
        uq = ref.N @ uh[conn[kk]]
        e0 += np.sum(dS * ((ut(sq.T) - me) - (uq - mh)) ** 2)
        gh = np.einsum('i,qic->qc', uh[conn[kk]], gG)        # nq x 3
        if embed is not None:
            ge = surface_gradient_exact(embed, grad_ut, weight, sq).T
        else:
            ge = grad_ut(sq.T).T
        e1 += np.sum(dS * np.sum((ge - gh) ** 2, axis=1))
    return math.sqrt(e0), math.sqrt(e1)


def eigenvalues(K, M, k=16, sigma=-1.0):
    t0 = time.time()
    vals = spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=sigma, which='LM', return_eigenvectors=False)
    return np.sort(vals), time.time() - t0

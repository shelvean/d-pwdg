"""flatquot.py: broken Bernstein-Bezier splines on flat quotients of the
rectangle [0,a] x [0,b]: torus, Klein bottle, Mobius strip, cylinder, disk.

Topology enters through the identification of sides (the smoothness matrix J
carries it); geometry enters through a conformal factor w = exp(2 lambda) in the
mass matrix, load vector and mean constraint. The stiffness matrix is the flat
Dirichlet matrix of the rectangle whatever the metric.

Mesh: criss-cross (each cell of an n x m grid split by both diagonals), which is
symmetric under the reflections used by the Klein bottle and Mobius pairings, so
paired domain points match exactly.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from itertools import combinations_with_replacement
from math import comb, factorial
from prolong import rr_prolongation

# ----------------------------------------------------------------------------
# multi-indices and planar Bernstein basis
# ----------------------------------------------------------------------------
def multi_indices(d):
    return [(i, j, d - i - j) for i in range(d, -1, -1) for j in range(d - i, -1, -1)]

def bernstein(d, b):
    """b: (3, n) barycentric coordinates -> (N_d, n) Bernstein values."""
    idx = multi_indices(d)
    out = np.empty((len(idx), b.shape[1]))
    for k, (i, j, l) in enumerate(idx):
        out[k] = factorial(d) / (factorial(i) * factorial(j) * factorial(l)) * b[0]**i * b[1]**j * b[2]**l
    return out

def bernstein_grad(d, b, grad_b):
    """gradient of each Bernstein polynomial; grad_b: (3,2) gradients of b_i.
    Returns (N_d, 2, n)."""
    idx = multi_indices(d)
    pos = {a: k for k, a in enumerate(idx)}
    Bm = bernstein(d - 1, b) if d > 0 else None
    idxm = multi_indices(d - 1) if d > 0 else []
    posm = {a: k for k, a in enumerate(idxm)}
    out = np.zeros((len(idx), 2, b.shape[1]))
    for a, k in pos.items():
        for i in range(3):
            am = list(a); am[i] -= 1
            if am[i] < 0: continue
            out[k] += d * np.outer(grad_b[i], Bm[posm[tuple(am)]])
    return out

# ----------------------------------------------------------------------------
# quadrature on a triangle: conical Gauss product rule (exact to degree 2q-1)
# ----------------------------------------------------------------------------
def tri_quadrature(q):
    x, w = np.polynomial.legendre.leggauss(q)
    x = 0.5 * (x + 1); w = 0.5 * w
    pts, wts = [], []
    for i in range(q):
        for j in range(q):
            u, v = x[i], x[j]
            # Duffy: (u, v(1-u))
            pts.append((u, v * (1 - u))); wts.append(w[i] * w[j] * (1 - u))
    return np.array(pts), np.array(wts)

# ----------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------
def crisscross_mesh(n, m, a=1.0, b=1.0):
    """n x m cells, each split into 4 triangles by its two diagonals."""
    xs = np.linspace(0, a, n + 1); ys = np.linspace(0, b, m + 1)
    V = [(x, y) for y in ys for x in xs]
    idx = lambda i, j: j * (n + 1) + i
    C = []
    tris = []
    for j in range(m):
        for i in range(n):
            c = len(V); V.append((0.5 * (xs[i] + xs[i + 1]), 0.5 * (ys[j] + ys[j + 1])))
            v00, v10, v01, v11 = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            tris += [(v00, v10, c), (v10, v11, c), (v11, v01, c), (v01, v00, c)]
    return np.array(V), np.array(tris)

# identification maps on the rectangle: each returns the image of a point on a
# side under the pairing, or None. Sides: L (x=0), R (x=a), B (y=0), T (y=b).
def pairings(kind, a, b):
    P = {}
    if kind in ('torus', 'klein', 'cylinder', 'mobius'):
        if kind == 'mobius':
            P['LR'] = lambda p: (p[0] + a, b - p[1])       # left -> right with flip
        else:
            P['LR'] = lambda p: (p[0] + a, p[1])
    if kind == 'torus':
        P['BT'] = lambda p: (p[0], p[1] + b)
    if kind == 'klein':
        P['BT'] = lambda p: (a - p[0], p[1] + b)           # bottom -> top with flip
    return P

class Quotient:
    def __init__(self, kind, n, m, d, r=0, a=1.0, b=1.0, tol=1e-9):
        self.kind, self.d, self.r, self.a, self.b = kind, d, r, a, b
        self.V, self.tris = crisscross_mesh(n, m, a, b)
        self.pair = pairings(kind, a, b)
        self.idx = multi_indices(d)
        self.Nloc = len(self.idx)
        self.N = self.Nloc * len(self.tris)
        self._domain_points()
        self._merge_dofs(tol)
        self._smoothness(tol)

    # domain points of every triangle (broken numbering)
    def _domain_points(self):
        d = self.d
        pts = np.zeros((self.N, 2))
        for t, T in enumerate(self.tris):
            P = self.V[list(T)]
            for k, (i, j, l) in enumerate(self.idx):
                pts[t * self.Nloc + k] = (i * P[0] + j * P[1] + l * P[2]) / d
        self.pts = pts

    def _canonical(self, p):
        """map a point to a canonical representative under the pairings"""
        x, y = p; a, b = self.a, self.b; tol = 1e-9
        if 'LR' in self.pair and abs(x - a) < tol:
            x, y = 0.0, (b - y if self.kind == 'mobius' else y)
        if 'BT' in self.pair and abs(y - b) < tol:
            x, y = (a - x if self.kind == 'klein' else x), 0.0
            if 'LR' in self.pair and abs(x - a) < tol:   # corner
                x = 0.0
        return (round(x, 9), round(y, 9))

    def _merge_dofs(self, tol):
        """C^0 merging of domain points modulo the identifications: gives the
        global C^0 numbering (used for the C^0 space and for nodal handling)."""
        keys = {}
        g = np.empty(self.N, dtype=int)
        for k in range(self.N):
            c = self._canonical(self.pts[k])
            if c not in keys: keys[c] = len(keys)
            g[k] = keys[c]
        self.c0map = g; self.n0 = len(keys)

    # --- smoothness rows -----------------------------------------------------
    def _edge_table(self, tol):
        """edges as (triangle, local edge) with their partner across the edge,
        including glued edges. Local edge e of triangle T is opposite vertex e."""
        d = self.d
        # key an edge by the canonical images of its two endpoints
        table = {}
        for t, T in enumerate(self.tris):
            P = self.V[list(T)]
            for e in range(3):
                q0, q1 = P[(e + 1) % 3], P[(e + 2) % 3]
                key = frozenset([self._canonical(q0), self._canonical(q1)])
                table.setdefault(key, []).append((t, e))
        return table

    def _smoothness(self, tol):
        d, r = self.d, self.r
        rows, cols, vals = [], [], []
        nrow = 0
        table = self._edge_table(tol)
        for key, lst in table.items():
            if len(lst) != 2: continue      # boundary edge
            (t1, e1), (t2, e2) = lst
            P1 = self.V[list(self.tris[t1])]; P2 = self.V[list(self.tris[t2])]
            # bring triangle 2 into the frame of triangle 1: its shared vertices
            # must coincide with triangle 1's shared vertices; find the rigid
            # motion (pairing map) doing so, then transform its third vertex.
            s1 = [P1[(e1 + 1) % 3], P1[(e1 + 2) % 3]]
            s2 = [P2[(e2 + 1) % 3], P2[(e2 + 2) % 3]]
            v4 = P2[e2]
            g = self._motion(s2, s1)
            if g is None:  # identical positions (interior edge): identity
                Q = v4
            else:
                Q = g(v4)
            # match orientation: which of s1 corresponds to which of s2
            # barycentric coordinates of Q w.r.t. triangle 1
            beta = np.linalg.solve(np.vstack([P1.T, np.ones(3)]), np.array([Q[0], Q[1], 1.0]))
            # local edge coefficient correspondence: domain points on the edge
            # of t2 mapped by g must equal those of t1
            for rr in range(r + 1):
                for row in self._cr_rows(t1, e1, t2, e2, g, beta, rr):
                    for (c, v) in row:
                        rows.append(nrow); cols.append(c); vals.append(v)
                    nrow += 1
        self.J = sp.csr_matrix((vals, (rows, cols)), shape=(nrow, self.N))

    def _motion(self, s2, s1):
        """rigid motion of the plane sending segment s2 onto s1 (as sets), or
        None if they already coincide."""
        A = np.array(s2); B = np.array(s1)
        if np.allclose(A, B, atol=1e-9) or np.allclose(A, B[::-1], atol=1e-9):
            return None
        # try the pairings (and their inverses / compositions)
        cands = []
        for k, f in self.pair.items():
            cands.append(f)
            if k == 'LR':
                cands.append(lambda p, f=f: (p[0] - self.a, (self.b - p[1]) if self.kind == 'mobius' else p[1]))
            if k == 'BT':
                cands.append(lambda p, f=f: ((self.a - p[0]) if self.kind == 'klein' else p[0], p[1] - self.b))
        for g in cands:
            img = np.array([g(p) for p in A])
            if np.allclose(img, B, atol=1e-8) or np.allclose(img, B[::-1], atol=1e-8):
                return g
        raise RuntimeError('no pairing maps edge onto its partner')

    def _cr_rows(self, t1, e1, t2, e2, g, beta, rr):
        """C^rr conditions across the shared edge, in planar Bernstein form
        (Lai-Schumaker Thm 2.28), written in the frame of triangle 1 with
        triangle 2's coefficients relabelled by the pairing."""
        d = self.d
        pos = {a: k for k, a in enumerate(self.idx)}
        # domain point correspondence on the edge: map t2's domain points into
        # t1's frame and match to t1's edge points
        P1 = self.V[list(self.tris[t1])]
        # orientation of t2's edge relative to t1's edge
        rowsout = []
        # coefficient indexing: in triangle 1, edge e1 is opposite vertex e1; the
        # C^r condition involves coefficients of t1 with index e1-component = rr
        # and coefficients of t2 with index e2-component = rr expanded by beta.
        # Use the generic formulation: for each multi-index gamma of t1 with
        # gamma[e1] = rr (rows), c2[gamma with e2 slot] = sum_{|mu|=rr} binom
        # ... Implemented via evaluation identity instead: derivative matching.
        # We use the standard Bernstein C^r formula:
        # c2_{alpha + rr e_{e2}} = sum_{|mu| = rr} B^rr_mu(beta) c1_{alpha + mu}
        # where alpha ranges over indices with alpha[e1]=0, |alpha| = d - rr,
        # expressed in t1's frame (indices permuted so shared vertices match).
        perm = self._vertex_perm(t1, e1, t2, e2, g)
        for alpha in multi_indices(d - rr):
            if alpha[e1] != 0: continue
            row = []
            # t2 coefficient: index in t2's own numbering
            a2 = [0, 0, 0]
            for i in range(3):
                a2[perm[i]] = alpha[i]
            a2[e2] += rr
            row.append((t2 * self.Nloc + pos[tuple(a2)], 1.0))
            for mu in multi_indices(rr):
                coef = factorial(rr) / (factorial(mu[0]) * factorial(mu[1]) * factorial(mu[2])) * beta[0]**mu[0] * beta[1]**mu[1] * beta[2]**mu[2]
                a1 = tuple(alpha[i] + mu[i] for i in range(3))
                row.append((t1 * self.Nloc + pos[a1], -coef))
            rowsout.append(row)
        return rowsout

    def _vertex_perm(self, t1, e1, t2, e2, g):
        """perm[i] = local index in t2 of the vertex matching t1's local vertex i
        (perm[e1] = e2)."""
        P1 = self.V[list(self.tris[t1])]; P2 = self.V[list(self.tris[t2])]
        img = [np.array(g(p)) if g is not None else np.array(p) for p in P2]
        perm = [None] * 3
        perm[e1] = e2
        for i in range(3):
            if i == e1: continue
            for k in range(3):
                if k == e2: continue
                if np.allclose(P1[i], img[k], atol=1e-8): perm[i] = k
        assert None not in perm, 'vertex matching failed'
        return perm

    # --- assembly --------------------------------------------------------------
    def assemble(self, w=None, f=None, q=None):
        """broken stiffness K, weighted mass M_w, load F, constant vector c1."""
        d = self.d
        q = q or (d + 2)
        P, W = tri_quadrature(q)
        F = np.zeros(self.N)
        rows, cols, kv, mv = [], [], [], []
        for t, T in enumerate(self.tris):
            V = self.V[list(T)]
            J = np.array([V[1] - V[0], V[2] - V[0]]).T
            area = 0.5 * abs(np.linalg.det(J))
            # barycentric gradients
            A = np.vstack([V.T, np.ones(3)])
            Ainv = np.linalg.inv(A)
            grad_b = Ainv[:, :2]              # rows: grad b_i
            b = np.vstack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]])
            X = V[0][:, None] * b[0] + V[1][:, None] * b[1] + V[2][:, None] * b[2]
            B = bernstein(d, b)
            G = bernstein_grad(d, b, grad_b)
            wq = W * 2 * area
            ww = wq * (w(X) if w is not None else 1.0)
            Kt = np.einsum('iax,jax,x->ij', G, G, wq)
            Mt = np.einsum('ix,jx,x->ij', B, B, ww)
            sl = slice(t * self.Nloc, (t + 1) * self.Nloc)
            ii, jj = np.meshgrid(range(t * self.Nloc, (t + 1) * self.Nloc), range(t * self.Nloc, (t + 1) * self.Nloc), indexing='ij')
            rows += list(ii.ravel()); cols += list(jj.ravel()); kv += list(Kt.ravel()); mv += list(Mt.ravel())
            if f is not None:
                F[sl] = B @ (ww * f(X))
        K = sp.csr_matrix((kv, (rows, cols)), shape=(self.N, self.N))
        M = sp.csr_matrix((mv, (rows, cols)), shape=(self.N, self.N))
        c1 = np.ones(self.N)   # planar Bernstein: constant has all coefficients 1
        return K, M, F, c1

    def nullspace(self):
        if self.r == 0 and self.J.shape[0] == 0:
            return sp.identity(self.N, format='csr')
        Z = rr_prolongation(self.J)[0] if self.J.shape[0] else sp.identity(self.N, format="csr")
        return sp.csr_matrix(Z)

    def eigen(self, k=12, w=None, q=None, sigma=-1.0):
        K, M, F, c1 = self.assemble(w=w, q=q)
        Z = self.nullspace()
        Kz = (Z.T @ K @ Z).tocsc(); Mz = (Z.T @ M @ Z).tocsc()
        vals, vecs = spla.eigsh(Kz, k=k, M=Mz, sigma=sigma, which='LM')
        o = np.argsort(vals)
        return vals[o], Z @ vecs[:, o], Z

    def poisson(self, f, w=None, q=None):
        K, M, F, c1 = self.assemble(w=w, f=f, q=q)
        Z = self.nullspace()
        Kz = Z.T @ K @ Z; g = Z.T @ (M @ c1); b = Z.T @ F
        n = Kz.shape[0]
        A = sp.bmat([[Kz, g[:, None]], [g[None, :], None]]).tocsc()
        sol = spla.spsolve(A, np.concatenate([b, [0.0]]))
        return Z @ sol[:n], sol[n]

    def evaluate(self, c, X):
        """evaluate a broken-coefficient spline at points X (n,2): locate triangle."""
        out = np.empty(len(X))
        for p_i, p in enumerate(X):
            for t, T in enumerate(self.tris):
                V = self.V[list(T)]
                A = np.vstack([V.T, np.ones(3)])
                bb = np.linalg.solve(A, np.array([p[0], p[1], 1.0]))
                if bb.min() > -1e-10:
                    out[p_i] = bernstein(self.d, bb[:, None])[:, 0] @ c[t * self.Nloc:(t + 1) * self.Nloc]
                    break
        return out

    def errors(self, c, u, grad_u, w=None, q=None, meanfree=True):
        """weighted L2(M) error and flat H1 seminorm error of the broken spline
        with coefficients c against u (callable on (2,n) arrays) and grad_u
        (callable returning (2,n)); both made w-mean-free if meanfree."""
        d = self.d; q = q or (d + 4)
        P, W = tri_quadrature(q)
        e0 = e1 = 0.0; mu = 0.0; mh = 0.0; area = 0.0
        blocks = []
        for t, T in enumerate(self.tris):
            V = self.V[list(T)]
            J = np.array([V[1] - V[0], V[2] - V[0]]).T; a = 0.5 * abs(np.linalg.det(J))
            A = np.vstack([V.T, np.ones(3)]); grad_b = np.linalg.inv(A)[:, :2]
            b = np.vstack([1 - P[:, 0] - P[:, 1], P[:, 0], P[:, 1]])
            X = V[0][:, None] * b[0] + V[1][:, None] * b[1] + V[2][:, None] * b[2]
            B = bernstein(d, b); G = bernstein_grad(d, b, grad_b)
            cc = c[t * self.Nloc:(t + 1) * self.Nloc]
            wq = W * 2 * a; ww = wq * (w(X) if w is not None else 1.0)
            uh = cc @ B; gh = np.einsum('i,iax->ax', cc, G)
            ue = u(X); ge = grad_u(X)
            blocks.append((ww, wq, uh, ue, gh, ge))
            mu += ww @ ue; mh += ww @ uh; area += ww.sum()
        for ww, wq, uh, ue, gh, ge in blocks:
            du = (uh - mh / area) - (ue - mu / area) if meanfree else uh - ue
            e0 += ww @ du**2
            e1 += wq @ ((gh - ge)**2).sum(0)
        return np.sqrt(e0), np.sqrt(e1)

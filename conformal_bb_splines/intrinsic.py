"""
=============================================================================
SPLINES ON AN INTRINSIC HYPERBOLIC TRIANGULATION
=============================================================================

The existing pipeline needs a fundamental polygon: a global set of hyperboloid
points, side pairings, and a geometric merge of coincident domain points.
Uniformisation does not produce that. What it produces is a TRIANGULATION with
hyperbolic edge lengths and every vertex angle sum equal to 2 pi.

That is enough, and it is in fact the more natural input. Place each triangle in
H^2 on its own from its three edge lengths, and give every interior edge the
isometry that carries the neighbour's placement into this one. Then

  * the dof identification is purely COMBINATORIAL, no KD-tree and no tolerance;
  * the C^r rows are the ordinary ones, computed against the neighbour's fourth
    vertex pulled back by the edge transition, exactly as a side pairing is
    handled in the polygon case, which is the special case where the transition
    happens to be a deck transformation;
  * and the condition that makes the space well defined around a vertex is that
    the angles there sum to 2 pi. THAT IS THE UNIFORMISATION CONDITION ITSELF.
    In the polygon picture it appears as Poincare's corner-cycle condition; here
    it is what the Newton solve enforced.

So the geometry enters only through the edge lengths, and the polygon, the
generators and the vertex cycle all disappear.
=============================================================================
"""
import numpy as np

from hypgeom import J, mink, dist, pairing_map
from hypbb import bern_indices, bern_eval, local_matrices, constant_coeffs


# ------------------------------------------------------------- placement
def place(l):
    """
    Vertex matrices for every triangle from the side lengths l[f,a], where
    l[f,a] is the side OPPOSITE local vertex a. Returns (nf,3,3), columns are
    the hyperboloid positions of local vertices 0,1,2.
    """
    a, b, c = l[:, 0], l[:, 1], l[:, 2]      # |v1v2|, |v0v2|, |v0v1|
    cs = (np.cosh(b)*np.cosh(c) - np.cosh(a))/(np.sinh(b)*np.sinh(c))
    th = np.arccos(np.clip(cs, -1.0, 1.0))
    nf = l.shape[0]
    V = np.zeros((nf, 3, 3))
    V[:, 0, 0] = 1.0
    V[:, :, 1] = np.stack([np.cosh(c), np.sinh(c), np.zeros(nf)], axis=1)
    V[:, :, 2] = np.stack([np.cosh(b), np.sinh(b)*np.cos(th),
                           np.sinh(b)*np.sin(th)], axis=1)
    return V


# ------------------------------------------------------------- adjacency
def edge_table(faces, key_of=None):
    """
    for every interior edge: (f, m, f2, m2) where the edge is opposite local
    vertex m in face f and opposite m2 in face f2.

    Keying by the unordered vertex pair is only valid on a SIMPLICIAL complex.
    A quotient triangulation is a Delta-complex: on the Bolza octagon at level 1
    there are 14 vertices and 48 edges, so distinct edges share endpoints and
    the pair key collapses four face slots onto one edge. Pass key_of(f, m)
    from an independent edge identification in that case.
    """
    d = {}
    for f, tri in enumerate(faces):
        for m in range(3):
            if key_of is None:
                g1, g2 = tri[(m + 1) % 3], tri[(m + 2) % 3]
                key = (min(g1, g2), max(g1, g2))
            else:
                key = key_of(f, m)
            d.setdefault(key, []).append((f, m))
    out = []
    for key, v in d.items():
        if len(v) != 2:
            raise RuntimeError(f"edge {key} has {len(v)} faces")
        (f, m), (f2, m2) = v
        out.append((f, m, f2, m2))
    return out


def transitions(faces, V, edges):
    """
    gamma[e] carries the placement of face f2 into the frame of face f, so that
    the two shared endpoints coincide and the neighbour unfolds across the edge
    """
    G = []
    for (f, m, f2, m2) in edges:
        g1, g2 = faces[f][(m + 1) % 3], faces[f][(m + 2) % 3]
        # local slots of g1, g2 in f2
        j1, j2 = _slots(faces[f2], g1, g2, m2)
        A, B = V[f2][:, j1], V[f2][:, j2]
        Ap, Bp = V[f][:, (m + 1) % 3], V[f][:, (m + 2) % 3]
        g = pairing_map(A, B, Ap, Bp)
        # the unfolded neighbour apex must land on the far side of the edge
        w = g @ V[f2][:, m2]
        U = np.column_stack([V[f][:, m], Ap, Bp])
        b = np.linalg.solve(U, w)
        if b[0] > 0:
            g = pairing_map(B, A, Ap, Bp)
        G.append(g)
    return G


# ------------------------------------------------------------------ dofs
def _slots(tri2, g1, g2, m2):
    """local slots of the two shared vertices in the neighbour, excluding m2"""
    rest = [(m2 + 1) % 3, (m2 + 2) % 3]
    if tri2[rest[0]] == g1 and tri2[rest[1]] == g2:
        return rest[0], rest[1]
    if tri2[rest[0]] == g2 and tri2[rest[1]] == g1:
        return rest[1], rest[0]
    raise RuntimeError("shared edge does not match in the neighbour")


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dof_map(faces, edges, d):
    """
    combinatorial C^0 identification of the degree-d domain points.
    Returns (dof array of length nf*nloc, ndof).
    """
    idx = bern_indices(d)
    nloc = len(idx)
    pos = {e: k for k, e in enumerate(idx)}
    nf = len(faces)
    uf = UF(nf*nloc)

    # corners
    corner_of = {}
    for f, tri in enumerate(faces):
        for m in range(3):
            e = [0, 0, 0]
            e[m] = d
            k = pos[tuple(e)]
            v = int(tri[m])
            if v in corner_of:
                uf.union(corner_of[v], f*nloc + k)
            else:
                corner_of[v] = f*nloc + k

    # edge interiors
    for (f, m, f2, m2) in edges:
        g1 = faces[f][(m + 1) % 3]
        g2 = faces[f][(m + 2) % 3]
        j1, j2 = _slots(faces[f2], g1, g2, m2)
        for w1 in range(1, d):
            w2 = d - w1                      # weights on g1 and g2
            e = [0, 0, 0]
            e[(m + 1) % 3] = w1
            e[(m + 2) % 3] = w2
            e2 = [0, 0, 0]
            e2[j1] = w1
            e2[j2] = w2
            uf.union(f*nloc + pos[tuple(e)], f2*nloc + pos[tuple(e2)])

    roots = np.array([uf.find(i) for i in range(nf*nloc)])
    _, inv = np.unique(roots, return_inverse=True)
    return inv, int(inv.max()) + 1


# -------------------------------------------------------------- assembly
def assemble(faces, V, dof, ndof, d, nquad=14):
    from scipy.sparse import coo_matrix
    nloc = len(bern_indices(d))
    rows, cols, mv, kv = [], [], [], []
    area = 0.0
    for t in range(len(faces)):
        M, K = local_matrices(V[t], d, n=nquad)
        gl = dof[t*nloc:(t + 1)*nloc]
        rows.append(np.repeat(gl, nloc))
        cols.append(np.tile(gl, nloc))
        mv.append(M.ravel())
        kv.append(K.ravel())
        c = constant_coeffs(V[t], d)
        area += float(c @ M @ c)
    R = np.concatenate(rows)
    C = np.concatenate(cols)
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (R, C)), shape=(ndof, ndof)).tocsr()
    return Mg, Kg, area


# ------------------------------------------------------------ smoothness
def smoothness_rows(faces, V, G, edges, dof, ndof, d, r=1):
    """
    C^r rows across every interior edge. The neighbour's fourth vertex is
    pulled back by the edge transition and written in the barycentric frame of
    this triangle; the rows are then the ordinary Bernstein ones. C^0 is
    already imposed by the dof identification, so only r = 1 rows are emitted.
    """
    idx = bern_indices(d)
    nloc = len(idx)
    pos = {e: k for k, e in enumerate(idx)}
    out = []
    for e, (f, m, f2, m2) in enumerate(edges):
        g1 = faces[f][(m + 1) % 3]
        g2 = faces[f][(m + 2) % 3]
        j1, j2 = _slots(faces[f2], g1, g2, m2)
        w = G[e] @ V[f2][:, m2]                    # neighbour apex, unfolded
        # beta from  w = b0 u_m + b1 u_{m+1} + b2 u_{m+2}
        U = np.column_stack([V[f][:, m], V[f][:, (m + 1) % 3],
                             V[f][:, (m + 2) % 3]])
        beta = np.linalg.solve(U, w)
        for w1 in range(0, d):
            w2 = d - 1 - w1
            # neighbour coefficient one layer in from the edge
            e2 = [0, 0, 0]
            e2[m2] = 1
            e2[j1] = w1
            e2[j2] = w2
            col2 = f2*nloc + pos[tuple(e2)]
            # this triangle's three contributing coefficients
            a0 = [0, 0, 0]; a0[m] = 1; a0[(m+1) % 3] = w1; a0[(m+2) % 3] = w2
            a1 = [0, 0, 0]; a1[(m+1) % 3] = w1 + 1; a1[(m+2) % 3] = w2
            a2 = [0, 0, 0]; a2[(m+1) % 3] = w1; a2[(m+2) % 3] = w2 + 1
            cols = [f*nloc + pos[tuple(a0)], f*nloc + pos[tuple(a1)],
                    f*nloc + pos[tuple(a2)]]
            out.append((col2, cols, beta))
    S = np.zeros((len(out), ndof))
    for i, (col2, cols, beta) in enumerate(out):
        S[i, dof[col2]] += 1.0
        S[i, dof[cols[0]]] -= beta[0]
        S[i, dof[cols[1]]] -= beta[1]
        S[i, dof[cols[2]]] -= beta[2]
    return S

"""
hypmesh.py -- geodesic triangulation of the Bolza octagon and global C^0 assembly.

Geometry is EXACT at every refinement level: subdividing a geodesic triangle at
geodesic midpoints keeps the three sub-edges along the original geodesics, so the
union of the four children is the parent exactly.  There is no geometric error to
account for, unlike the conformally transplanted genus-zero setting.

Boundary identification: refinement is isometry-equivariant, so the subdivision nodes
of side j map exactly onto those of side j+4 under the pairing g_j (with a reversal,
since g_j sends v_j -> v_{j+5} and v_{j+1} -> v_{j+4}).  Node merging is therefore
exact matching, not interpolation.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from hypgeom import (dist, geodesic_point, bolza_octagon, side_maps, inv_lorentz,
                     normalize_point, triangle_area_defect)
from hypbb import bern_indices, vertex_matrix, local_matrices, constant_coeffs


class UnionFind:
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


def octagon_mesh(level):
    """Fan triangulation of the Bolza octagon, uniformly refined `level` times.

    Returns (tris, V0, gens): tris is a list of 3x3 vertex matrices."""
    V0, gens, R = bolza_octagon()
    O = np.array([1.0, 0.0, 0.0])
    tris = [np.column_stack([O, V0[j], V0[(j + 1) % 8]]) for j in range(8)]
    for _ in range(level):
        out = []
        for T in tris:
            a, b, c = T[:, 0], T[:, 1], T[:, 2]
            ab = geodesic_point(a, b, 0.5)
            bc = geodesic_point(b, c, 0.5)
            ca = geodesic_point(c, a, 0.5)
            out += [np.column_stack([a, ab, ca]),
                    np.column_stack([ab, b, bc]),
                    np.column_stack([ca, bc, c]),
                    np.column_stack([ab, bc, ca])]
        tris = out
    return tris, V0, gens


def domain_points(T, d):
    """Physical positions of the degree-d domain points of a triangle."""
    idx = np.array(bern_indices(d), float) / d
    return normalize_point((T @ idx.T).T)


def on_side(X, V0, j, tol=1e-9):
    """Boolean mask: which points of X lie on the geodesic side j."""
    a, b = V0[j], V0[(j + 1) % 8]
    L = dist(a, b)
    return np.abs(dist(X, a) + dist(X, b) - L) < tol


def build_dofs(tris, V0, gens, d, tol=1e-8):
    """Global C^0 numbering: merge coincident domain points, then apply the pairings."""
    P = np.vstack([domain_points(T, d) for T in tris])
    n = P.shape[0]
    uf = UnionFind(n)

    tree = cKDTree(P)
    for a, b in tree.query_pairs(tol):
        uf.union(a, b)

    g = side_maps(gens)
    for j in range(8):
        m_src = on_side(P, V0, j)
        m_dst = on_side(P, V0, (j + 4) % 8)
        if not m_src.any() or not m_dst.any():
            continue
        src = np.where(m_src)[0]
        dst = np.where(m_dst)[0]
        img = (g[j] @ P[src].T).T
        t2 = cKDTree(P[dst])
        dd, ii = t2.query(img)
        if dd.max() > 1e-7:
            raise RuntimeError(f"side {j}: unmatched boundary node, max gap {dd.max():.2e}")
        for a, b in zip(src, dst[ii]):
            uf.union(a, b)

    roots = np.array([uf.find(i) for i in range(n)])
    _, inv = np.unique(roots, return_inverse=True)
    return inv, P


def assemble(level, d, nquad=14):
    """Global mass and stiffness for S^0_d on the Bolza surface."""
    tris, V0, gens = octagon_mesh(level)
    dof, P = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d))
    ndof = int(dof.max()) + 1

    rows, cols, mv, kv = [], [], [], []
    area = 0.0
    for t, T in enumerate(tris):
        M, K = local_matrices(T, d, n=nquad)
        gl = dof[t * nloc:(t + 1) * nloc]
        R = np.repeat(gl, nloc)
        C = np.tile(gl, nloc)
        rows.append(R)
        cols.append(C)
        mv.append(M.ravel())
        kv.append(K.ravel())
        c = constant_coeffs(T, d)
        area += float(c @ M @ c)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    Mg = coo_matrix((np.concatenate(mv), (rows, cols)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (rows, cols)), shape=(ndof, ndof)).tocsr()
    return Mg, Kg, area, len(tris), ndof


def global_constant(level, d):
    """Global coefficient vector of the constant 1 (even d only)."""
    tris, V0, gens = octagon_mesh(level)
    dof, P = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d))
    u = np.zeros(int(dof.max()) + 1)
    for t, T in enumerate(tris):
        u[dof[t * nloc:(t + 1) * nloc]] = constant_coeffs(T, d)
    return u

"""General side-pairing surfaces: the pairing pattern, not just the polygon, fixes
the surface.  Includes Klein's 14-gon pairing (1-6, 3-8, 5-10, 7-12, 9-14, 11-2, 13-4)."""
import numpy as np
from math import gcd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from hypgeom import (regular_polygon, pairing_map, inv_lorentz, dist, geodesic_point,
                     triangle_area_defect, angle_at)
from hypbb import bern_indices, local_matrices, constant_coeffs
from hypmesh import UnionFind, domain_points

OPPOSITE = lambda n: [(j, j + n // 2) for j in range(n // 2)]
KLEIN14 = [(0, 5), (2, 7), (4, 9), (6, 11), (8, 13), (10, 1), (12, 3)]


def partner_map(n, pairs):
    p = {}
    for a, b in pairs:
        p[a % n] = b % n
        p[b % n] = a % n
    assert len(p) == n, "pairing must cover every side"
    return p


def vertex_classes(n, pairs):
    """Corner walk j -> partner(j)+1; returns the list of cycles."""
    p = partner_map(n, pairs)
    seen, cycles = set(), []
    for s in range(n):
        if s in seen:
            continue
        cyc, j = [], s
        while j not in seen:
            seen.add(j)
            cyc.append(j)
            j = (p[j] + 1) % n
        cycles.append(cyc)
    return cycles


def surface(n, pairs):
    cyc = vertex_classes(n, pairs)
    theta = 2 * np.pi * len(cyc) / n
    V, R = regular_polygon(n, theta)
    p = partner_map(n, pairs)
    g = [None] * n
    for a, b in pairs:
        g[a] = pairing_map(V[a], V[(a + 1) % n], V[(b + 1) % n], V[b])
        g[b] = inv_lorentz(g[a])
    chi = len(cyc) - n // 2 + 1
    return V, g, p, dict(n=n, classes=len(cyc), cycles=cyc, theta=theta,
                         chi=chi, genus=(2 - chi) // 2, R=R)


def mesh(n, pairs, level):
    V0, g, p, info = surface(n, pairs)
    O = np.array([1.0, 0.0, 0.0])
    tris = [np.column_stack([O, V0[j], V0[(j + 1) % n]]) for j in range(n)]
    for _ in range(level):
        out = []
        for T in tris:
            a, b, c = T[:, 0], T[:, 1], T[:, 2]
            ab, bc, ca = (geodesic_point(a, b, .5), geodesic_point(b, c, .5),
                          geodesic_point(c, a, .5))
            out += [np.column_stack([a, ab, ca]), np.column_stack([ab, b, bc]),
                    np.column_stack([ca, bc, c]), np.column_stack([ab, bc, ca])]
        tris = out
    return tris, V0, g, p, info


def build(n, pairs, level, d, nquad=18):
    tris, V0, g, p, info = mesh(n, pairs, level)
    P = np.vstack([domain_points(T, d) for T in tris])
    uf = UnionFind(P.shape[0])
    for a, b in cKDTree(P).query_pairs(1e-8):
        uf.union(a, b)

    def on_side(X, j):
        a, b = V0[j], V0[(j + 1) % n]
        return np.abs(dist(X, a) + dist(X, b) - dist(a, b)) < 1e-9

    for j in range(n):
        src = np.where(on_side(P, j))[0]
        dst = np.where(on_side(P, p[j]))[0]
        if src.size == 0:
            continue
        dd, ii = cKDTree(P[dst]).query((g[j] @ P[src].T).T)
        assert dd.max() < 1e-7, f"side {j}: gap {dd.max():.2e}"
        for a, b in zip(src, dst[ii]):
            uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(P.shape[0])])
    _, dof = np.unique(roots, return_inverse=True)

    nloc = len(bern_indices(d))
    ndof = int(dof.max()) + 1
    R, C, mv, kv = [], [], [], []
    area = 0.0
    for t, T in enumerate(tris):
        M, K = local_matrices(T, d, n=nquad)
        gl = dof[t * nloc:(t + 1) * nloc]
        R.append(np.repeat(gl, nloc))
        C.append(np.tile(gl, nloc))
        mv.append(M.ravel())
        kv.append(K.ravel())
        area += float(constant_coeffs(T, d) @ M @ constant_coeffs(T, d))
    R, C = np.concatenate(R), np.concatenate(C)
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (R, C)), shape=(ndof, ndof)).tocsr()
    return Mg, Kg, area, len(tris), ndof, info


def cycle_word(g, cycle):
    """Product of side maps around a corner cycle.  NOTE the composition order:
    the maps compose on the LEFT along the walk (equivalently, right-multiply in
    reverse order).  The opposite-side pairing is symmetric enough that either order
    gives the identity, which hides the convention; Klein's pairing does not, and the
    wrong order is off by ten orders of magnitude."""
    W = np.eye(3)
    for j in cycle:
        W = g[j] @ W
    return W

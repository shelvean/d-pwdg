"""
hypsurf.py -- regular hyperbolic n-gons with opposite-side identification, any genus.

For an n-gon (n even) with side j glued to side j+n/2 by v_j -> v_{j+n/2+1},
v_{j+1} -> v_{j+n/2}, the corner walk is j -> j + (n/2 + 1) mod n, so the number of
vertex CLASSES is c = gcd(n/2+1, n) and each class has n/c corners.  Each class must
carry total angle 2 pi, so the interior angle is theta = 2 pi c / n.  Then

    E = n/2,  F = 1,  V = c,   chi = c - n/2 + 1,   genus = (2 - chi)/2.

n = 8  -> c = gcd(5,8) = 1, theta = pi/4,   chi = -2, genus 2: the Bolza surface.
n = 14 -> c = gcd(8,14) = 2, theta = 2 pi/7, chi = -4, genus 3: an opposite-side genus-3 surface.

This is NOT the Klein quartic.  The Klein quartic uses a different pairing of the same
regular 14-gon; see klein.py.  The pairing pattern, not only the polygon, fixes the surface.

Nothing in the spline construction changes with genus.  Only n changes.
"""

import numpy as np
from math import gcd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from hypgeom import (dist, geodesic_point, regular_polygon, pairing_map, inv_lorentz,
                     normalize_point, triangle_area_defect)
from hypbb import bern_indices, local_matrices, constant_coeffs
from hypmesh import UnionFind, domain_points


def regular_surface(n):
    """Vertices, generators and topology of the genus-g surface from a regular n-gon."""
    assert n % 2 == 0
    c = gcd(n // 2 + 1, n)
    theta = 2 * np.pi * c / n
    V, R = regular_polygon(n, theta)
    gens = [pairing_map(V[j], V[(j + 1) % n],
                        V[(j + n // 2 + 1) % n], V[(j + n // 2) % n])
            for j in range(n // 2)]
    chi = c - n // 2 + 1
    return V, gens, R, dict(n=n, classes=c, theta=theta, chi=chi, genus=(2 - chi) // 2)


def side_maps_n(gens, n):
    return list(gens) + [inv_lorentz(m) for m in gens]


def polygon_mesh(n, level):
    V0, gens, R, info = regular_surface(n)
    O = np.array([1.0, 0.0, 0.0])
    tris = [np.column_stack([O, V0[j], V0[(j + 1) % n]]) for j in range(n)]
    for _ in range(level):
        out = []
        for T in tris:
            a, b, c = T[:, 0], T[:, 1], T[:, 2]
            ab = geodesic_point(a, b, 0.5)
            bc = geodesic_point(b, c, 0.5)
            ca = geodesic_point(c, a, 0.5)
            out += [np.column_stack([a, ab, ca]), np.column_stack([ab, b, bc]),
                    np.column_stack([ca, bc, c]), np.column_stack([ab, bc, ca])]
        tris = out
    return tris, V0, gens, info


def on_side_n(X, V0, j, n, tol=1e-9):
    a, b = V0[j], V0[(j + 1) % n]
    return np.abs(dist(X, a) + dist(X, b) - dist(a, b)) < tol


def build_dofs_n(tris, V0, gens, d, n, tol=1e-8):
    P = np.vstack([domain_points(T, d) for T in tris])
    uf = UnionFind(P.shape[0])
    for a, b in cKDTree(P).query_pairs(tol):
        uf.union(a, b)
    g = side_maps_n(gens, n)
    for j in range(n):
        src = np.where(on_side_n(P, V0, j, n))[0]
        dst = np.where(on_side_n(P, V0, (j + n // 2) % n, n))[0]
        if src.size == 0 or dst.size == 0:
            continue
        dd, ii = cKDTree(P[dst]).query((g[j] @ P[src].T).T)
        if dd.max() > 1e-7:
            raise RuntimeError(f"side {j}: gap {dd.max():.2e}")
        for a, b in zip(src, dst[ii]):
            uf.union(a, b)
    roots = np.array([uf.find(i) for i in range(P.shape[0])])
    _, inv = np.unique(roots, return_inverse=True)
    return inv, P


def assemble_n(n, level, d, nquad=16):
    tris, V0, gens, info = polygon_mesh(n, level)
    dof, _ = build_dofs_n(tris, V0, gens, d, n)
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
        c = constant_coeffs(T, d)
        area += float(c @ M @ c)
    R, C = np.concatenate(R), np.concatenate(C)
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (R, C)), shape=(ndof, ndof)).tocsr()
    return Mg, Kg, area, len(tris), ndof, info


from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .metric import CallableMetric
from .global_complex import FacetPairing


PHI = (1.0 + np.sqrt(5.0))/2.0


def qmul(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.ndim == 1:
        a0,a1,a2,a3 = a
    else:
        a0,a1,a2,a3 = a.T
    b0,b1,b2,b3 = b
    return np.stack([
        a0*b0-a1*b1-a2*b2-a3*b3,
        a0*b1+a1*b0+a2*b3-a3*b2,
        a0*b2-a1*b3+a2*b0+a3*b1,
        a0*b3+a1*b2-a2*b1+a3*b0
    ], axis=-1)


def left_matrix(g):
    return np.stack([qmul(g, e) for e in np.eye(4)], axis=1)


def icosians():
    pts = []
    for i in range(4):
        for s in (1,-1):
            e = np.zeros(4)
            e[i] = s
            pts.append(e)

    for s in itertools.product((0.5,-0.5), repeat=4):
        pts.append(np.array(s))

    even = [
        p for p in itertools.permutations(range(4))
        if sum(p[i] > p[j] for i in range(4) for j in range(i+1,4)) % 2 == 0
    ]
    base = np.array([PHI, 1.0, 1.0/PHI, 0.0])/2.0
    for p in even:
        for s in itertools.product((1,-1), repeat=3):
            v = base.copy()
            v[:3] *= s
            w = np.zeros(4)
            w[list(p)] = v
            pts.append(w)
    return np.unique(np.round(np.asarray(pts), 12), axis=0)


def refine_spherical_tetrahedra(cells):
    out = []
    for V in cells:
        v = [V[:,i] for i in range(4)]
        m = {
            (i,j):(v[i]+v[j])/np.linalg.norm(v[i]+v[j])
            for i in range(4) for j in range(i+1,4)
        }

        for i in range(4):
            out.append(
                np.stack(
                    [v[i]] +
                    [m[tuple(sorted((i,j)))] for j in range(4) if j != i],
                    axis=1
                )
            )

        diag = min(
            [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))],
            key=lambda p: np.linalg.norm(m[p[0]]-m[p[1]])
        )
        a,c = diag
        eq = [e for e in m if e not in (a,c)]
        order = [eq[0]]
        while len(order) < 4:
            for e in eq:
                if e not in order and len(set(e)&set(order[-1])) == 1:
                    order.append(e)
                    break
        for j in range(4):
            out.append(
                np.stack([m[a],m[c],m[order[j]],m[order[(j+1)%4]]],axis=1)
            )
    return out


class SphereRadialMetric(CallableMetric):
    """
    Pullback of the unit-round S^3 metric by radial projection of an affine
    tetrahedron in R^4.

    The FEM sees only the resulting 3x3 tensor G_K(xhat).
    """

    def __init__(self, vertex_matrix):
        V = np.asarray(vertex_matrix, float)
        if V.shape != (4,4):
            raise ValueError("vertex_matrix must be 4x4")
        self.vertex_matrix = V.copy()
        Dy = V[:,1:] - V[:,[0]]

        def metric_fn(xhat):
            xhat = np.asarray(xhat, float)
            b = np.r_[1.0-np.sum(xhat), xhat]
            y = V @ b
            r = np.linalg.norm(y)
            Jrad = np.eye(4)/r - np.outer(y,y)/(r**3)
            J = Jrad @ Dy
            G = J.T @ J
            return 0.5*(G+G.T)

        super().__init__(3, metric_fn)


def _canonical_face_key(face_points, transforms, decimals=9):
    best = None
    for L in transforms:
        P = np.round(face_points @ L.T, decimals)
        rows = tuple(sorted(map(tuple, P.tolist())))
        if best is None or rows < best:
            best = rows
    return best


def pair_facets_by_group(cells, transforms, decimals=9, tol=1e-8):
    """
    Pair every tetrahedral half-facet modulo a finite linear group action.

    The returned vertex permutation is the affine reference-facet map used by
    the finite element trace constraint.
    """
    halves = []
    groups = {}

    for ci,V in enumerate(cells):
        pts = V.T
        for f in range(4):
            loc = [j for j in range(4) if j != f]
            face = pts[loc]
            key = _canonical_face_key(face, transforms, decimals)
            h = len(halves)
            halves.append((ci,f,loc,face))
            groups.setdefault(key,[]).append(h)

    sizes = {len(v) for v in groups.values()}
    if sizes != {2}:
        bad = {m:sum(len(v)==m for v in groups.values()) for m in sorted(sizes)}
        raise RuntimeError(f"facet orbits are not pairs: {bad}")

    pairings = []
    for ids in groups.values():
        ia, ib = ids
        ca,fa,la,Aface = halves[ia]
        cb,fb,lb,Bface = halves[ib]

        permutation = None
        for L in transforms:
            image = Bface @ L.T
            p = []
            ok = True
            for a in Aface:
                d = np.linalg.norm(image-a, axis=1)
                j = int(np.argmin(d))
                if d[j] > tol or j in p:
                    ok = False
                    break
                p.append(j)
            if ok:
                permutation = tuple(p)
                break

        if permutation is None:
            raise RuntimeError("could not recover reference-facet map")

        pairings.append(
            FacetPairing(ca,fa,cb,fb,permutation)
        )

    return pairings


def poincare_homology_sphere(level=0):
    """
    Five-tetrahedron quotient of the geodesic 600-cell by the binary
    icosahedral group, optionally followed by symmetric geodesic refinement.
    """
    G = icosians()
    if len(G) != 120:
        raise RuntimeError("600-cell vertex set is incomplete")

    transforms = [left_matrix(g) for g in G]
    tree = cKDTree(G)

    hull = ConvexHull(G)
    tets = sorted(set(tuple(sorted(map(int,t))) for t in hull.simplices))

    seen = set()
    reps = []
    for t in tets:
        if t in seen:
            continue
        reps.append(t)
        for L in transforms:
            ids = tuple(sorted(map(int, tree.query((L @ G[list(t)].T).T)[1])))
            seen.add(ids)

    if len(reps) != 5 or len(seen) != 600:
        raise RuntimeError("unexpected 600-cell quotient orbit count")

    cells = []
    for t in reps:
        V = G[list(t)].T.copy()
        if np.linalg.det(V) < 0:
            V[:,[0,1]] = V[:,[1,0]]
        cells.append(V)

    for _ in range(int(level)):
        cells = refine_spherical_tetrahedra(cells)

    pairings = pair_facets_by_group(cells, transforms)
    metrics = [SphereRadialMetric(V) for V in cells]

    return {
        "cells": cells,
        "pairings": pairings,
        "metrics": metrics,
        "group_transforms": transforms,
        "exact_volume": 2*np.pi**2/120.0,
    }

"""Basic mesh utilities used by the direction-adaptive PWDG experiments.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

Notes
-----
This module intentionally contains only the small amount of mesh bookkeeping
needed by the reproducibility scripts.  The numerical experiments use a
structured triangulation of a square.  More elaborate adaptive meshes are
constructed directly in the corresponding experiment script.

The flux constants ``ALPHA`` and ``BETA`` are the symmetric values used in the
all-Dirichlet experiments in the manuscript.
"""

from __future__ import annotations

import numpy as np

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ALPHA = 0.5
BETA = 0.5


def square_mesh(n:
    int, box: tuple[float, float] = (-1.0, 1.0)):
    """Return a uniform triangular mesh of a square.

    Parameters
    ----------
    n:
        Number of square cells in each coordinate direction.  Every square is
        divided into two triangles, so the returned mesh has ``2*n**2``
        triangles.
    box:
        Pair ``(a,b)`` defining the square ``[a,b] x [a,b]``.

    Returns
    -------
    vertices, triangles:
        ``vertices`` is an ``(Nv,2)`` array of coordinates and ``triangles``
        is an ``(Nt,3)`` array of vertex indices.
    """
    a, b = box
    xs = np.linspace(a, b, n + 1)
    ys = np.linspace(a, b, n + 1)
    vertices = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def vertex_id(i:
        int, j: int) -> int:
        return j * (n + 1) + i

    triangles = []
    for j in range(n):
        for i in range(n):
            v00 = vertex_id(i, j)
            v10 = vertex_id(i + 1, j)
            v01 = vertex_id(i, j + 1)
            v11 = vertex_id(i + 1, j + 1)
            triangles.append((v00, v10, v11))
            triangles.append((v00, v11, v01))

    return vertices, np.asarray(triangles, dtype=int)


def edges(triangles:
    np.ndarray):
    """Construct the edge-to-element connectivity of a triangular mesh.

    Boundary edges are represented by ``[element,-1]``.  Interior edges have
    two adjacent element indices.  The orientation of an edge is not used to
    define jumps
    each experiment reconstructs the outward normal from the
    element centroid.
    """
    edge_map: dict[tuple[int, int], int] = {}
    edge_vertices = []
    edge_elements = []

    for element, triangle in enumerate(triangles):
        local_edges = (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        )
        for a, b in local_edges:
            key = (min(a, b), max(a, b))
            if key in edge_map:
                edge_elements[edge_map[key]][1] = element
            else:
                edge_map[key] = len(edge_vertices)
                edge_vertices.append(key)
                edge_elements.append([element, -1])

    return np.asarray(edge_vertices, dtype=int), np.asarray(edge_elements, dtype=int)

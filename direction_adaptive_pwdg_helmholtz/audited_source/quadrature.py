"""Quadrature rules used by the solver and the error computations.

Two rules are needed.  Edge integrals use Gauss-Legendre on [0,1], since
every integrand on an edge is a product of complex exponentials and is
smooth.  Element integrals are used only for measuring errors against an
exact solution; the integrand there oscillates at the wavenumber, so a
single fixed rule on the element is not enough and we subdivide.
"""

import numpy as np


def gauss01(n):
    """n-point Gauss-Legendre nodes and weights on [0, 1], weights sum to 1."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def triangle_rule(nsub):
    """
    Composite rule on the reference triangle in barycentric coordinates.

    The triangle is split into nsub**2 congruent subtriangles and the
    three-point midpoint rule, exact for quadratics, is applied on each.
    Returns barycentric coordinates lam of shape (m, 3) and weights w of
    shape (m,) that sum to one, so that

        int_K f dA = area(K) * sum_i w_i f(lam_i . vertices).
    """
    verts = []
    for i in range(nsub):
        for j in range(nsub - i):
            # "upward" subtriangle
            verts.append([(i, j), (i + 1, j), (i, j + 1)])
            # "downward" subtriangle, present except along the far edge
            if i + j < nsub - 1:
                verts.append([(i + 1, j), (i, j + 1), (i + 1, j + 1)])

    def bary(node):
        i, j = node
        return np.array([1.0 - (i + j) / nsub, i / nsub, j / nsub])

    lam, w = [], []
    wsub = 1.0 / (3.0 * nsub ** 2)          # area fraction / 3 points
    for tri in verts:
        a, b, c = (bary(v) for v in tri)
        lam.extend([0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)])
        w.extend([wsub, wsub, wsub])
    return np.array(lam), np.array(w)

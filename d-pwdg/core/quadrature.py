"""Quadrature rules used by the PWDG reproducibility code.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

The edge integrals use Gauss-Legendre quadrature on ``[0,1]``.  Area error
measurements use a tensor-product rule mapped to the reference triangle.  The
Trefftz matrices themselves are assembled from edge integrals.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

__author__ = "Shelvean Kapita"
__date__ = "August 2026"


def gauss01(order:
    int):
    """Gauss-Legendre nodes and weights on the unit interval."""
    x, w = leggauss(order)
    return (x + 1.0) / 2.0, w / 2.0


def triangle_rule(order:
    int):
    """Product Gauss rule on the reference triangle.

    The mapping ``(x,y) -> (x,(1-x)y)`` converts the unit square to the
    reference triangle.  The returned coordinates are barycentric and the
    weights are normalized for the reference triangle convention used in the
    error routine.
    """
    x, wx = gauss01(max(2, order))
    y, wy = gauss01(max(2, order))

    barycentric = []
    weights = []
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            l2 = xi
            l3 = (1.0 - xi) * yj
            l1 = 1.0 - l2 - l3
            barycentric.append((l1, l2, l3))
            weights.append(2.0 * wx[i] * wy[j] * (1.0 - xi))

    return np.asarray(barycentric, dtype=float), np.asarray(weights, dtype=float)

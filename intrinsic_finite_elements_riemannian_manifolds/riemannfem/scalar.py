
from __future__ import annotations
import numpy as np
from .reference import simplex_duffy, SimplexBernstein


def scalar_local_matrices(space: SimplexBernstein, metric, q=None):
    """Mass and Laplace--Beltrami stiffness matrices on one cell.

    The cell is already represented on the reference simplex.  Geometry enters
    only through G_K(xhat).
    """
    if metric.dim != space.n:
        raise ValueError("metric and reference simplex dimensions differ")
    q = int(q or max(space.p + 3, 5))
    lams, ws = simplex_duffy(space.n, q)
    M = np.zeros((space.nloc, space.nloc))
    K = np.zeros_like(M)

    for lam, w in zip(lams, ws):
        xhat = lam[1:]
        val, grad = space.values_grads(lam)
        G = np.asarray(metric.metric(xhat), dtype=float)
        Gi = np.linalg.inv(G)
        mu = np.sqrt(np.linalg.det(G))
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("metric must be positive definite")
        M += w*mu*np.outer(val, val)
        K += w*mu*(grad @ Gi @ grad.T)
    return M, K

"""Vectorized metric mass matrix for trimmed polynomial k-forms.

The reference basis values at the quadrature points do not depend on the cell, so
they are tabulated once per (space, q).  Only the metric changes from cell to cell.
The local mass matrix is then one contraction,

    M_ij = sum_q  w_q sqrt(det G_q)  V_q[i,:] H_q V_q[j,:]^T ,

which is formed as T = V H (batched) followed by a single matrix product, so the work
per cell is one BLAS call of size nloc x (nq ncomp) x nloc instead of nq Python-level
rank updates.  The result agrees with riemannfem.forms.local_hodge_mass to rounding.
"""
import numpy as np
from riemannfem.forms import wedge_metric_matrix
from riemannfem.reference import simplex_duffy

_TAB = {}


def tabulate(space, q):
    key = (space.n, space.r, space.k, q)
    if key not in _TAB:
        lams, ws = simplex_duffy(space.n, q)
        pts = np.array([lam[1:] for lam in lams])
        B = np.array([space.values(x) for x in pts])        # (nq, nloc, ncomp)
        _TAB[key] = (pts, np.asarray(ws, float), B)
    return _TAB[key]


def local_hodge_mass_fast(space, metric, q=None):
    q = int(q or max(space.r + 3, 5))
    pts, ws, B = tabulate(space, q)
    G = np.array([np.asarray(metric.metric(x), float) for x in pts])   # (nq, n, n)
    Gi = np.linalg.inv(G)
    mu = np.sqrt(np.linalg.det(G))
    H = np.array([wedge_metric_matrix(g, space.k) for g in Gi])       # (nq, c, c)
    W = (ws * mu)[:, None, None] * H
    T = np.einsum('qic,qcd->iqd', B, W, optimize=True)                # (nloc, nq, c)
    nloc = B.shape[1]
    M = T.reshape(nloc, -1) @ B.transpose(1, 0, 2).reshape(nloc, -1).T
    return 0.5 * (M + M.T)

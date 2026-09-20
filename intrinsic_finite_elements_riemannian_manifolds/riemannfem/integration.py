
from __future__ import annotations
import numpy as np
from .reference import simplex_duffy


def cell_volume(metric, q=8):
    lam, w = simplex_duffy(metric.dim, int(q))
    val = 0.0
    for l, wi in zip(lam, w):
        G = np.asarray(metric.metric(l[1:]), float)
        val += wi*np.sqrt(np.linalg.det(G))
    return float(val)


def mesh_volume(metrics, q=8):
    return float(sum(cell_volume(g,q=q) for g in metrics))

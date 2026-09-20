
from __future__ import annotations
from typing import Protocol, Callable
import numpy as np


class PullbackMetric(Protocol):
    """Metric provider on one reference cell.

    metric(xhat) returns the n x n Riemannian metric tensor G_K(xhat)
    in reference coordinates.  No ambient embedding is part of this API.
    """
    dim: int
    def metric(self, xhat: np.ndarray) -> np.ndarray: ...


class ConstantMetric:
    def __init__(self, G):
        G = np.asarray(G, dtype=float)
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise ValueError("G must be square")
        self.G = G
        self.dim = G.shape[0]

    def metric(self, xhat):
        return self.G


class CallableMetric:
    def __init__(self, dim: int, fn: Callable[[np.ndarray], np.ndarray]):
        self.dim = int(dim)
        self.fn = fn

    def metric(self, xhat):
        G = np.asarray(self.fn(np.asarray(xhat, dtype=float)), dtype=float)
        if G.shape != (self.dim, self.dim):
            raise ValueError("metric callback returned wrong shape")
        return G

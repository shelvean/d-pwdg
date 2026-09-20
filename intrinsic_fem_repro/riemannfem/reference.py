
from __future__ import annotations
import itertools
import math
import numpy as np


def multi_indices(n: int, p: int):
    """All (n+1)-component multi-indices alpha with |alpha|=p."""
    out = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for a in range(remaining, -1, -1):
            rec(prefix + [a], remaining-a, slots-1)
    rec([], p, n+1)
    return out


def simplex_duffy(n: int, q: int):
    """Tensor Gauss rule on the reference n-simplex.

    Reference coordinates are x=(x1,...,xn), xi>=0, sum xi<=1.
    We also return barycentric coordinates lambda=(lambda0,...,lambdan).
    """
    z, w = np.polynomial.legendre.leggauss(q)
    t = (z + 1.0)/2.0
    w = w/2.0

    lambdas = []
    weights = []
    for ids in itertools.product(range(q), repeat=n):
        ts = np.array([t[i] for i in ids], dtype=float)
        ws = np.prod([w[i] for i in ids])
        rem = 1.0
        x = np.empty(n)
        jac = 1.0
        for j in range(n):
            # Duffy map:
            # x_j = t_j prod_{m<j}(1-t_m).
            # Hence det J = prod_{j=1}^{n-1} prod_{m<j}(1-t_m).
            if j > 0:
                jac *= rem
            x[j] = rem * ts[j]
            rem *= (1.0-ts[j])
        lam = np.empty(n+1)
        lam[0] = rem
        lam[1:] = x
        lambdas.append(lam)
        weights.append(ws*jac)
    return np.asarray(lambdas), np.asarray(weights)


class SimplexBernstein:
    """Degree-p Bernstein basis on the reference n-simplex."""

    def __init__(self, n: int, p: int):
        self.n = int(n)
        self.p = int(p)
        self.indices = multi_indices(self.n, self.p)
        self.nloc = len(self.indices)

    def values_grads(self, lam):
        """Values and gradients wrt x=(lambda1,...,lambdan)."""
        lam = np.asarray(lam, dtype=float)
        vals = np.zeros(self.nloc)
        grads = np.zeros((self.nloc, self.n))
        # d lambda_0 / dx_j = -1, d lambda_j / dx_j = 1
        dlam = np.zeros((self.n+1, self.n))
        dlam[0, :] = -1.0
        for j in range(self.n):
            dlam[j+1, j] = 1.0

        pfac = math.factorial(self.p)
        for aidx, alpha in enumerate(self.indices):
            c = pfac
            for a in alpha:
                c /= math.factorial(a)

            v = c
            for i, a in enumerate(alpha):
                if a:
                    v *= lam[i]**a
            vals[aidx] = v

            for i, a in enumerate(alpha):
                if a == 0:
                    continue
                term = c*a
                for m, am in enumerate(alpha):
                    power = am - (1 if m == i else 0)
                    if power:
                        term *= lam[m]**power
                grads[aidx] += term*dlam[i]
        return vals, grads


from __future__ import annotations

"""
Algebraic high-order trimmed polynomial differential forms.

This module constructs
    P_r^- Lambda^k(R^n)
directly from the FEEC identity
    P_r^- Lambda^k = P_{r-1} Lambda^k + kappa H_{r-1} Lambda^{k+1}.

The representation is intrinsic to the reference simplex.  No ambient
embedding of the manifold is used.

The basis produced here is an algebraic basis selected from a natural spanning
set.  A Bernstein-conditioned change of basis can be layered on top later
without changing the space, exterior derivative, traces, or pullbacks.
"""

import itertools
import math
import numpy as np
import scipy.linalg as la

from .reference import simplex_duffy


def comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def trimmed_dimension(n: int, r: int, k: int) -> int:
    """dim P_r^- Lambda^k in n dimensions."""
    if r < 1:
        raise ValueError("trimmed family uses r >= 1")
    if not (0 <= k <= n):
        return 0
    return comb(r + k - 1, k) * comb(n + r, n - k)


def monomial_indices(n: int, degree: int, homogeneous=False):
    """Multi-indices alpha in N^n."""
    out = []
    if homogeneous:
        def rec(prefix, rem, slots):
            if slots == 1:
                out.append(tuple(prefix + [rem]))
                return
            for a in range(rem, -1, -1):
                rec(prefix + [a], rem-a, slots-1)
        rec([], degree, n)
        return out

    for d in range(degree + 1):
        out.extend(monomial_indices(n, d, homogeneous=True))
    return out


def wedge_indices(n: int, k: int):
    return list(itertools.combinations(range(n), k))


def _wedge_sign(seq):
    """Sign needed to sort a sequence with distinct entries."""
    inv = 0
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            inv += seq[i] > seq[j]
    return -1 if inv % 2 else 1


def _poly_add(a, b, scale=1.0):
    out = dict(a)
    for alpha, v in b.items():
        out[alpha] = out.get(alpha, 0.0) + scale*v
        if abs(out[alpha]) < 1e-14:
            out.pop(alpha, None)
    return out


def _poly_mul(a, b):
    out = {}
    for aa, av in a.items():
        for bb, bv in b.items():
            cc = tuple(x+y for x,y in zip(aa,bb))
            out[cc] = out.get(cc, 0.0) + av*bv
    return out


def _poly_pow_linear(b, row, power):
    """(b + row dot y)^power as sparse polynomial."""
    n = len(row)
    z = (0,)*n
    out = {z: 1.0}
    lin = {z: float(b)}
    for j, a in enumerate(row):
        if abs(a) > 0:
            e = [0]*n
            e[j] = 1
            lin[tuple(e)] = float(a)
    for _ in range(power):
        out = _poly_mul(out, lin)
    return out


def polynomial_pullback(alpha, A, b):
    """Pull back x^alpha under x=A y+b."""
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    m = A.shape[1]
    out = {(0,)*m: 1.0}
    for i, p in enumerate(alpha):
        if p:
            out = _poly_mul(out, _poly_pow_linear(b[i], A[i], p))
    return out


class AmbientPolynomialForms:
    """Coefficient space P_degree Lambda^k in coordinate monomial basis."""

    def __init__(self, n, degree, k):
        self.n = int(n)
        self.degree = int(degree)
        self.k = int(k)
        self.monomials = monomial_indices(n, degree)
        self.wedges = wedge_indices(n, k)
        self.keys = [(a,I) for a in self.monomials for I in self.wedges]
        self.index = {q:i for i,q in enumerate(self.keys)}
        self.dim = len(self.keys)

    def vector(self, terms):
        v = np.zeros(self.dim)
        for key, val in terms.items():
            if key not in self.index:
                if abs(val) > 1e-12:
                    raise ValueError(f"term {key} lies outside ambient space")
                continue
            v[self.index[key]] += val
        return v

    def terms(self, v, tol=1e-12):
        return {self.keys[i]: float(x) for i,x in enumerate(v) if abs(x)>tol}


def exterior_derivative_terms(terms, n):
    out = {}
    for (alpha, I), c in terms.items():
        I = tuple(I)
        for j in range(n):
            if alpha[j] == 0 or j in I:
                continue
            beta = list(alpha)
            beta[j] -= 1
            seq = (j,) + I
            J = tuple(sorted(seq))
            s = _wedge_sign(seq)
            key = (tuple(beta), J)
            out[key] = out.get(key, 0.0) + c*alpha[j]*s
    return out


def koszul_terms(alpha, I, n):
    """kappa(x^alpha dx_I), where I has degree k+1."""
    out = {}
    I = tuple(I)
    for pos, j in enumerate(I):
        beta = list(alpha)
        beta[j] += 1
        J = I[:pos] + I[pos+1:]
        s = -1 if pos % 2 else 1
        key = (tuple(beta), J)
        out[key] = out.get(key, 0.0) + s
    return out


class TrimmedPolynomialFormSpace:
    """
    Reference space P_r^- Lambda^k on the n-simplex.

    Basis columns are stored in the ambient monomial k-form coefficient space.
    """

    def __init__(self, n: int, r: int, k: int, rank_tol=1e-11):
        self.n = int(n)
        self.r = int(r)
        self.k = int(k)
        if r < 1 or not (0 <= k <= n):
            raise ValueError("require r>=1 and 0<=k<=n")

        self.ambient = AmbientPolynomialForms(n, r, k)
        gens = []

        # P_{r-1} Lambda^k
        for alpha in monomial_indices(n, r-1):
            for I in wedge_indices(n, k):
                gens.append(self.ambient.vector({(alpha,I):1.0}))

        # kappa H_{r-1} Lambda^{k+1}
        if k < n:
            for alpha in monomial_indices(n, r-1, homogeneous=True):
                for I in wedge_indices(n, k+1):
                    gens.append(self.ambient.vector(koszul_terms(alpha,I,n)))

        S = np.column_stack(gens) if gens else np.zeros((self.ambient.dim,0))
        # Rank-revealing QR selects an actual basis from the natural generators.
        Q,R,piv = la.qr(S, mode="economic", pivoting=True)
        if R.size:
            diag = np.abs(np.diag(R))
            scale = diag[0] if len(diag) else 1.0
            rank = int(np.sum(diag > rank_tol*max(1.0,scale)))
        else:
            rank = 0
        target = trimmed_dimension(n,r,k)
        if rank != target:
            raise RuntimeError(
                f"trimmed basis rank {rank} != expected {target} for "
                f"(n,r,k)=({n},{r},{k})"
            )
        self.B = S[:, piv[:rank]].copy()
        self.nloc = rank
        self._pinv = np.linalg.pinv(self.B, rcond=rank_tol)

    @property
    def dim(self):
        return self.n

    @property
    def degree(self):
        return self.r

    @property
    def form_degree(self):
        return self.k

    def coordinates(self, ambient_vector, tol=5e-10):
        c = self._pinv @ ambient_vector
        err = np.linalg.norm(self.B@c-ambient_vector)
        den = max(1.0, np.linalg.norm(ambient_vector))
        if err > tol*den:
            raise ValueError(f"form is outside trimmed space: rel residual={err/den:.3e}")
        return c

    def basis_terms(self, j):
        return self.ambient.terms(self.B[:,j])

    def exterior_derivative_matrix(self):
        if self.k == self.n:
            return np.zeros((0,self.nloc))
        target = TrimmedPolynomialFormSpace(self.n,self.r,self.k+1)
        D = np.zeros((target.nloc,self.nloc))
        for j in range(self.nloc):
            t = exterior_derivative_terms(self.basis_terms(j), self.n)
            v = target.ambient.vector(t)
            D[:,j] = target.coordinates(v)
        return D

    def values(self, x):
        """Basis coefficient arrays at x.

        Returns shape (nloc, nwedge), where components refer to dx_I.
        """
        x = np.asarray(x,float)
        out = np.zeros((self.nloc,len(self.ambient.wedges)))
        widx = {I:j for j,I in enumerate(self.ambient.wedges)}
        for j in range(self.nloc):
            for (alpha,I), c in self.basis_terms(j).items():
                mon = 1.0
                for q,a in enumerate(alpha):
                    if a:
                        mon *= x[q]**a
                out[j,widx[I]] += c*mon
        return out

    def pullback_matrix(self, A, b, target=None):
        """
        Pullback by affine map x=A y+b.

        A has shape (self.n, m).  If target is omitted, m=self.n and the
        target is the same P_r^- Lambda^k space.  Rectangular A gives traces.
        """
        A = np.asarray(A,float)
        b = np.asarray(b,float)
        m = A.shape[1]
        if A.shape[0] != self.n or b.shape != (self.n,):
            raise ValueError("incompatible affine map")
        if target is None:
            target = TrimmedPolynomialFormSpace(m,self.r,self.k)
        if target.n != m or target.r != self.r or target.k != self.k:
            raise ValueError("wrong target space")

        T = np.zeros((target.nloc,self.nloc))
        target_wedges = target.ambient.wedges

        for col in range(self.nloc):
            out_terms = {}
            for (alpha,I), coeff in self.basis_terms(col).items():
                ppoly = polynomial_pullback(alpha,A,b)
                if self.k == 0:
                    minors = {():1.0}
                else:
                    minors = {}
                    rows = list(I)
                    for J in target_wedges:
                        sub = A[np.ix_(rows,list(J))]
                        det = float(np.linalg.det(sub))
                        if abs(det) > 1e-14:
                            minors[J] = det
                for beta,pv in ppoly.items():
                    for J,mv in minors.items():
                        key=(beta,J)
                        out_terms[key]=out_terms.get(key,0.0)+coeff*pv*mv
            v = target.ambient.vector(out_terms)
            T[:,col] = target.coordinates(v)
        return T

    def facet_trace_matrix(self, facet):
        """
        Trace from tetra/simplex facet 'facet' (opposite vertex facet) to the
        canonical reference (n-1)-simplex with remaining vertices in sorted order.
        """
        if self.n < 2:
            raise ValueError("no facet trace in dimension < 2")
        if not (0 <= facet <= self.n):
            raise ValueError("facet index out of range")
        if self.k > self.n-1:
            return np.zeros((0,self.nloc))

        V = np.vstack([np.zeros(self.n), np.eye(self.n)])
        q = [i for i in range(self.n+1) if i != facet]
        b = V[q[0]]
        A = np.column_stack([V[q[j]]-b for j in range(1,self.n)])
        target = TrimmedPolynomialFormSpace(self.n-1,self.r,self.k)
        return self.pullback_matrix(A,b,target=target)


def simplex_vertex_permutation_affine(perm):
    """Affine x=A y+b induced by a permutation of reference-simplex vertices.

    perm[j] is the image of vertex j.
    """
    perm = tuple(int(x) for x in perm)
    n = len(perm)-1
    if sorted(perm) != list(range(n+1)):
        raise ValueError("not a vertex permutation")
    V = np.vstack([np.zeros(n),np.eye(n)])
    b = V[perm[0]]
    A = np.column_stack([V[perm[j]]-b for j in range(1,n+1)])
    return A,b


def wedge_metric_matrix(Ginv, k):
    """Gram matrix of coordinate k-covectors dx_I."""
    Ginv = np.asarray(Ginv,float)
    n = Ginv.shape[0]
    W = wedge_indices(n,k)
    if k == 0:
        return np.ones((1,1))
    H = np.empty((len(W),len(W)))
    for a,I in enumerate(W):
        for b,J in enumerate(W):
            H[a,b] = np.linalg.det(Ginv[np.ix_(I,J)])
    return H


def local_hodge_mass(space: TrimmedPolynomialFormSpace, metric, q=None):
    """Metric mass/Hodge matrix for a trimmed polynomial k-form space."""
    if metric.dim != space.n:
        raise ValueError("metric dimension mismatch")
    q = int(q or max(space.r+3,5))
    lams, ws = simplex_duffy(space.n,q)
    M = np.zeros((space.nloc,space.nloc))
    for lam,w in zip(lams,ws):
        x = lam[1:]
        G = np.asarray(metric.metric(x),float)
        Gi = np.linalg.inv(G)
        mu = np.sqrt(np.linalg.det(G))
        H = wedge_metric_matrix(Gi,space.k)
        V = space.values(x)
        M += w*mu*(V@H@V.T)
    return M

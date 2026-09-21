"""Selberg trace formula check on the Bolza surface.

  sum_j h(r_j) = (A/4pi) int r h(r) tanh(pi r) dr
                 + sum_{[gamma]} (ell_0 / (2 sinh(ell_gamma/2))) ghat(ell_gamma)

with lambda_j = 1/4 + r_j^2 and the geometric sum over HYPERBOLIC CONJUGACY CLASSES
(one per oriented closed geodesic), ell_0 the primitive length.  Test function
h(r) = exp(-t r^2), so ghat(u) = exp(-u^2/(4t)) / (2 sqrt(pi t)).
Note lambda_0 = 0 gives r_0 = i/2, i.e. r_0^2 = -1/4, which the formula handles.
"""
import numpy as np, collections
from scipy.integrate import quad
from hypgeom import inv_lorentz
from klein import surface, OPPOSITE


def group_ball(G, nmax, verbose=False):
    seen = [np.eye(3)]
    keys = {tuple(np.round(np.eye(3).ravel(), 5))}
    frontier = [np.eye(3)]
    for L in range(nmax):
        new = []
        for W in frontier:
            for A in G:
                X = A @ W
                k = tuple(np.round(X.ravel(), 5))
                if k not in keys:
                    keys.add(k); seen.append(X); new.append(X)
        frontier = new
    return seen


def classify(elts, conjugators, tol=1e-6):
    reps, assign = [], []
    for X in elts:
        f = None
        for r, R in enumerate(reps):
            for A in conjugators:
                if np.max(np.abs(A @ X @ inv_lorentz(A) - R)) < tol * max(np.max(np.abs(R)), 1.0):
                    f = r; break
            if f is not None: break
        if f is None:
            reps.append(X); f = len(reps) - 1
        assign.append(f)
    return reps, assign


def length_classes(E, conjugators, lmax=8.0, tol=1e-7):
    """Group hyperbolic elements by translation length, count conjugacy classes."""
    tr = np.array([float(np.trace(X)) for X in E])
    ok = tr > 3 + 1e-9
    ell = np.full(len(E), np.nan)
    ell[ok] = np.arccosh(np.clip((tr[ok] - 1) / 2, 1, None))
    order = np.argsort(np.where(np.isnan(ell), 1e9, ell))
    groups = []
    for i in order:
        if np.isnan(ell[i]) or ell[i] > lmax: continue
        if groups and abs(ell[i] - groups[-1][0]) < tol:
            groups[-1][1].append(i)
        else:
            groups.append([ell[i], [i]])
    out = []
    for L, idx in groups:
        reps, asg = classify([E[i] for i in idx], conjugators)
        out.append((L, len(reps), len(idx)))
    return out


def ghat(u, t):
    return np.exp(-u * u / (4 * t)) / (2 * np.sqrt(np.pi * t))


def identity_term(area, t):
    f = lambda r: r * np.exp(-t * r * r) * np.tanh(np.pi * r)
    val, _ = quad(f, 0, np.inf, limit=400)
    return area / (4 * np.pi) * 2 * val


def geometric_term(classes, t, kmax=6):
    s = 0.0
    for L, ncls, _ in classes:
        for k in range(1, kmax + 1):
            s += ncls * L / (2 * np.sinh(k * L / 2)) * ghat(k * L, t)
    return s


def spectral_term(lams, t):
    return float(np.sum(np.exp(-t * (np.asarray(lams) - 0.25))))


def length_classes_fast(E, conj, lmax=8.0, tol=1e-7, dec=5):
    """Canonical-form classification: the class key is the lexicographically smallest
    rounded matrix in the orbit {A X A^{-1}} over a ball of conjugators."""
    Ea = np.array(E)
    Ca = np.array(conj)
    Ci = np.array([inv_lorentz(A) for A in conj])
    tr = np.trace(Ea, axis1=1, axis2=2)
    ok = tr > 3 + 1e-9
    ell = np.full(len(E), np.nan)
    ell[ok] = np.arccosh(np.clip((tr[ok] - 1) / 2, 1, None))
    keep = np.where(ok & (ell <= lmax))[0]
    seen = {}          # orbit key -> class id
    out = {}
    nxt = 0
    for i in keep:
        X = Ea[i]
        k = tuple(np.round(X.ravel(), 4))
        L = round(float(ell[i]), 6)
        if k in seen:
            cid = seen[k]
        else:
            cid = nxt; nxt += 1
            orb = np.einsum("nab,bc,ncd->nad", Ca, X, Ci)
            for m in orb:
                seen.setdefault(tuple(np.round(m.ravel(), 4)), cid)
        out.setdefault(L, set()).add(cid)
    return sorted((L, len(v)) for L, v in out.items())

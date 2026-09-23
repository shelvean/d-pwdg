"""Geometric-decomposition (Bernstein-Whitney) finite element complex on a glued
tetrahedral mesh, for the spaces P_r^- Lambda^k, k = 0, 1, 2, 3.

Basis (Arnold, Falk and Winther, geometric decomposition).  On a tetrahedron with
barycentric coordinates lambda_0..lambda_3, the functions

    c(alpha) lambda^alpha phi_sigma,   |alpha| = r - 1,
    phi_sigma = k! sum_i (-1)^i lambda_{sigma_i} dlambda_{sigma_0} ^ .. ^ dlambda_{sigma_k}
                (hat on i),

with sigma an increasing (k+1)-tuple and alpha_s = 0 for every s < min(sigma), form a
basis of P_r^- Lambda^k.  Each function belongs to the subsimplex f = supp(alpha) u sigma,
and the rule "alpha_s = 0 for s < min sigma" only involves vertices of f.  So each
subsimplex may use its own vertex order.  Here every subsimplex class of the glued mesh
carries one canonical vertex order, and every cell builds the functions of that class in
that order.  A shared function is then the same geometric form seen from both sides, and
continuity is the identification of coefficients, with no moment degrees of freedom and
no inverse.  c(alpha) = (r-1)!/alpha! is the Bernstein normalization; it is invariant
under vertex permutation, so it is consistent across cells.

Differential.  With |alpha| = r - 1,

    d(lambda^alpha phi_sigma) = 1/(k+1) sum_{m not in sigma} [ (r+k) lambda^alpha
                                - alpha_m sum_l lambda^{alpha - e_m + e_l} ] phi_{m sigma},

using  dlambda_sigma = 1/(k+1)! sum_{m not in sigma} phi_{m sigma}  and
dlambda_j ^ phi_sigma = k! lambda_j dlambda_sigma - phi_{j sigma}/(k+1).  Terms that
violate the ordering rule are rewritten with the Koszul relation
    sum_t (-1)^t lambda_{rho_t} phi_{rho \\ rho_t} = 0,
which needs a single step.  All coefficients are rational with denominator k+1, so
d o d = 0 holds to rounding at every degree.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from math import factorial
from functools import lru_cache
from riemannfem.forms import monomial_indices, wedge_indices
from aad import (duffy_points, radial_metric_batch, wedge_gram_batch, bernstein_moments,
                 product_maps)


# ------------------------------------------------------------------ entity classes
def entity_classes(ncells, pairings):
    """For each dimension d, map (cell, sorted local vertex tuple) -> (class id, slots),
    where slots[t] is the cell-local vertex occupying canonical position t."""
    out = []
    for d in range(4):
        nodes = [(c, S) for c in range(ncells) for S in itertools.combinations(range(4), d + 1)]
        adj = {n: [] for n in nodes}
        if d <= 2:
            for p in pairings:
                ca, fa, cb, fb, perm = (p.cell_plus, p.facet_plus, p.cell_minus,
                                        p.facet_minus, p.vertex_permutation)
                la = [j for j in range(4) if j != fa]
                lb = [j for j in range(4) if j != fb]
                m = {la[i]: lb[perm[i]] for i in range(3)}
                minv = {v: u for u, v in m.items()}
                for S in itertools.combinations(la, d + 1):
                    T = tuple(sorted(m[v] for v in S))
                    adj[(ca, S)].append(((cb, T), m))
                    adj[(cb, T)].append(((ca, S), minv))
        info, ncls = {}, 0
        for root in nodes:
            if root in info:
                continue
            info[root] = (ncls, tuple(root[1]))
            stack = [root]
            while stack:
                node = stack.pop()
                slots = info[node][1]
                for nb, m in adj[node]:
                    s2 = tuple(m[v] for v in slots)
                    if nb in info:
                        if info[nb][1] != s2:
                            raise RuntimeError(f'entity {nb} identified with itself nontrivially')
                        continue
                    info[nb] = (ncls, s2)
                    stack.append(nb)
            ncls += 1
        out.append((info, ncls))
    return out


# ------------------------------------------------------------------ local basis
@lru_cache(maxsize=None)
def slot_functions(dim, r, k):
    """(alpha over slots, sigma over slots) for the functions of a subsimplex of
    dimension dim whose support is the whole subsimplex."""
    fns = []
    if dim < k:
        return tuple(fns)
    slots = range(dim + 1)
    for sigma in itertools.combinations(slots, k + 1):
        for alpha in monomial_indices(dim + 1, r - 1, homogeneous=True):
            if set(s for s in slots if alpha[s] > 0) | set(sigma) != set(slots):
                continue
            if any(alpha[s] > 0 for s in range(min(sigma))):
                continue
            fns.append((tuple(alpha), tuple(sigma)))
    return tuple(fns)


def cnorm(alpha):
    n = sum(alpha)
    return factorial(n) / np.prod([factorial(a) for a in alpha])


def local_basis(cell, classes, r, k):
    """list of (global key, alpha over local vertices, sigma as local vertices in
    canonical order, alpha over slots, slots tuple)."""
    out = []
    for d in range(k, 4):
        info, _ = classes[d]
        for S in itertools.combinations(range(4), d + 1):
            cid, slots = info[(cell, S)]
            for aslot, sslot in slot_functions(d, r, k):
                alpha = [0, 0, 0, 0]
                for t, a in enumerate(aslot):
                    alpha[slots[t]] += a
                sigma = tuple(slots[t] for t in sslot)
                out.append(((d, cid, aslot, sslot), tuple(alpha), sigma, S))
    return out


# ------------------------------------------------------------------ Bernstein coefficients
DLAM = np.array([[-1., -1., -1.], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])


def dlam_wedge(verts, k):
    """components of dlambda_{v1} ^ ... ^ dlambda_{vk} on wedge_indices(3, k)."""
    W = wedge_indices(3, k)
    if k == 0:
        return np.ones(1)
    R = DLAM[list(verts)]                       # k x 3
    return np.array([np.linalg.det(R[:, list(I)]) for I in W])


def coefficients(basis, r, k):
    """A[gamma, component, i]: Bernstein coefficients (degree r) of each basis form."""
    gam = monomial_indices(4, r, homogeneous=True)
    gidx = {g: i for i, g in enumerate(gam)}
    nw = len(wedge_indices(3, k))
    A = np.zeros((len(gam), nw, len(basis)))
    for j, (_, alpha, sigma, _S) in enumerate(basis):
        c = cnorm(alpha) * factorial(k)
        for i, v in enumerate(sigma):
            g = list(alpha); g[v] += 1; g = tuple(g)
            lam_to_B = np.prod([factorial(x) for x in g]) / factorial(r)
            rest = sigma[:i] + sigma[i + 1:]
            A[gidx[g], :, j] += c * (-1) ** i * lam_to_B * dlam_wedge(rest, k)
    return A


def mass(A, V, Qm, r, k, q):
    X, W = duffy_points(q)
    pts = X.reshape(-1, 3)
    G = radial_metric_batch(V, Qm, pts)
    Gi = np.linalg.inv(G)
    if k == 0:
        H = np.ones((len(pts), 1, 1))
    else:
        H = wedge_gram_batch(Gi, k)
    nc = H.shape[-1]
    Fw = (W.reshape(-1) * np.sqrt(np.linalg.det(G)))[:, None, None] * H
    F = Fw.transpose(1, 2, 0).reshape(nc, nc, q, q, q)
    mom = bernstein_moments(F, 2 * r, q)
    _, I1, I2, I3, K = product_maps(r)
    Mb = K[None, None] * mom[:, :, I1, I2, I3]
    M = np.einsum('aci,cdab,bdj->ij', A, Mb, A, optimize=True)
    return 0.5 * (M + M.T)


# ------------------------------------------------------------------ differential
def perm_sign(seq):
    seq = list(seq); s = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                s = -s
    return s


def to_key(cell, classes, beta, tau, coef, out):
    """add coef * lambda^beta phi_tau (tau: local vertices, any order) to out,
    rewritten in the canonical basis of its subsimplex."""
    f = tuple(sorted(set(v for v in range(4) if beta[v] > 0) | set(tau)))
    d = len(f) - 1
    cid, slots = classes[d][0][(cell, f)]
    pos = {v: t for t, v in enumerate(slots)}
    ts = [pos[v] for v in tau]
    sgn = perm_sign(ts)
    ts = tuple(sorted(ts))
    b = [0] * (d + 1)
    for v in f:
        b[pos[v]] = beta[v]
    terms = [(tuple(b), ts, sgn * coef)]
    if any(b[s] > 0 for s in range(ts[0])):
        # one Koszul step with i the smallest offending slot
        i = next(s for s in range(ts[0]) if b[s] > 0)
        rho = (i,) + ts
        terms = []
        bm = list(b); bm[i] -= 1
        for t in range(1, len(rho)):
            bb = list(bm); bb[rho[t]] += 1
            tt = rho[:t] + rho[t + 1:]
            terms.append((tuple(bb), tt, (-1) ** (t + 1) * sgn * coef))
    for bb, tt, cc in terms:
        key = ((d, cid, bb, tt), f)
        out[key] = out.get(key, 0.0) + cc / cnorm(bb)


def local_d(cell, classes, basis, r, k):
    """dict: source key -> {target key: coefficient}, targets in P_r^- Lambda^{k+1}."""
    res = {}
    for key, alpha, sigma, S in basis:
        out = {}
        c = cnorm(alpha)
        for m in range(4):
            if m in sigma:
                continue
            tau = (m,) + sigma
            to_key(cell, classes, alpha, tau, c * (r + k) / (k + 1), out)
            if alpha[m] > 0:
                for l in range(4):
                    beta = list(alpha); beta[m] -= 1; beta[l] += 1
                    to_key(cell, classes, tuple(beta), tau, -c * alpha[m] / (k + 1), out)
        res[(key, S)] = {t: v for t, v in out.items() if abs(v) > 0}
    return res


def cell_d(cell, classes, basis, r, k):
    """d of the global basis functions restricted to one cell, in global keys.

    On a Delta-complex a global function may have several local copies in one cell
    (a cell can contain two vertices, edges or faces of the same class).  Its
    restriction is the sum of the copies, so contributions of source copies are
    added, and the coefficient of a target is read off any one of its copies, all of
    which must agree."""
    acc = {}
    for (skey, S), targets in local_d(cell, classes, basis, r, k).items():
        a = acc.setdefault(skey, {})
        for (tkey, f), v in targets.items():
            a[(tkey, f)] = a.get((tkey, f), 0.0) + v
    res, worst = {}, 0.0
    for skey, a in acc.items():
        per = {}
        for (tkey, f), v in a.items():
            per.setdefault(tkey, {})[f] = v
        out = {}
        for tkey, copies in per.items():
            vals = list(copies.values())
            worst = max(worst, max(vals) - min(vals)) if len(vals) > 1 else worst
            out[tkey] = vals[0]
        res[skey] = out
    return res, worst


# ------------------------------------------------------------------ global complex
class BWComplex:
    def __init__(self, cells, pairings, r, Qm, q=None, forms=(0, 1, 2, 3)):
        self.cells, self.r, self.Qm = cells, r, Qm
        self.q = q or r + 4
        self.classes = entity_classes(len(cells), pairings)
        self.basis = {k: [local_basis(c, self.classes, r, k) for c in range(len(cells))]
                      for k in forms}
        self.index = {}
        for k in forms:
            idx = {}
            for b in self.basis[k]:
                for e in b:
                    idx.setdefault(e[0], len(idx))
            self.index[k] = idx
        self.dims = {k: len(self.index[k]) for k in forms}

    def assemble_mass(self, k):
        rows, cols, vals = [], [], []
        idx = self.index[k]
        for c, b in enumerate(self.basis[k]):
            A = coefficients(b, self.r, k)
            M = mass(A, self.cells[c], self.Qm, self.r, k, self.q)
            g = np.array([idx[e[0]] for e in b])
            rows.append(np.repeat(g, len(g))); cols.append(np.tile(g, len(g)))
            vals.append(M.ravel())
        N = self.dims[k]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, N))

    def assemble_d(self, k, check=True):
        src, tgt = self.index[k], self.index[k + 1]
        entries, worst = {}, 0.0
        copy_worst = 0.0
        for c, b in enumerate(self.basis[k]):
            cd, cw = cell_d(c, self.classes, b, self.r, k)
            copy_worst = max(copy_worst, cw)
            for s, targets in cd.items():
                for t, v in targets.items():
                    key = (tgt[t], src[s])
                    if key in entries:
                        worst = max(worst, abs(entries[key] - v))
                    else:
                        entries[key] = v
        if check and max(worst, copy_worst) > 1e-12:
            raise RuntimeError(f'd not single valued: across cells {worst:.2e}, across copies {copy_worst:.2e}')
        self.d_consistency = (worst, copy_worst)
        ij = np.array(list(entries.keys())); vv = np.array(list(entries.values()))
        return sp.csr_matrix((vv, (ij[:, 0], ij[:, 1])), shape=(len(tgt), len(src)))

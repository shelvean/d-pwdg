import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import deque, defaultdict

from barynets import indices

# =============================================================================
# # prolongation (exact-elimination) solver for the obstacle problem
# #
# # The penalty scheme in pde.pd_obstacle enforces the smoothness and
# # boundary constraints by K + (1/eps) Q^T Q and iterates. This module
# # instead eliminates them exactly: it builds Z with Q Z = 0, so that
# # c = c_p + Z alpha satisfies the constraints by construction, and solves
# # the resulting BOX-constrained problem by a primal-dual active set method.
# #
# # Requires r = 0, where the prolongation is Boolean and the coefficient
# # constraint c >= chi transfers to a box constraint on alpha.
# #
# # example:
# # > from prolong import pd_obstacle_prolong
# # > c, lam, its, active = pd_obstacle_prolong(v,t,e,te,M,K,S,B,G,f,chi,d)
# =============================================================================


def ndof(d):
    return int((d + 1)*(d + 2)/2)


def _bfs_order(t):
    adj = defaultdict(list)
    cnt = defaultdict(list)
    for kk, tri in enumerate(t):
        for a in range(3):
            cnt[tuple(sorted((tri[a], tri[(a+1) % 3])))].append(kk)
    for key, ts in cnt.items():
        if len(ts) == 2:
            adj[ts[0]].append((ts[1], key)); adj[ts[1]].append((ts[0], key))
    seen = {0}; order = [(0, None)]; q = deque([0])
    while q:
        a = q.popleft()
        for (b, key) in adj[a]:
            if b not in seen:
                seen.add(b); order.append((b, key)); q.append(b)
    for kk in range(t.shape[0]):
        if kk not in seen: order.append((kk, None))
    return order


def column_order(v, t, d):
    """
    Locality-respecting ordering: triangles breadth-first from a root, and
    within each triangle, domain points nearest the entering edge first.  This
    mimics the one-dimensional sweep and is what makes Z local.
    """
    m = ndof(d)
    I, J, K = indices(d)
    idxs = np.vstack([I, J, K])
    pos = {}; p = 0
    for (kk, key) in _bfs_order(t):
        tri = list(t[kk])
        if key is None:
            keyf = np.zeros(m, dtype=int)
        else:
            io = ({0, 1, 2} - {tri.index(key[0]), tri.index(key[1])}).pop()
            keyf = idxs[io]
        for a in np.argsort(keyf, kind='stable'):
            pos[kk*m + int(a)] = p; p += 1
    return pos


def build_prolongation(S, ntot, pos, tol=1e-8):
    """
    Sparse Gaussian elimination on S to reduced row echelon form, pivoting as
    early as possible in `pos`.  Rank-revealing, so redundant smoothness
    conditions and the extra dimensions at singular vertices are handled with
    no case analysis.  Returns Z with S Z = 0 and columns spanning ker S.
    """
    Scsr = S.tocsr()
    pivots = {}; occ = {}
    for i in range(Scsr.shape[0]):
        a, b = Scsr.indptr[i], Scsr.indptr[i+1]
        row = {int(c): float(x) for c, x in zip(Scsr.indices[a:b], Scsr.data[a:b])}
        for pc in [c for c in list(row) if c in pivots]:
            f = row.get(pc, 0.0)
            if f == 0.0: continue
            for cc, x in pivots[pc].items():
                nv = row.get(cc, 0.0) - f*x
                if abs(nv) < tol: row.pop(cc, None)
                else: row[cc] = nv
        cand = [c for c in row if abs(row[c]) > tol]
        if not cand: continue
        pc = min(cand, key=lambda c: pos[c])
        f = row[pc]
        row = {c: x/f for c, x in row.items() if abs(x/f) > tol}
        row[pc] = 1.0
        for qc in list(occ.get(pc, ())):
            qrow = pivots[qc]
            if pc in qrow:
                g = qrow.pop(pc); occ[pc].discard(qc)
                for cc, x in row.items():
                    if cc == pc: continue
                    nv = qrow.get(cc, 0.0) - g*x
                    if abs(nv) < tol:
                        if cc in qrow: qrow.pop(cc); occ.setdefault(cc, set()).discard(qc)
                    else:
                        qrow[cc] = nv; occ.setdefault(cc, set()).add(qc)
        pivots[pc] = row
        for cc in row:
            if cc != pc: occ.setdefault(cc, set()).add(pc)
    D = sorted(pivots)
    F = sorted(set(range(ntot)) - set(D))
    fidx = {c: j for j, c in enumerate(F)}
    rows, cols, vals = [], [], []
    for j, c in enumerate(F):
        rows.append(c); cols.append(j); vals.append(1.0)
    for pc, row in pivots.items():
        for c, x in row.items():
            if c == pc: continue
            rows.append(pc); cols.append(fidx[c]); vals.append(-x)
    Z = sp.csr_matrix((vals, (rows, cols)), shape=(ntot, len(F)))
    return Z, np.array(D), np.array(F)

def particular_solution(Q, g, K=None):
    """
    Any c_p with Q c_p = g.  Uses the minimum-norm least-squares solution,
    which is cheap because Q is real and k-independent, and is computed once.
    """
    QQt = (Q @ Q.T).tocsc()
    QQt = QQt + 1e-12*abs(QQt.diagonal()).max()*sp.eye(QQt.shape[0], format='csc')
    y = spla.splu(QQt).solve(np.asarray(g).ravel())
    return Q.T @ y



def boolean_map(Z, tol=1e-12):
    """
    Verify that Z is a Boolean prolongation and return the index map.

    Returns
    -------
    jmap  : int array of length ntot.  jmap[i] is the reduced unknown that
            B-coefficient i equals, or -1 if row i of Z is zero (a coefficient
            pinned by the Dirichlet data).

    Raises ValueError if any row has more than one nonzero or a nonzero that
    is not 1, which is what happens for r >= 1.
    """
    Z = sp.csr_matrix(Z)
    nnz = np.diff(Z.indptr)
    bad = int((nnz > 1).sum())
    if bad:
        raise ValueError(
            f"Z is not Boolean: {bad} of {Z.shape[0]} rows have more than one "
            f"nonzero.  The C^0 collapse is invalid; this is expected for "
            f"r >= 1.  Use obstacle_pdas.build_reduced instead.")
    if Z.nnz and not np.allclose(Z.data, 1.0, atol=tol):
        worst = Z.data[np.argmax(np.abs(Z.data - 1.0))]
        raise ValueError(
            f"Z has a nonzero entry {worst!r} different from 1; the rows are "
            f"scalings rather than identifications and the max-collapse below "
            f"would be wrong.")
    jmap = np.full(Z.shape[0], -1, dtype=np.int64)
    rows = np.repeat(np.arange(Z.shape[0]), nnz)
    jmap[rows] = Z.indices
    return jmap


def collapse_bounds(jmap, dlow, n, atol=1e-12):
    """
    Collapse the ntot componentwise constraints onto n bounds.

        {alpha : Z alpha >= dlow}  ==  {alpha : alpha >= psit}

    exactly, with psit_j the largest of the bounds referring to alpha_j.

    The rows with jmap == -1 refer to no unknown at all.  For those the
    constraint is 0 >= dlow_i, i.e. c_p,i >= chi_i, a condition on the boundary
    data.  It is checked here and reported, not iterated on.
    """
    dlow = np.asarray(dlow).ravel()
    psit = np.full(n, -np.inf)
    live = jmap >= 0
    np.maximum.at(psit, jmap[live], dlow[live])

    fixed = ~live
    infeas = fixed & (dlow > atol)
    if infeas.any():
        worst = float(dlow[infeas].max())
        # A pinned COEFFICIENT below the obstacle does not by itself mean the
        # data is infeasible. Bernstein coefficients bound the function they
        # represent only from outside, so a trace g_h that satisfies g_h >= chi
        # pointwise can still have coefficients below chi on a coarse mesh at
        # low degree; this is the same strictness of the coefficient constraint
        # that Theorem (feasibility) trades accuracy for, appearing here in the
        # boundary data. Degree elevation drives the coefficients to the
        # function values and removes it, as does refining the boundary.
        #
        # Since these coefficients are fixed by the equality constraint B c = g
        # and are never iterated on, the box constraint is vacuous for them.
        # Warn rather than refuse, and report what would fix it.
        import warnings as _w
        _w.warn(
            f"{int(infeas.sum())} Dirichlet-pinned coefficients lie below the "
            f"obstacle by up to {worst:.3e}. These are fixed by B c = g and are "
            f"not iterated on, so the solve proceeds. If g >= chi holds "
            f"pointwise this is the Bernstein coefficients undershooting the "
            f"trace, and raising the degree or refining the boundary removes "
            f"it; if g < chi somewhere on the boundary the continuous problem "
            f"really is infeasible.", RuntimeWarning)
    if not np.isfinite(psit).all():
        # cannot happen for a prolongation built from build_prolongation, since
        # each free coefficient supplies its own row, but do not assume it
        psit[~np.isfinite(psit)] = -np.inf
    return psit, fixed


# ----------------------------------------------------------------------
# 2. the solver
# ----------------------------------------------------------------------
class BoxObstacle:
    """
    Reduced obstacle problem  min 1/2 a^T H a - a^T b  s.t.  a >= psit,
    solved by primal-dual active sets with the inactive-set primal step.

    Public interface matches obstacle_pdas.ReducedObstacle:
        .solve(alpha=None, active=None, cmax=100, tol=0.0, sigma=1.0)
        .n, .N, .nsolve
    with two differences, both documented at the call sites:
        * `active` may be given in coefficient space (length-ntot indices or a
          boolean mask) or in reduced space; both are accepted.
        * `hist` records the REDUCED active-set size |A| <= n, not the
          coefficient-space count, since that is now the number of genuinely
          distinct active constraints.
    """

    def __init__(self, H, b, Z, dlow, jmap=None, verbose=False):
        self.H = sp.csr_matrix(H)
        self.b = np.asarray(b).ravel()
        self.Z = sp.csr_matrix(Z)
        self.dlow = np.asarray(dlow).ravel()
        self.n = self.H.shape[0]
        self.N = self.Z.shape[0]
        self.verbose = verbose

        self.jmap = boolean_map(self.Z) if jmap is None else np.asarray(jmap)
        self.psit, self.fixed = collapse_bounds(self.jmap, self.dlow, self.n)
        self.constrained = np.isfinite(self.psit)

        # counters, named to line up with the dual solver's for the tables
        self.nsolve = 0          # sparse factorisations + solves
        self.nfact = 0           # factorisations only
        self.cg_its = 0          # unused, kept for interface compatibility

    # ---- one PDAS step -------------------------------------------------
    def _step(self, Amask):
        """
        Given the active mask, return (alpha, lam_reduced).

            alpha_A = psit_A
            H_II alpha_I = b_I - H_IA psit_A
            lam = H alpha - b   (supported on A)
        """
        alpha = np.zeros(self.n)
        alpha[Amask] = self.psit[Amask]
        I = np.flatnonzero(~Amask)
        if I.size:
            HII = self.H[I][:, I].tocsc()
            rhs = self.b[I] - (self.H[I] @ alpha)
            alpha[I] = spla.splu(HII).solve(rhs)
            self.nfact += 1
            self.nsolve += 1
        lam = np.zeros(self.n)
        r = self.H @ alpha - self.b
        lam[Amask] = r[Amask]
        return alpha, lam

    # ---- multiplier in coefficient space -------------------------------
    def _lam_full(self, alpha, Amask):
        """
        Push the reduced multiplier back to the ntot B-coefficients so the
        return value has the same shape and meaning as the dual solver's.

        With rho = H alpha - b in reduced space and lambda_j = rho_j on A, the
        split across the coefficients sharing a domain point is taken
        proportionally to their own residual contribution; the aggregate over
        each domain point reproduces lambda_j exactly.  Coefficients whose
        domain point is inactive, and Dirichlet-pinned coefficients, get zero.
        """
        lam_full = np.zeros(self.N)
        live = self.jmap >= 0
        act_rows = live & Amask[np.where(live, self.jmap, 0)]
        if not act_rows.any():
            return lam_full
        rho = self.H @ alpha - self.b
        # distribute rho_j over the rows mapping to j, weighted by row count
        cnt = np.bincount(self.jmap[live], minlength=self.n).astype(float)
        cnt[cnt == 0.0] = 1.0
        lam_full[act_rows] = rho[self.jmap[act_rows]] / cnt[self.jmap[act_rows]]
        return lam_full

    # ---- active set bookkeeping ---------------------------------------
    def _as_mask(self, active):
        """Accept coefficient-space or reduced-space active sets."""
        m = np.zeros(self.n, dtype=bool)
        if active is None:
            return m
        a = np.asarray(active)
        if a.dtype == bool:
            if a.size == self.N:
                live = (self.jmap >= 0) & a
                m[self.jmap[live]] = True
                return m & self.constrained
            if a.size == self.n:
                return a & self.constrained
            raise ValueError(f"boolean active mask of length {a.size}, "
                             f"expected {self.n} or {self.N}")
        # integer indices are read in COEFFICIENT space, which is what the dual
        # solver's `active` argument meant, so warm starts carry over unchanged
        a = a.astype(np.int64).ravel()
        if a.size == 0:
            return m
        if a.max() >= self.N:
            raise ValueError(f"active index {int(a.max())} out of range "
                             f"for {self.N} coefficients")
        j = self.jmap[a]
        m[j[j >= 0]] = True
        return m & self.constrained

    # ---- driver ---------------------------------------------------------
    def solve(self, alpha=None, active=None, cmax=100, tol=0.0, sigma=1.0,
              mode=None):
        """
        PDAS.  Returns (alpha, lam_full, iterations, history of reduced |A|).

        `mode` is accepted and ignored; there is one route now.

        The active set is updated by the standard complementarity test
            A = { j : lambda_j + sigma (psit_j - alpha_j) > tol }
        restricted to the constrained indices.
        """
        A = self._as_mask(active)
        hist, seen = [], set()
        alpha = np.zeros(self.n)
        lam = np.zeros(self.n)

        for it in range(1, cmax + 1):
            alpha, lam = self._step(A)
            hist.append(int(A.sum()))
            Anew = (lam + sigma*(self.psit - alpha) > tol) & self.constrained
            if self.verbose:
                print(f"      PDAS {it:3d}: |A| = {int(A.sum()):5d}  changes = "
                      f"{int((Anew ^ A).sum()):4d}  factorisations {self.nfact}")
            if np.array_equal(Anew, A):
                return alpha, self._lam_full(alpha, A), it, hist
            key = Anew.tobytes()
            if key in seen:
                # PDAS cycling; the primal iterate is still feasible and the
                # sets differ by a handful of indices at the free boundary
                if self.verbose:
                    print("      PDAS cycling detected, stopping")
                return alpha, self._lam_full(alpha, A), it, hist
            seen.add(key)
            A = Anew
        return alpha, self._lam_full(alpha, A), cmax, hist


# ----------------------------------------------------------------------
# 3. same entry point as obstacle_pdas.build_reduced
# ----------------------------------------------------------------------


def pd_obstacle_prolong(v, t, e, te, M, K, S, B, G, F, chi, d, r=0,
                        cmax=200, verbose=False):
# =============================================================================
#     # drop-in alternative to pde.pd_obstacle using exact elimination.
#     #
#     # returns (c, lam, it_count, active) with the same meaning as
#     # pd_obstacle: c and lam are column vectors of B-coefficients, active is
#     # the index set of active coefficients.
# =============================================================================
    if r != 0:
        raise ValueError("the prolongation solver requires r = 0")
    ntot = K.shape[0]
    Q = sp.vstack([sp.csr_matrix(S), sp.csr_matrix(B)], format='csr')
    g = np.concatenate([np.zeros(S.shape[0]), np.asarray(G).ravel()])

    order = column_order(v, t, d)
    Z, D, Ffac = build_prolongation(Q, ntot, order)
    c_p = particular_solution(Q, g)
    # one step of iterative refinement. The Boolean prolongation gives QZ = 0
    # exactly, so the only equality residual is the one carried by the
    # particular solution; refining it once drives Q c_p - G to zero in
    # floating point.
    c_p = c_p - particular_solution(Q, Q @ c_p - g)

    H = (Z.T @ K @ Z).tocsr()
    b = -(Z.T @ (K @ c_p - M @ np.asarray(F).ravel()))
    dlow = np.asarray(chi).ravel() - c_p

    prob = BoxObstacle(H, b, Z, dlow, verbose=verbose)
    alpha, lam_alg, its, hist = prob.solve(cmax=cmax)
    c = Z @ alpha + c_p

    # BoxObstacle returns the ALGEBRAIC multiplier on the full coefficient
    # vector. pde.pd_obstacle returns the B-coefficients of the multiplier
    # FUNCTION, so divide by the integrals of the basis functions to match.
    from barynets import bbint
    Dint = np.asarray(bbint(v, t, d)).ravel()
    lam = np.divide(np.asarray(lam_alg).ravel(), Dint,
                    out=np.zeros(ntot), where=Dint != 0)

    active = np.argwhere(lam > 0).ravel()
    return c.reshape(-1, 1), lam.reshape(-1, 1), int(its), active


def pd_obstacle_ssn(v, t, e, te, M, K, S, B, G, F, chi, d, r=0,
                    kappa=0.3, cmax=60, verbose=False):
# =============================================================================
#     # same interface as pd_obstacle_prolong, but the reduced box-constrained
#     # problem is solved by semismooth Newton on the penalized
#     # Fischer-Burmeister reformulation instead of by a primal-dual active set
#     # iteration. See ssn.py.
#     #
#     # ssn_qp is written for an UPPER bound c <= ub, while the obstacle
#     # constraint is a LOWER bound alpha >= dlow. Substituting beta = -alpha
#     # turns
#     #     min 1/2 alpha' H alpha - b' alpha   s.t.  alpha >= dlow
#     # into
#     #     min 1/2 beta' H beta + b' beta      s.t.  beta <= -dlow,
#     # and the multiplier is unchanged: H alpha - b = lambda in both.
# =============================================================================
    import numpy as np
    import scipy.sparse as sp
    from ssn import ssn_qp

    if r != 0:
        raise ValueError("the prolongation route requires r = 0")
    ntot = K.shape[0]
    Q = sp.vstack([sp.csr_matrix(S), sp.csr_matrix(B)], format='csr')
    g = np.concatenate([np.zeros(S.shape[0]), np.asarray(G).ravel()])

    order = column_order(v, t, d)
    Z, D, Ffac = build_prolongation(Q, ntot, order)
    c_p = particular_solution(Q, g)
    # one step of iterative refinement. The Boolean prolongation gives QZ = 0
    # exactly, so the only equality residual is the one carried by the
    # particular solution; refining it once drives Q c_p - G to zero in
    # floating point.
    c_p = c_p - particular_solution(Q, Q @ c_p - g)

    H = (Z.T @ K @ Z).tocsr()
    b = -(Z.T @ (K @ c_p - M @ np.asarray(F).ravel()))
    dlow = np.asarray(chi).ravel() - c_p
    jmap = boolean_map(Z)
    n = Z.shape[1]
    lb = np.full(n, -np.inf)
    for i in range(ntot):
        j = jmap[i]
        if j >= 0 and dlow[i] > lb[j]:
            lb[j] = dlow[i]

    Qe = sp.csr_matrix((0, n))
    beta, lam_red, info = ssn_qp(H, Qe, -b, -lb, kappa=kappa, cmax=cmax,
                                 verbose=verbose)
    alpha = -beta
    c = Z @ alpha + c_p

    from barynets import bbint
    lam_full = np.zeros(ntot)
    Zc = Z.tocsc()
    lr = np.asarray(lam_red).ravel()
    for j in range(n):
        s_, e_ = Zc.indptr[j], Zc.indptr[j + 1]
        if lr[j] != 0.0:
            lam_full[Zc.indices[s_:e_]] += lr[j]*Zc.data[s_:e_]
    Dint = np.asarray(bbint(v, t, d)).ravel()
    lam = np.divide(lam_full, Dint, out=np.zeros(ntot), where=Dint != 0)

    active = np.argwhere(lam > 0).ravel()
    return c.reshape(-1, 1), lam.reshape(-1, 1), int(info['its']), active

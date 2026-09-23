"""Metric mass matrices for trimmed polynomial k-forms by the Bernstein-Bezier moment
method of Ainsworth, Andriamaro and Davydov (SIAM J. Sci. Comput. 33 (2011)).

Nothing here evaluates the form basis.  Each basis form is written in Bernstein
coordinates,  phi_i = sum_{alpha, c} A[alpha, c, i] B^r_alpha dx^c,  using the
existing converter monomial_to_bernstein.  With the product rule

    B^r_alpha B^r_beta = [ C(alpha+beta, alpha) / C(2r, r) ] B^{2r}_{alpha+beta},

the mass matrix needs only the degree-2r Bernstein moments of the weights
f_cd = sqrt(det G) (wedge G^{-1})_{cd}:

    mu_gamma(f) = int_T B^{2r}_gamma f dx,   |gamma| = 2r.

In Duffy coordinates the Bernstein polynomial factorizes,
    B^m_gamma = b^m_{g1}(t1) b^{m-g1}_{g2}(t2) b^{m-g1-g2}_{g3}(t3),
so the moments are three one-dimensional contractions (sum factorization) over the
same tensor Gauss rule that simplex_duffy uses.  The mass matrix therefore agrees with
riemannfem.forms.local_hodge_mass up to rounding.
"""
import numpy as np
from functools import lru_cache
from math import comb
from riemannfem.forms import TrimmedPolynomialFormSpace, wedge_metric_matrix, monomial_indices
from riemannfem.bernstein_forms import monomial_to_bernstein


@lru_cache(maxsize=None)
def gauss01(q):
    z, w = np.polynomial.legendre.leggauss(q)
    return (z + 1.0) / 2.0, w / 2.0


@lru_cache(maxsize=None)
def univariate_table(m, q):
    """b[n, j, i] = C(n, j) t_i^j (1 - t_i)^(n - j) for 0 <= j <= n <= m."""
    t, _ = gauss01(q)
    b = np.zeros((m + 1, m + 1, q))
    for n in range(m + 1):
        for j in range(n + 1):
            b[n, j] = comb(n, j) * t ** j * (1 - t) ** (n - j)
    return b


def duffy_points(q):
    """points of the tensor rule, in the order of simplex_duffy (t1 outermost)."""
    t, w = gauss01(q)
    T1, T2, T3 = np.meshgrid(t, t, t, indexing='ij')
    W = w[:, None, None] * w[None, :, None] * w[None, None, :]
    jac = (1 - T1) ** 2 * (1 - T2)
    x1 = T1
    x2 = T2 * (1 - T1)
    x3 = T3 * (1 - T1) * (1 - T2)
    X = np.stack([x1, x2, x3], axis=-1)            # (q, q, q, 3)
    return X, W * jac


def bernstein_moments(F, m, q):
    """mu[..., g1, g2, g3] = sum_{i1 i2 i3} F[..., i1, i2, i3] B^m_gamma(t_i),
    with gamma0 = m - g1 - g2 - g3.  F already carries weights and Jacobian.
    Entries with g1 + g2 + g3 > m are zero."""
    b = univariate_table(m, q)
    lead = F.shape[:-3]
    # stage 1: contract t3 for every (n3, g3)
    S3 = np.einsum('ngk,...abk->...ngab', b, F, optimize=True)      # (..., n3, g3, i1, i2)
    mu = np.zeros(lead + (m + 1, m + 1, m + 1))
    for g1 in range(m + 1):
        n2 = m - g1
        g2 = np.arange(n2 + 1)
        n3 = n2 - g2
        # stage 2: contract t2 with degree n2, picking the matching n3 slab
        S3g = S3[..., n3, :, :, :]                                   # (..., g2, g3, i1, i2)
        S2 = np.einsum('gk,...gcak->...gca', b[n2, :n2 + 1, :], S3g, optimize=True)
        # stage 3: contract t1 with degree m
        mu[..., g1, :n2 + 1, :] = np.einsum('a,...gca->...gc', b[m, g1, :], S2, optimize=True)
    return mu


@lru_cache(maxsize=None)
def product_maps(r):
    """index arrays for gamma = alpha + beta and the product-rule constant."""
    alphas = monomial_indices(4, r, homogeneous=True)     # (a0, a1, a2, a3), a0 <-> lambda0
    N = len(alphas)
    I1 = np.zeros((N, N), int); I2 = np.zeros((N, N), int); I3 = np.zeros((N, N), int)
    K = np.zeros((N, N))
    den = comb(2 * r, r)
    for i, a in enumerate(alphas):
        for j, c in enumerate(alphas):
            g = [a[s] + c[s] for s in range(4)]
            I1[i, j], I2[i, j], I3[i, j] = g[1], g[2], g[3]
            K[i, j] = np.prod([comb(a[s] + c[s], a[s]) for s in range(4)]) / den
    return alphas, I1, I2, I3, K


@lru_cache(maxsize=None)
def bernstein_coefficients(r, k):
    """A[alpha, c, i]: Bernstein coefficients of the algebraic basis of the space."""
    V = TrimmedPolynomialFormSpace(3, r, k)
    H = monomial_to_bernstein(3, r, k) @ V.B
    nw = len(V.ambient.wedges)
    return H.reshape(-1, nw, V.nloc)


def local_hodge_mass_aad(space, metric, q=None):
    r, k = space.r, space.k
    q = int(q or max(r + 3, 5))
    X, W = duffy_points(q)
    pts = X.reshape(-1, 3)
    G = np.array([np.asarray(metric.metric(x), float) for x in pts])
    Gi = np.linalg.inv(G)
    mu_ = np.sqrt(np.linalg.det(G))
    H = np.array([wedge_metric_matrix(g, k) for g in Gi])           # (nq, c, c)
    nc = H.shape[-1]
    Fw = (W.reshape(-1) * mu_)[:, None, None] * H                   # (nq, c, c)
    F = Fw.transpose(1, 2, 0).reshape(nc, nc, q, q, q)
    mom = bernstein_moments(F, 2 * r, q)                            # (c, c, m+1, m+1, m+1)
    alphas, I1, I2, I3, K = product_maps(r)
    Mb = K[None, None] * mom[:, :, I1, I2, I3]                      # (c, c, N, N)
    A = bernstein_coefficients(r, k)                                # (N, c, nloc)
    M = np.einsum('aci,cdab,bdj->ij', A, Mb, A, optimize=True)
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# fully vectorized weights for the Seifert-Weber cells: the hyperboloid radial metric
# G(x) = Dy^T (Q / r2 + Qy Qy^T / r2^2) Dy,  y = V b(x),  r2 = -y^T Q y,
# evaluated at all quadrature points at once, and the k-covector Gram matrices
# H_IJ = det(G^{-1}[I, J]) built for every point in one pass.
from riemannfem.forms import wedge_indices


def radial_metric_batch(V, Qm, pts):
    V = np.asarray(V, float)
    Dy = V[:, 1:] - V[:, [0]]
    B = np.c_[1 - pts.sum(1), pts]                    # (nq, 4)
    Y = B @ V.T                                       # (nq, 4)
    QY = Y @ Qm
    r2 = -np.einsum('qi,qi->q', Y, QY)
    P = QY @ Dy                                       # (nq, 3)
    G = (Dy.T @ Qm @ Dy)[None] / r2[:, None, None] + P[:, :, None] * P[:, None, :] / r2[:, None, None] ** 2
    return 0.5 * (G + G.transpose(0, 2, 1))


def wedge_gram_batch(Gi, k):
    W = wedge_indices(Gi.shape[-1], k)
    if k == 1:
        return Gi
    nq = len(Gi)
    H = np.empty((nq, len(W), len(W)))
    for a, I in enumerate(W):
        for b, J in enumerate(W):
            H[:, a, b] = np.linalg.det(Gi[:, I][:, :, J])
    return H


def local_hodge_mass_aad_cell(space, V, Qm, q=None):
    """same matrix as local_hodge_mass_aad, with the metric evaluated in batch."""
    r, k = space.r, space.k
    q = int(q or max(r + 3, 5))
    X, W = duffy_points(q)
    pts = X.reshape(-1, 3)
    G = radial_metric_batch(V, Qm, pts)
    Gi = np.linalg.inv(G)
    mu_ = np.sqrt(np.linalg.det(G))
    H = wedge_gram_batch(Gi, k)
    nc = H.shape[-1]
    Fw = (W.reshape(-1) * mu_)[:, None, None] * H
    F = Fw.transpose(1, 2, 0).reshape(nc, nc, q, q, q)
    mom = bernstein_moments(F, 2 * r, q)
    alphas, I1, I2, I3, K = product_maps(r)
    Mb = K[None, None] * mom[:, :, I1, I2, I3]
    A = bernstein_coefficients(r, k)
    M = np.einsum('aci,cdab,bdj->ij', A, Mb, A, optimize=True)
    return 0.5 * (M + M.T)

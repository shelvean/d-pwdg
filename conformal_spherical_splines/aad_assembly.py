"""
aad_assembly.py
===============

Optimal assembly of the spherical spline mass and stiffness matrices in the
style of Ainsworth, Andriamaro, and Davydov, "Bernstein-Bezier finite
elements of arbitrary order and optimal assembly procedures", SIAM J. Sci.
Comput. 33 (2011) 3087-3109.

The two structural identities transfer verbatim to spherical Bernstein
basis functions because they are pointwise algebraic in the barycentric
coordinates:

  (P1)  B^d_alpha B^e_beta
            = [C(d,alpha) C(e,beta) / C(d+e, alpha+beta)] B^{d+e}_{alpha+beta},
        with C(d,alpha) = d!/alpha! the multinomial coefficient, so every
        product of two basis functions is a SINGLE scaled basis function of
        the sum degree;
  (P2)  directional derivatives of the degree-d homogeneous extension are
        constant-coefficient shifts to degree d-1:
            D_c H_a = d sum_m Vinv[m, c] B^{d-1}_{a - e_m}.

Consequences for assembly over one spherical triangle with quadrature
(nodes sb_q, weights W_q, nq points, m_d basis functions):

  mass      M_ab = sum_q W_q w_q B_a B_b
                 = pc(a, b) * mu_{a+b},        mu = B^{2d} @ (W * w),
  stiffness K_ab = sum_{m,n} pc1(a-e_m, b-e_n) * mu^{mn}_{a+b-e_m-e_n},
            mu^{mn} = B^{2d-2} @ (W * c_mn),
            c_mn(v) = d^2 [ Vinv_m . Vinv_n - (Vinv_m . v)(Vinv_n . v) ],

where the six independent c_mn encode the tangential projector. The
per-triangle cost drops from O(nq m_d^2) inner products to O(nq m_{2d})
moment work plus O(m_d^2) gathers with precomputed coefficients, and the
result is EXACTLY the old quadrature assembly (same nodes, same weights),
reorganized; agreement is to roundoff, which the test script checks.

The BL18 second-order energy form is deliberately left on the direct
quadrature path: its integrand mixes three basis levels (H, grad H,
Hess H) and a naive channel expansion needs ~169 weight functions, which
erases the gain; exploiting the analytic structure of the channel traces
is noted as future work.
"""
import numpy as np
import numpy.linalg as la
from math import comb, factorial
from scipy.sparse import block_diag, bsr_matrix

from barynets import indices, locate
from sphsplines import refquad, triquad_sphere, bern_eval


def _multinom(d, I, J, K):
    out = np.empty(I.size, dtype=float)
    for i in range(I.size):
        out[i] = factorial(d) // (factorial(int(I[i])) * factorial(int(J[i]))
                                  * factorial(int(K[i])))
    return out


def precompute_mass_tables(d):
    """PC0[a,b] and IDX0[a,b] with M_ab = PC0[a,b] * mu2d[IDX0[a,b]]."""
    I, J, K = indices(d)
    I2, J2, K2 = indices(2 * d)
    m = I.size
    cd = _multinom(d, I, J, K)
    c2d = _multinom(2 * d, I2, J2, K2)
    ia = I[:, None] + I[None, :]
    ja = J[:, None] + J[None, :]
    ka = K[:, None] + K[None, :]
    IDX0 = locate(ia.ravel(), ja.ravel(), ka.ravel(), I2, J2, K2
                  ).reshape(m, m)
    PC0 = (cd[:, None] * cd[None, :]) / c2d[IDX0]
    return PC0, IDX0


def precompute_stiff_tables(d):
    """CF[m][n], IDX[m][n] (invalid shifts masked to coefficient 0)."""
    I, J, K = indices(d)
    I1, J1, K1 = indices(d - 1)
    I2, J2, K2 = indices(2 * d - 2)
    m = I.size
    c1 = _multinom(d - 1, I1, J1, K1)
    c2 = _multinom(2 * d - 2, I2, J2, K2)
    e = np.eye(3, dtype=np.int64)
    # positions and coefficients of a - e_m in the degree d-1 lattice
    pos1, cf1 = [], []
    for mm in range(3):
        ii, jj, kk = I - e[mm, 0], J - e[mm, 1], K - e[mm, 2]
        ok = (ii >= 0) & (jj >= 0) & (kk >= 0)
        p = np.full(m, 0, dtype=np.int64)
        p[ok] = locate(ii[ok], jj[ok], kk[ok], I1, J1, K1)
        pos1.append(p)
        c = np.zeros(m)
        c[ok] = c1[p[ok]]
        cf1.append((c, ok))
    CF = [[None] * 3 for _ in range(3)]
    IDX = [[None] * 3 for _ in range(3)]
    for mm in range(3):
        cm, okm = cf1[mm]
        im, jm, km = I - e[mm, 0], J - e[mm, 1], K - e[mm, 2]
        for nn in range(3):
            cn, okn = cf1[nn]
            inn, jnn, knn = I - e[nn, 0], J - e[nn, 1], K - e[nn, 2]
            ia = im[:, None] + inn[None, :]
            ja = jm[:, None] + jnn[None, :]
            ka = km[:, None] + knn[None, :]
            ok = okm[:, None] & okn[None, :]
            idx = np.zeros((m, m), dtype=np.int64)
            idx[ok] = locate(ia[ok], ja[ok], ka[ok], I2, J2, K2)
            cf = np.zeros((m, m))
            cf[ok] = (cm[:, None] * cn[None, :])[ok] / c2[idx[ok]]
            CF[mm][nn] = cf
            IDX[mm][nn] = idx
    return CF, IDX


def assemble_sphere_aad(v, t, d, q=12, weight=None):
    """Weighted mass and (unweighted) tangential stiffness block-diagonal
    matrices on the discontinuous coefficients, batched AAD moment
    assembly. Because B^D(sb) = B^D(lam) / r^D with lam the fixed
    reference barycentrics, the degree-2d and degree-(2d-2) basis
    matrices are triangle-independent, so all per-triangle moments become
    two BLAS products against fixed reference matrices; per-triangle work
    is O(1) gathers per matrix entry. The radial powers move into the
    quadrature weights. Returns (M, K) matching assemble_sphere."""
    lam, wref = refquad(q)
    nq = wref.size
    ntri = t.shape[0]
    PC0, IDX0 = precompute_mass_tables(d)
    CF, IDX = precompute_stiff_tables(d)
    B2D = bern_eval(2 * d, lam)              # (m2d, nq), fixed
    B2Dm2 = bern_eval(2 * d - 2, lam)        # (m2d-2, nq), fixed

    Vm = np.stack([np.stack([v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]],
                            axis=1) for kk in range(ntri)])   # (ntri,3,3)
    U = np.einsum('tci,iq->tcq', Vm, lam)                     # flat points
    r = np.sqrt((U ** 2).sum(axis=1))                         # (ntri, nq)
    pts = U / r[:, None, :]
    nrm = np.cross(Vm[:, :, 1] - Vm[:, :, 0], Vm[:, :, 2] - Vm[:, :, 0])
    A2 = np.linalg.norm(nrm, axis=1)
    hn = np.abs(np.einsum('tc,tc->t', Vm[:, :, 0], nrm)) / A2
    W = wref[None, :] * (A2 * hn)[:, None] / r ** 3           # (ntri, nq)

    if weight is not None:
        wall = weight(pts.transpose(1, 0, 2).reshape(3, -1)
                      ).reshape(ntri, nq)
    else:
        wall = 1.0
    coefM = W * wall / r ** (2 * d)
    MUm = coefM @ B2D.T                                       # (ntri, m2d)
    Mblocks = PC0[None, :, :] * MUm[:, IDX0]

    Vinv = np.linalg.inv(Vm)                                  # (ntri,3,3)
    Gram = np.einsum('tmc,tnc->tmn', Vinv, Vinv)
    Vv = np.einsum('tmc,tcq->tmq', Vinv, pts)
    m = PC0.shape[0]
    Kblocks = np.zeros((ntri, m, m))
    Wk = W / r ** (2 * d - 2)
    for mm in range(3):
        for nn in range(3):
            coef = d * d * (Gram[:, mm, nn][:, None]
                            - Vv[:, mm] * Vv[:, nn]) * Wk
            MU = coef @ B2Dm2.T                               # (ntri, m2d-2)
            Kblocks += CF[mm][nn][None, :, :] * MU[:, IDX[mm][nn]]
    idx = np.arange(ntri + 1)
    Msp = bsr_matrix((Mblocks, idx[:-1], idx), shape=(ntri * m,) * 2).tocsr()
    Ksp = bsr_matrix((Kblocks, idx[:-1], idx), shape=(ntri * m,) * 2).tocsr()
    return Msp, Ksp


def sphere_pencil_aad(v, t, d, r, q=12, weight=None):
    """Drop-in replacement for sphsplines.sphere_pencil using the AAD
    assembler: returns (Z, Kz, Mz, S, M, K)."""
    from sphsplines import smoothness_sphere
    from prolong import column_order, build_prolongation
    m = (d + 1) * (d + 2) // 2
    ntot = m * t.shape[0]
    S = smoothness_sphere(v, t, d, r)
    Z, D, F = build_prolongation(S, ntot, column_order(v, t, d))
    M, K = assemble_sphere_aad(v, t, d, q=q, weight=weight)
    Kz = (Z.T @ K @ Z).tocsc()
    Mz = (Z.T @ M @ Z).tocsc()
    return Z, Kz, Mz, S, M, K


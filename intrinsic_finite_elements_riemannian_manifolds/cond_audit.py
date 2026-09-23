"""Conditioning audit on one Seifert-Weber cell, 1-forms.

Four local bases of the same space P_r^- Lambda^1, all with mass matrices built by the
sum-factorized Bernstein moments of aad.py on the same quadrature:

  alg   the algebraic basis V.B (monomial coefficients, rank-revealing QR);
  dual  the moment-dual basis actually used by the complex, E^{-1} applied to alg;
  bw    the Arnold-Falk-Winther geometric-decomposition basis  lambda^alpha phi_sigma,
        phi_sigma the Whitney form of the edge sigma = (i, j), |alpha| = r - 1,
        alpha_m = 0 for m < i;
  bw-s  the same basis with each function rescaled to unit norm on the regular
        tetrahedron, a scaling that depends only on (alpha, sigma) up to vertex
        permutation and so can be applied consistently across cells.
"""
import os, sys, numpy as np
os.environ['RIEMANNFEM_EQUILIBRATE'] = '1'
from math import factorial
import hypforms
from sw_forms import seifert_weber
from riemannfem.forms import TrimmedPolynomialFormSpace, monomial_indices
from riemannfem.entity_assembly import reference_entity_dofs
from aad import (duffy_points, radial_metric_batch, wedge_gram_batch, bernstein_moments,
                 product_maps, bernstein_coefficients)


def mass_from_A(A, V, Qm, r, q, k=1):
    X, W = duffy_points(q)
    pts = X.reshape(-1, 3)
    G = radial_metric_batch(V, Qm, pts)
    Gi = np.linalg.inv(G)
    H = wedge_gram_batch(Gi, k)
    nc = H.shape[-1]
    Fw = (W.reshape(-1) * np.sqrt(np.linalg.det(G)))[:, None, None] * H
    F = Fw.transpose(1, 2, 0).reshape(nc, nc, q, q, q)
    mom = bernstein_moments(F, 2 * r, q)
    _, I1, I2, I3, K = product_maps(r)
    Mb = K[None, None] * mom[:, :, I1, I2, I3]
    M = np.einsum('aci,cdab,bdj->ij', A, Mb, A, optimize=True)
    return 0.5 * (M + M.T)


def whitney_bernstein_A(r):
    """Bernstein coefficients (alpha, component, basis) of lambda^alpha phi_sigma."""
    alphas = monomial_indices(4, r, homogeneous=True)
    idx = {a: i for i, a in enumerate(alphas)}
    dlam = {0: np.array([-1., -1., -1.]), 1: np.array([1., 0, 0]),
            2: np.array([0, 1., 0]), 3: np.array([0, 0, 1.])}
    cols = []
    for i in range(4):
        for j in range(i + 1, 4):
            for a in monomial_indices(4, r - 1, homogeneous=True):
                if any(a[m] > 0 for m in range(i)):
                    continue
                col = np.zeros((len(alphas), 3))
                for (p, s, sign) in ((i, j, 1.0), (j, i, -1.0)):
                    g = list(a); g[p] += 1; g = tuple(g)
                    lam_to_B = np.prod([factorial(x) for x in g]) / factorial(r)
                    col[idx[g]] += sign * lam_to_B * dlam[s]
                cols.append(col)
    return np.stack(cols, axis=-1)                     # (N_r, 3, nloc)


def cond(M):
    w = np.linalg.eigvalsh(M)
    return w[-1] / w[0] if w[0] > 0 else np.inf


if __name__ == '__main__':
    cells, pairings, metrics = seifert_weber(0)
    V = cells[3]
    Qm = hypforms.Q
    # regular tetrahedron in Euclidean R^3 as a Minkowski cell is not needed: use the
    # Euclidean metric on the regular tetrahedron for the permutation-invariant scaling.
    print(f"{'r':>2} {'nloc':>5} {'cond E':>9} {'cond E eq':>9} {'alg':>9} {'dual':>9} {'bw':>9} {'bw-s':>9}")
    for r in range(3, 11):
        q = r + 4
        nloc = r * (r + 2) * (r + 3) // 2
        Ab = whitney_bernstein_A(r)
        assert Ab.shape[-1] == nloc, (Ab.shape, nloc)
        Mbw = mass_from_A(Ab, V, Qm, r, q)
        d = 1.0 / np.sqrt(np.diag(Mbw))
        cbw, cbws = cond(Mbw), cond(d[:, None] * Mbw * d[None, :])
        row = [f"{r:>2}", f"{nloc:>5}"]
        if r <= 8:
            sp = TrimmedPolynomialFormSpace(3, r, 1)
            Aa = bernstein_coefficients(r, 1)
            Malg = mass_from_A(Aa, V, Qm, r, q)
            E, Einv, _, ceq = reference_entity_dofs(r, 1)
            s = np.linalg.svd(E, compute_uv=False)
            Mdual = Einv.T @ Malg @ Einv
            row += [f"{s[0]/s[-1]:9.1e}", f"{ceq:9.1e}", f"{cond(Malg):9.1e}", f"{cond(Mdual):9.1e}"]
        else:
            row += [f"{'':>9}"] * 4
        row += [f"{cbw:9.1e}", f"{cbws:9.1e}"]
        print(" ".join(row), flush=True)

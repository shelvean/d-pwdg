import numpy as np, itertools
from riemannfem.forms import monomial_indices, wedge_indices
from bw_complex import coefficients, local_basis, cell_d, entity_classes


def ext_derivative(A, r, k):
    """exterior derivative of forms given by Bernstein coefficients A (degree r) in
    reference coordinates, returned as degree-r Bernstein coefficients of (k+1)-forms."""
    gam = monomial_indices(4, r, homogeneous=True); gi = {g: i for i, g in enumerate(gam)}
    Wk, Wk1 = wedge_indices(3, k), wedge_indices(3, k + 1); wi = {I: i for i, I in enumerate(Wk1)}
    dl = {0: -1.0}
    out = np.zeros((len(gam), len(Wk1), A.shape[-1]))
    for a, g in enumerate(gam):
        for c, I in enumerate(Wk):
            coef = A[a, c]
            if not np.any(coef):
                continue
            for m in range(3):                      # d/dx_m, x_m = lambda_{m+1}
                if m in I:
                    continue
                J = tuple(sorted((m,) + tuple(I)))
                sgn = (-1) ** sum(1 for x in I if x < m)
                for l, dld in ((0, -1.0), (m + 1, 1.0)):
                    if g[l] == 0:
                        continue
                    b = list(g); b[l] -= 1                 # B^{r-1}_b, factor r
                    for e in range(4):                  # degree elevation to r
                        bb = list(b); bb[e] += 1
                        w = (b[e] + 1) / r
                        out[gi[tuple(bb)], wi[J]] += sgn * dld * r * w * coef
    return out


if __name__ == '__main__':
    import hypforms
    from sw_forms import seifert_weber
    cells, pairings, _ = seifert_weber(0)
    cl = entity_classes(len(cells), pairings)
    for r in (2, 4, 6):
        for k in (0, 1, 2):
            worst = 0.0
            for c in (0, 17, 43):
                bk, bk1 = local_basis(c, cl, r, k), local_basis(c, cl, r, k + 1)
                Ak, Ak1 = coefficients(bk, r, k), coefficients(bk1, r, k + 1)
                # restrictions of global functions: sum of local copies
                def collapse(b, A):
                    keys = list(dict.fromkeys(e[0] for e in b))
                    kp = {kk: i for i, kk in enumerate(keys)}
                    G = np.zeros(A.shape[:2] + (len(keys),))
                    for j, e in enumerate(b):
                        G[:, :, kp[e[0]]] += A[:, :, j]
                    return keys, kp, G
                ks, ksp, Ak = collapse(bk, Ak)
                kt, ktp, Ak1 = collapse(bk1, Ak1)
                D = np.zeros((len(kt), len(ks)))
                cd, cw = cell_d(c, cl, bk, r, k)
                for key, targets in cd.items():
                    for t, v in targets.items():
                        D[ktp[t], ksp[key]] = v
                lhs = ext_derivative(Ak, r, k)
                rhs = np.einsum('acj,jn->acn', Ak1, D)
                worst = max(worst, np.abs(lhs - rhs).max() / max(1e-300, np.abs(lhs).max()))
            print(f'r={r} k={k}: local d formula vs Bernstein differentiation, rel err {worst:.1e}', flush=True)

"""
Errors against the exact solution, in the relative L2 norm and in the
broken wavenumber weighted H1 (energy) norm

    ||v||^2_{1,kappa,h} = sum_K ( ||grad v||^2_{L2(K)}
                                  + kappa_K^2 ||v||^2_{L2(K)} ).

Element integrals are computed with the composite rule of quadrature.py.
The default nsub = 6 subdivisions per direction resolves the oscillation on
the meshes used here; increase it if kappa h is large.
"""

import numpy as np

from quadrature import triangle_rule


def _points(s, e, lam):
    return lam @ s.v[s.t[e]]


def l2_error(s, nsub=6):
    """Relative L2 error, ||u - u_h|| / ||u||."""
    lam, w = triangle_rule(nsub)
    num = den = 0.0
    for e in range(s.nt):
        P = _points(s, e, lam)
        a = s.element_area(e)
        ue = s.exact(P[:, 0], P[:, 1])
        num += a * np.sum(w * np.abs(s.uh(e, P) - ue) ** 2)
        den += a * np.sum(w * np.abs(ue) ** 2)
    return float(np.sqrt(num / den))


def energy_error(s, nsub=6):
    """Absolute and reference energy norms, (||u - u_h||, ||u||)."""
    lam, w = triangle_rule(nsub)
    num = den = 0.0
    for e in range(s.nt):
        P = _points(s, e, lam)
        a, k = s.element_area(e), s.kap[e]
        ue = s.exact(P[:, 0], P[:, 1])
        ge = s.exact.grad(P[:, 0], P[:, 1])
        ev, eg = s.uh(e, P) - ue, s.grad_uh(e, P) - ge
        num += a * np.sum(w * (np.sum(np.abs(eg) ** 2, axis=-1)
                               + k ** 2 * np.abs(ev) ** 2))
        den += a * np.sum(w * (np.sum(np.abs(ge) ** 2, axis=-1)
                               + k ** 2 * np.abs(ue) ** 2))
    return float(np.sqrt(num)), float(np.sqrt(den))

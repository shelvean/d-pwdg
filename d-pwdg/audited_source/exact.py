"""
Exact solution for the transmission and total internal reflection problem.

This is Example 3 of

    S. Kapita, P. Monk and T. Warburton, Residual-based adaptivity and PWDG
    methods for the Helmholtz equation, SIAM J. Sci. Comput. 37 (2015),
    A1525-A1553, section 6.3,

also used as Example 3 of Congreve, Houston and Perugia, Adv. Comput. Math.
45 (2019) 361-393.  A plane wave in the lower half plane, of refractive
index n1, meets the interface y = 0 with the upper half plane, of index n2.
Writing d = (cos theta_i, sin theta_i),

    K1 = omega n1 d1,     K2 = sqrt(omega^2 n2^2 - K1^2),
    R  = -(K2 - omega n1 d2) / (K2 + omega n1 d2),     T = 1 + R,

the total field is

    u = T exp(i(K1 x + K2 y))                               for y > 0,
    u = exp(i omega n1 (d1 x + d2 y))
          + R exp(i omega n1 (d1 x - d2 y))                 for y < 0.

When omega^2 n2^2 < K1^2 the square root is taken with positive imaginary
part, the field above the interface decays exponentially, and the incident
wave is totally internally reflected.  That is the case theta_i = 29 degrees
with n1 = 2, n2 = 1, and it is the harder of the two for a plane wave basis,
since evanescent waves are not in the discrete space.  At theta_i = 69
degrees the transmitted wave propagates.
"""

import numpy as np


class Transmission:

    def __init__(self, omega, theta_deg, n1=2.0, n2=1.0):
        th = np.deg2rad(theta_deg)
        self.omega, self.n1, self.n2 = omega, n1, n2
        self.d1, self.d2 = np.cos(th), np.sin(th)
        self.K1 = omega * n1 * self.d1
        rad = (omega * n2) ** 2 - self.K1 ** 2
        self.K2 = np.sqrt(rad) if rad >= 0 else 1j * np.sqrt(-rad)
        self.evanescent = rad < 0
        denom = self.K2 + omega * n1 * self.d2
        self.R = -(self.K2 - omega * n1 * self.d2) / denom
        self.T = 1.0 + self.R

    def wavenumber(self, y):
        """Local wavenumber, piecewise constant across the interface."""
        return np.where(y < 0.0, self.omega * self.n1, self.omega * self.n2)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        k1 = self.omega * self.n1
        up = self.T * np.exp(1j * (self.K1 * x + self.K2 * y))
        dn = (np.exp(1j * k1 * (self.d1 * x + self.d2 * y))
              + self.R * np.exp(1j * k1 * (self.d1 * x - self.d2 * y)))
        return np.where(y > 0.0, up, dn)

    def grad(self, x, y):
        """Gradient of the exact solution, shape (..., 2)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        k1 = self.omega * self.n1
        up = self.T * np.exp(1j * (self.K1 * x + self.K2 * y))
        gup = np.stack([1j * self.K1 * up, 1j * self.K2 * up], axis=-1)
        a = np.exp(1j * k1 * (self.d1 * x + self.d2 * y))
        b = self.R * np.exp(1j * k1 * (self.d1 * x - self.d2 * y))
        gdn = np.stack([1j * k1 * self.d1 * (a + b),
                        1j * k1 * self.d2 * (a - b)], axis=-1)
        return np.where((y > 0.0)[..., None], gup, gdn)


class PlaneWave:
    """A single plane wave, used for the patch test."""

    def __init__(self, kappa, direction):
        self.kappa = kappa
        self.d = np.asarray(direction, dtype=float)

    def wavenumber(self, y):
        return np.full(np.shape(y), self.kappa, dtype=float)

    def __call__(self, x, y):
        return np.exp(1j * self.kappa * (self.d[0] * np.asarray(x)
                                         + self.d[1] * np.asarray(y)))

    def grad(self, x, y):
        u = self(x, y)
        return np.stack([1j * self.kappa * self.d[0] * u,
                         1j * self.kappa * self.d[1] * u], axis=-1)


class Hankel:
    """
    Outgoing cylindrical wave from a source outside the domain,

        u(x) = H_0^{(1)}(kappa |x - x0|),

    which solves the Helmholtz equation and is smooth on any domain not
    containing x0.  This is the smooth benchmark of Congreve, Houston and
    Perugia (2019), section 4.1, who take the unit square with
    x0 = (-1/4, 0).
    """

    def __init__(self, kappa, source=(-0.25, 0.0)):
        self.kappa = kappa
        self.x0 = np.asarray(source, dtype=float)

    def wavenumber(self, y):
        return np.full(np.shape(y), self.kappa, dtype=float)

    def _r(self, x, y):
        return np.hypot(np.asarray(x) - self.x0[0], np.asarray(y) - self.x0[1])

    def __call__(self, x, y):
        from scipy.special import hankel2
        return hankel2(0, self.kappa * self._r(x, y))

    def grad(self, x, y):
        from scipy.special import hankel2
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = self._r(x, y)
        # d/dz H_0 = -H_1, and grad r = (x - x0) / r
        radial = -self.kappa * hankel2(1, self.kappa * r) / r
        return np.stack([radial * (x - self.x0[0]),
                         radial * (y - self.x0[1])], axis=-1)


class BesselMode:
    """
    Cylindrical mode

        u(r, theta) = J_xi(kappa r) sin(xi theta),      theta in [0, 2 pi),

    which solves the Helmholtz equation wherever it is defined.  The angle
    is measured from the positive x axis and wrapped into [0, 2 pi), so the
    branch cut of a non-integer order lies along theta = 0.

    Two cases are used here, both from Kapita, Monk and Warburton (2015).
    With xi = 2 the mode is entire and smooth, section 6.1 there, and may be
    posed on any domain.  With xi = 2/3 it behaves like r^(2/3) at the
    origin, so u lies in H^(5/3 - eps) and no better, section 6.2 there;
    it must be posed on the L-shaped domain, whose boundary contains the
    branch cut theta = 0 and whose reentrant corner is the singular point.
    Convergence on quasi-uniform meshes is then limited by the singularity
    rather than by the discrete space, which is the point of the example.
    """

    def __init__(self, kappa, xi=2.0):
        self.kappa, self.xi = kappa, float(xi)

    def wavenumber(self, y):
        return np.full(np.shape(y), self.kappa, dtype=float)

    def __call__(self, x, y):
        from scipy.special import jv
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        th = np.arctan2(y, x) % (2.0 * np.pi)
        return jv(self.xi, self.kappa * r) * np.sin(self.xi * th)

    def grad(self, x, y):
        from scipy.special import jv, jvp
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        r = np.maximum(np.hypot(x, y), 1e-300)     # guard the origin
        th = np.arctan2(y, x) % (2.0 * np.pi)
        k, xi = self.kappa, self.xi
        ur = k * jvp(xi, k * r) * np.sin(xi * th)
        ut = xi * jv(xi, k * r) * np.cos(xi * th) / r
        c, sn = np.cos(th), np.sin(th)
        return np.stack([ur * c - ut * sn, ur * sn + ut * c], axis=-1)

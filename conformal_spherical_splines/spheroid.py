"""
spheroid.py
===========

The prolate/oblate spheroid x^2/a^2 + y^2/a^2 + z^2/c^2 = 1 as the first
curved genus-zero geometry: closed-form conformal parametrization from the
sphere, the induced conformal weight w = e^{2 lambda}, an independent
closed-form area check, and a dense spherical-harmonic Galerkin reference
solver for the transplanted pencil.

Conformal factor by Mercator matching. The spheroid metric in colatitude
theta and longitude phi is E dtheta^2 + G dphi^2 with
E = a^2 cos^2 + c^2 sin^2, G = a^2 sin^2. Both surfaces are conformal to a
cylinder through their Mercator variables; matching them equator to equator
gives the conformal map Theta -> theta(Theta) determined by

    ln tan(theta/2) + G0(theta) = ln tan(Theta/2),
    G0(theta) = int_{pi/2}^theta [sqrt(E(t)) - a] / (a sin t) dt,

with G0 smooth up to the poles (the integrand vanishes like sin t), and the
weight w(Theta) = a^2 sin^2 theta / sin^2 Theta, computed stably as

    sin theta / sin Theta = e^{-G0(theta)} (1 + t_Theta^2) / (1 + t_theta^2),
    t_x = tan(x/2),  t_theta = t_Theta e^{-G0(theta)}.

By symmetry w is even in z, so the Hersch centering (paper eq. (2)) holds
automatically. The closed-form area (prolate, c > a):
A = 2 pi a^2 + 2 pi a c arcsin(e)/e, e = sqrt(1 - a^2/c^2), is the
independent check on the weight: int_S w dA must reproduce it.
"""

import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh


class SpheroidWeight:
    """Callable conformal weight w(pts) for the spheroid (a, a, c)."""

    def __init__(self, a=1.0, c=1.5, ngrid=20001):
        self.a, self.c = float(a), float(c)
        # G0 on a fine theta grid by cumulative Simpson (integrand smooth)
        th = np.linspace(0.0, math.pi, ngrid)
        integ = np.zeros_like(th)
        s = np.sin(th)
        mask = s > 0
        E = self.a**2 * np.cos(th[mask])**2 + self.c**2 * np.sin(th[mask])**2
        integ[mask] = (np.sqrt(E) - self.a) / (self.a * s[mask])
        # endpoint limits: integrand -> 0 at both poles
        from scipy.integrate import cumulative_simpson
        G = cumulative_simpson(integ, x=th, initial=0.0)
        Gmid = np.interp(math.pi / 2, th, G)
        self._G0 = CubicSpline(th, G - Gmid)

    def theta_of_Theta(self, Theta):
        """Invert ln t_th + G0(th) = ln t_Th by damped fixed point:
        th = 2 atan( t_Th * exp(-G0(th)) ). Vectorized, monotone."""
        Theta = np.asarray(Theta, dtype=float)
        tTh = np.tan(0.5 * Theta)
        th = Theta.copy()
        for _ in range(60):
            th_new = 2.0 * np.arctan(tTh * np.exp(-self._G0(th)))
            if np.max(np.abs(th_new - th)) < 1e-15:
                th = th_new
                break
            th = th_new
        return th

    def w_of_Theta(self, Theta):
        Theta = np.asarray(Theta, dtype=float)
        th = self.theta_of_Theta(Theta)
        tTh = np.tan(0.5 * Theta)
        tth = tTh * np.exp(-self._G0(th))
        ratio = np.exp(-self._G0(th)) * (1.0 + tTh**2) / (1.0 + tth**2)
        return (self.a * ratio) ** 2

    def __call__(self, pts):
        """pts: 3 x nq array of unit vectors."""
        z = np.clip(pts[2], -1.0, 1.0)
        return self.w_of_Theta(np.arccos(z))

    def embed(self, pts):
        """Push sphere points forward to the spheroid surface (3 x nq)."""
        z = np.clip(pts[2], -1.0, 1.0)
        Theta = np.arccos(z)
        th = self.theta_of_Theta(Theta)
        sTh = np.sin(Theta)
        cphi = np.where(sTh > 1e-14, pts[0] / np.maximum(sTh, 1e-300), 1.0)
        sphi = np.where(sTh > 1e-14, pts[1] / np.maximum(sTh, 1e-300), 0.0)
        return np.vstack([self.a * np.sin(th) * cphi,
                          self.a * np.sin(th) * sphi,
                          self.c * np.cos(th)])

    def area_exact(self):
        a, c = self.a, self.c
        if abs(a - c) < 1e-14:
            return 4.0 * math.pi * a * a
        if c > a:                       # prolate
            e = math.sqrt(1.0 - a * a / (c * c))
            return 2 * math.pi * a * a + 2 * math.pi * a * c * math.asin(e) / e
        e = math.sqrt(1.0 - c * c / (a * a))   # oblate
        return 2 * math.pi * a * a + math.pi * c * c / e * math.log((1 + e) / (1 - e))


# ---------------------------------------------------------------------------
# dense spherical-harmonic reference for the pencil  a(u,v) = mu b_w(u,v)
# ---------------------------------------------------------------------------

def _legendre_norm(L, m, x):
    """Orthonormal associated Legendre \\bar P_l^m(x) for l = m..L at nodes x,
    normalized so that int_{-1}^{1} \\bar P_l^m \\bar P_l'^m dx = delta.
    Stable upward recurrence."""
    x = np.asarray(x, dtype=float)
    out = np.zeros((L - m + 1, x.size))
    # \bar P_m^m
    pmm = np.ones_like(x)
    if m > 0:
        s2 = (1.0 - x) * (1.0 + x)
        pmm = (-1.0) ** m * np.exp(
            0.5 * (math.lgamma(2 * m + 2) - math.lgamma(m + 1))
            - m * math.log(2.0)) / math.sqrt(math.factorial(2 * m)) \
            * s2 ** (m / 2.0)
        # normalize: \bar P_m^m = sqrt((2m+1)/2 * 1/(2m)!) (2m-1)!! (1-x^2)^{m/2}
        dfact = np.exp(math.lgamma(2 * m + 1) - m * math.log(2.0)
                       - math.lgamma(m + 1))          # (2m-1)!! = (2m)!/(2^m m!)
        pmm = ((-1.0) ** m) * math.sqrt((2 * m + 1) /
                                        (2.0 * math.factorial(2 * m))) \
            * dfact * s2 ** (m / 2.0)
    else:
        pmm = np.full_like(x, math.sqrt(0.5))
    out[0] = pmm
    if L > m:
        out[1] = x * math.sqrt(2 * m + 3.0) * pmm
    for l in range(m + 2, L + 1):
        A = math.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
        B = math.sqrt(((2.0 * l + 1.0) * ((l - 1.0)**2 - m * m))
                      / ((2.0 * l - 3.0) * (l * l - m * m)))
        out[l - m] = A * (x * out[l - m - 1] - B / math.sqrt(
            (2.0 * l - 1.0) / (2.0 * l - 3.0)) * out[l - m - 2] *
            math.sqrt((2.0 * l - 1.0) / (2.0 * l - 3.0)))
        # standard: \bar P_l = A x \bar P_{l-1} - (A/B') \bar P_{l-2}; use the
        # usual coefficients directly:
        out[l - m] = A * x * out[l - m - 1] - B * out[l - m - 2]
    return out


def sh_reference(weightTheta, L=60, nG=400, k=60):
    """
    Eigenvalues of the pencil a(u,v) = mu int w u v dA on the sphere in the
    real spherical-harmonic basis up to degree L, for a zonal weight
    w = w(Theta). Stiffness is diagonal l(l+1); the weighted mass couples
    only equal m (and, for even weights, equal parity, which we do not need
    to exploit). Returns the k smallest eigenvalues.
    """
    x, gw = np.polynomial.legendre.leggauss(nG)        # x = cos Theta
    W = weightTheta(np.arccos(x)) * gw                 # zonal weight * quad
    vals = []
    for m in range(0, L + 1):
        P = _legendre_norm(L, m, x)                    # (L-m+1) x nG
        Mm = (P * W[None, :]) @ P.T                    # 2pi cancels: both
        # forms carry the same azimuthal factor, so drop it consistently
        Km = np.diag([l * (l + 1.0) for l in range(m, L + 1)])
        mu = eigh(Km, Mm, eigvals_only=True)
        mult = 1 if m == 0 else 2
        for muv in mu:
            vals += [muv] * mult
    vals = np.sort(np.array(vals))
    return vals[:k]

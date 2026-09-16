"""
rbc.py: conformal weight for the Evans-Fung red blood cell discocyte,
by the generating-curve extension of the Mercator construction.

Profile (Evans-Fung 1972, Microvasc. Res. 4, 335-347), physical microns:
  r(eta) = R0 sin(eta),
  z(eta) = (R0/2) cos(eta) [C0 + C2 sin^2(eta) + C4 sin^4(eta)],
eta in [0, pi], R0 = 3.91, C0 = 0.207161, C2 = 2.002558, C4 = -1.122762.
The curve is regular everywhere (at the rim r' = 0 but z' != 0, at the
poles r' = R0), meets the axis perpendicularly, and the surface is NOT a
radial graph (the dimple), which the generating-curve route does not
require.

Mercator variable along the curve: t(eta) = ln tan(eta/2) + G(eta) with
  G'(eta) = sqrt(r'^2 + z'^2)/r - 1/sin(eta),
smooth up to both poles. Equatorial symmetry makes the Hersch centering
automatic with the equator normalization. Stable weight:
  w = (r/sin eta)^2 e^{-2 G~} ((1+T^2)/(1+tau^2))^2,
tau = tan(eta/2), T = tan(Theta/2) = tau e^{G~}, G~ = G - G(pi/2).
Physical units are kept (Area(M) in um^2, eigenvalues in um^-2).
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

R0 = 3.91
C0, C2, C4 = 0.207161, 2.002558, -1.122762


def profile(eta):
    s2 = np.sin(eta) ** 2
    r = R0 * np.sin(eta)
    z = 0.5 * R0 * np.cos(eta) * (C0 + C2 * s2 + C4 * s2 * s2)
    return r, z


def profile_d(eta):
    s, c = np.sin(eta), np.cos(eta)
    rp = R0 * c
    poly = C0 + C2 * s ** 2 + C4 * s ** 4
    polyp = (2 * C2 * s + 4 * C4 * s ** 3) * c
    zp = 0.5 * R0 * (-s * poly + c * polyp)
    return rp, zp


class CurveWeight:
    """Conformal weight for a surface of revolution with generating curve
    (r(eta), z(eta)), eta in [0, pi], r = a sin(eta) * (regular factor),
    equatorially symmetric."""

    def __init__(self, n=6000):
        eta = np.linspace(0.0, np.pi, n)
        gp = np.zeros(n)
        e = eta[1:-1]
        r, _ = profile(e)
        rp, zp = profile_d(e)
        gp[1:-1] = np.sqrt(rp ** 2 + zp ** 2) / r - 1.0 / np.sin(e)
        # endpoint limits: r ~ R0 eta, ds ~ R0 d eta (1 + O(eta^2)) so
        # G' -> 0 at both poles
        gp[0] = 0.0
        gp[-1] = 0.0
        G = CubicSpline(eta, gp).antiderivative()
        self.G = lambda p: G(p) - G(np.pi / 2)
        self.Gp = CubicSpline(eta, gp)

    def eta_of_Theta(self, Theta, iters=40):
        eta = np.array(Theta, dtype=float, copy=True)
        lt = np.log(np.tan(np.clip(Theta, 1e-14, np.pi - 1e-14) / 2))
        for _ in range(iters):
            F = np.log(np.tan(np.clip(eta, 1e-14, np.pi - 1e-14) / 2)) \
                + self.G(eta) - lt
            dF = 1.0 / np.maximum(np.sin(eta), 1e-14) + self.Gp(eta)
            eta = np.clip(eta - F / dF, 1e-12, np.pi - 1e-12)
        return eta

    def w_of_Theta(self, Theta):
        Theta = np.asarray(Theta, dtype=float)
        eta = self.eta_of_Theta(Theta)
        tau = np.tan(eta / 2)
        T = np.tan(Theta / 2)
        ratio = np.exp(-self.G(eta)) * (1 + T ** 2) / (1 + tau ** 2)
        r, _ = profile(eta)
        base = np.where(np.sin(eta) > 1e-12, r / np.sin(eta), R0)
        return base ** 2 * ratio ** 2

    def __call__(self, pts):
        z = np.clip(pts[2], -1.0, 1.0)
        return self.w_of_Theta(np.arccos(z))

    def embed(self, pts):
        z = np.clip(pts[2], -1.0, 1.0)
        Theta = np.arccos(z)
        eta = self.eta_of_Theta(Theta)
        rho = np.hypot(pts[0], pts[1])
        cphi = np.where(rho > 1e-14, pts[0] / np.maximum(rho, 1e-14), 1.0)
        sphi = np.where(rho > 1e-14, pts[1] / np.maximum(rho, 1e-14), 0.0)
        r, zz = profile(eta)
        return np.vstack([r * cphi, r * sphi, zz])

    def area_exact(self):
        f = lambda e: profile(e)[0] * np.hypot(*profile_d(e))
        val, _ = quad(f, 0, np.pi, limit=300)
        return 2 * np.pi * val

    def volume_exact(self):
        f = lambda e: profile(e)[0] ** 2 * (-profile_d(e)[1])
        val, _ = quad(f, 0, np.pi, limit=300)
        return np.pi * val

    def osc_lambda(self, n=4000):
        Th = np.linspace(1e-6, np.pi - 1e-6, n)
        lam = 0.5 * np.log(self.w_of_Theta(Th))
        return float(lam.max() - lam.min())


if __name__ == "__main__":
    cw = CurveWeight()
    A, V = cw.area_exact(), cw.volume_exact()
    r_, z_ = profile(np.linspace(0, np.pi, 2001))
    print(f"diameter {2*R0:.2f} um, min thickness {2*abs(z_[0]):.3f}, "
          f"max thickness {2*z_[:1001].max():.3f} um")
    print(f"area {A:.2f} um^2, volume {V:.2f} um^3, "
          f"sphericity {(np.pi**(1/3)*(6*V)**(2/3))/A:.4f}")
    print(f"osc lambda = {cw.osc_lambda():.4f}")

"""
revsurf.py: conformal weight for a closed axisymmetric surface of revolution
given as a radial graph R(psi) over the sphere, generalizing spheroid.py.

Surface: (R(psi) sin psi cos phi, R(psi) sin psi sin phi, R(psi) cos psi),
metric (R^2 + R'^2) dpsi^2 + R^2 sin^2 psi dphi^2. Mercator variable
t(psi) = int sqrt(R^2+R'^2)/(R sin psi) dpsi = ln tan(psi/2) + G(psi),
with G' = (sqrt(R^2+R'^2) - R)/(R sin psi) smooth up to the poles whenever
R'(0) = R'(pi) = 0. Matching to the sphere Mercator ln tan(Theta/2) with the
equator normalization t0 = G(pi/2) gives psi(Theta) from

    ln tan(psi/2) + G(psi) - G(pi/2) = ln tan(Theta/2),

and the conformal weight, in stable half-angle form (tau = tan(psi/2),
T = tan(Theta/2), T = tau e^{G - G(pi/2)}):

    w = R(psi)^2 e^{2(G - G(pi/2))} ((1 + tau^2)/(1 + T^2))^2 .

For R even in cos psi (equatorial symmetry) the Hersch centering holds by
symmetry. Area(M) = 2 pi int_0^pi R sin psi sqrt(R^2 + R'^2) dpsi is used as
an independent check of int_S w dA.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad


class RevolutionWeight:
    def __init__(self, R, Rp, n=4000):
        self.R, self.Rp = R, Rp
        psi = np.linspace(0.0, np.pi, n)
        gp = np.zeros(n)
        ps = psi[1:-1]
        num = np.sqrt(self.R(ps) ** 2 + self.Rp(ps) ** 2) - self.R(ps)
        gp[1:-1] = num / (self.R(ps) * np.sin(ps))
        # endpoint limits: numerator ~ R'^2/(2R), and R' ~ R''(pole) sin psi
        gp[0] = 0.0
        gp[-1] = 0.0
        G = CubicSpline(psi, gp).antiderivative()
        self.G = lambda p: G(p) - G(np.pi / 2)
        self.Gp = CubicSpline(psi, gp)

    def psi_of_Theta(self, Theta, iters=40):
        psi = Theta.copy()
        lt = np.log(np.tan(np.clip(Theta, 1e-14, np.pi - 1e-14) / 2))
        for _ in range(iters):
            F = np.log(np.tan(np.clip(psi, 1e-14, np.pi - 1e-14) / 2)) \
                + self.G(psi) - lt
            dF = 1.0 / np.maximum(np.sin(psi), 1e-14) + self.Gp(psi)
            psi = np.clip(psi - F / dF, 1e-12, np.pi - 1e-12)
        return psi

    def w_of_Theta(self, Theta):
        Theta = np.asarray(Theta, dtype=float)
        psi = self.psi_of_Theta(Theta)
        tau = np.tan(psi / 2)
        T = np.tan(Theta / 2)
        # sin psi / sin Theta = e^{-G} (1+T^2)/(1+tau^2), with T = tau e^{G}
        ratio = np.exp(-self.G(psi)) * (1 + T ** 2) / (1 + tau ** 2)
        return self.R(psi) ** 2 * ratio ** 2

    def __call__(self, pts):
        z = np.clip(pts[2], -1.0, 1.0)
        return self.w_of_Theta(np.arccos(z))

    def embed(self, pts):
        z = np.clip(pts[2], -1.0, 1.0)
        Theta = np.arccos(z)
        psi = self.psi_of_Theta(Theta)
        rho = np.hypot(pts[0], pts[1])
        cphi = np.where(rho > 1e-14, pts[0] / np.maximum(rho, 1e-14), 1.0)
        sphi = np.where(rho > 1e-14, pts[1] / np.maximum(rho, 1e-14), 0.0)
        Rv = self.R(psi)
        return np.vstack([Rv * np.sin(psi) * cphi,
                          Rv * np.sin(psi) * sphi,
                          Rv * np.cos(psi)])

    def area_exact(self):
        f = lambda p: (self.R(p) * np.sin(p)
                       * np.sqrt(self.R(p) ** 2 + self.Rp(p) ** 2))
        val, _ = quad(f, 0, np.pi, limit=200)
        return 2 * np.pi * val

    def osc_lambda(self, n=2000):
        Th = np.linspace(1e-6, np.pi - 1e-6, n)
        lam = 0.5 * np.log(self.w_of_Theta(Th))
        return float(lam.max() - lam.min())


def peanut(c=0.55):
    """Dumbbell of revolution R(psi) = 1 - c sin^2 psi: bulbs at the poles,
    neck of radius (1-c) at the equator."""
    R = lambda p: 1.0 - c * np.sin(p) ** 2
    Rp = lambda p: -2.0 * c * np.sin(p) * np.cos(p)
    return RevolutionWeight(R, Rp)

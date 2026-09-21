"""A torus of revolution whose meridian cross-section is the Evans-Fung
discocyte profile: the red-cell outline x = R cos t, z = (h(x)/2) sin t with
h(r) = c0 + c2 (r/R)^2 + c4 (r/R)^4 (biconcave), revolved about an axis at
distance D > R from the profile's center. Same conformal construction as
dtorus.py: rho(t) = D + R cos t, z(t) as above."""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
D, RC = 2.4, 1.0
c0, c2, c4 = 0.207, 2.003, -1.123        # Evans-Fung, in units of RC (thickness ~ 0.8 RC max)
h = lambda x: c0 + c2*(x/RC)**2 + c4*(x/RC)**4
dh = lambda x: (2*c2*x/RC**2 + 4*c4*x**3/RC**4)
rho  = lambda t: D + RC*np.cos(t)
drho = lambda t: -RC*np.sin(t)
zz   = lambda t: 0.5*h(RC*np.cos(t))*np.sin(t)
dz   = lambda t: 0.5*(dh(RC*np.cos(t))*(-RC*np.sin(t))*np.sin(t) + h(RC*np.cos(t))*np.cos(t))
speed = lambda t: np.hypot(drho(t), dz(t))
def u_of(t): return quad(lambda s: speed(s)/rho(s), 0.0, t, limit=200)[0]
U = u_of(2*np.pi)
_ts = np.linspace(0, 2*np.pi, 20001); _us = np.array([u_of(t) for t in _ts])
_spl = CubicSpline(_us, _ts)
def t_of(u): return _spl(np.mod(u, U))
def weight(X): return rho(t_of(X[0]))**2
def _reference_placeholder(): pass
if __name__ == "__main__":
    s = speed(_ts); print(f"U={U:.9f} tau={U/(2*np.pi):.6f}i  speed [{s.min():.3f},{s.max():.3f}]  rho [{rho(_ts).min():.3f},{rho(_ts).max():.3f}]")
    ddr = -RC*np.cos(_ts); ddz = np.gradient(dz(_ts), _ts)
    print("max curvature", (np.abs(drho(_ts)*ddz - dz(_ts)*ddr)/s**3).max())
    A = quad(lambda t: 2*np.pi*rho(t)*speed(t), 0, 2*np.pi, limit=200)[0]; print("area", A)

def reference(nmodes=40, nq=4001, k=12):
    """Fourier-Galerkin reference (azimuthal modes decouple); kept for checks."""
    from scipy.linalg import eigh
    u = np.linspace(0, U, nq, endpoint=False); du = U/nq; w = rho(t_of(u))**2
    vals = []
    for n in range(0, nmodes+1):
        M = 60; cols=[np.ones_like(u)]; lam=[0.0]
        for m in range(1, M+1):
            cols += [np.cos(2*np.pi*m*u/U), np.sin(2*np.pi*m*u/U)]; lam += [(2*np.pi*m/U)**2]*2
        Bm=np.array(cols); lam=np.array(lam); S=(Bm*du)@Bm.T; K=np.diag(lam)@S+n*n*S; Mw=(Bm*(du*w))@Bm.T
        ev=eigh(0.5*(K+K.T),0.5*(Mw+Mw.T),eigvals_only=True); vals += list(ev[:k])*(1 if n==0 else 2)
    return np.sort(np.array(vals))[:k]

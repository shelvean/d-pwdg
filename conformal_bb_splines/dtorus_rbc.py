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
if __name__ == '__main__':
    s = speed(_ts); print(f"U={U:.9f} tau={U/(2*np.pi):.6f}i  speed [{s.min():.3f},{s.max():.3f}]  rho [{rho(_ts).min():.3f},{rho(_ts).max():.3f}]")
    ddr = -RC*np.cos(_ts); ddz = np.gradient(dz(_ts), _ts)
    print("max curvature", (np.abs(drho(_ts)*ddz - dz(_ts)*ddr)/s**3).max())
    A = quad(lambda t: 2*np.pi*rho(t)*speed(t), 0, 2*np.pi, limit=200)[0]; print("area", A)

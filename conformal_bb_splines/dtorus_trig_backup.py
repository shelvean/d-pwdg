"""A distorted torus of revolution as a conformal metric on a flat torus.

Generating curve (rho(t), z(t)), t in [0,2pi), closed and away from the axis:
    rho = R + a cos t + b cos 3t,   z = c sin t + e sin 3t   (rounded-triangular section).
Metric (rho'^2+z'^2) dt^2 + rho^2 dphi^2. With the conformal variable
    u(t) = int_0^t sqrt(rho'^2+z'^2)/rho ds,     U = u(2pi),
the metric is rho^2 (du^2 + dphi^2): the surface is the flat torus
[0,U] x [0,2pi] with conformal factor w = rho(t(u))^2, modulus tau = i U/2pi.

Eigenvalue reference: w depends only on u, so with u_hat = f(u) e^{i n phi} the
problem -Delta f = mu w f splits into 1D problems -f'' + n^2 f = mu w f with
periodic f, solved by a dense Fourier-Galerkin method.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

R, A, B, C, E = 2.0, 0.80, 0.18, 0.90, 0.12
rho  = lambda t: R + A*np.cos(t) + B*np.cos(3*t)
drho = lambda t: -A*np.sin(t) - 3*B*np.sin(3*t)
zz   = lambda t: C*np.sin(t) + E*np.sin(3*t)
dz   = lambda t: C*np.cos(t) + 3*E*np.cos(3*t)
speed = lambda t: np.hypot(drho(t), dz(t))

def u_of(t): return quad(lambda s: speed(s)/rho(s), 0.0, t, limit=200)[0]
U = u_of(2*np.pi)

from scipy.interpolate import CubicSpline
_ts = np.linspace(0, 2*np.pi, 20001)
_us = np.array([u_of(t) for t in _ts])
_spl = CubicSpline(_us, _ts)          # accurate inverse of the conformal variable
def t_of(u):
    return _spl(np.mod(u, U))

def weight(X):
    """X: (2,n) points of [0,U]x[0,2pi] -> w"""
    return rho(t_of(X[0]))**2

def reference(nmodes=40, nq=4001, k=12):
    from scipy.linalg import eigh
    u = np.linspace(0, U, nq, endpoint=False); du = U/nq
    w = rho(t_of(u))**2
    vals = []
    for n in range(0, nmodes+1):
        # basis cos(2 pi m u/U), sin(2 pi m u/U), m = 0..M
        M = 60
        cols = [np.ones_like(u)]; lam = [0.0]
        for m in range(1, M+1):
            cols += [np.cos(2*np.pi*m*u/U), np.sin(2*np.pi*m*u/U)]
            lam += [(2*np.pi*m/U)**2, (2*np.pi*m/U)**2]
        Bm = np.array(cols); lam = np.array(lam)
        S = (Bm*du) @ Bm.T                       # mass in flat metric
        K = np.diag(lam) @ S + n*n*S             # -f'' + n^2 f
        Mw = (Bm*(du*w)) @ Bm.T
        K = 0.5*(K+K.T); Mw = 0.5*(Mw+Mw.T)
        ev = eigh(K, Mw, eigvals_only=True)
        mult = 1 if n == 0 else 2
        for v in ev[:k]: vals += [v]*mult
    return np.sort(np.array(vals))[:k]

if __name__ == '__main__':
    print(f"U = {U:.12f}   tau = {U/(2*np.pi):.12f} i")
    print(f"rho in [{rho(_ts).min():.4f}, {rho(_ts).max():.4f}]   "
          f"osc lambda = {0.5*(np.log(rho(_ts).max()**2)-np.log(rho(_ts).min()**2)):.4f}")
    area_exact = quad(lambda t: 2*np.pi*rho(t)*speed(t), 0, 2*np.pi, limit=200)[0]
    print(f"area (exact) = {area_exact:.12f}")
    ref = reference()
    print("reference eigenvalues:", np.round(ref, 9))

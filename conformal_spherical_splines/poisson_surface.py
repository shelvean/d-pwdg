"""
poisson_surface.py
==================

Solve the Poisson equation  -Delta_M u = f  on a smooth closed genus-zero
surface M given by a conformal parametrization Phi : S^2 -> M with conformal
weight w = e^{2 lambda}, using constrained spherical splines S^r_d.

Transplanted problem on the sphere (pullbacks marked with a tilde):

    find u~ in W^{1,2}(S^2),  int_S w u~ dA = 0, such that
    int_S grad u~ . grad v dA  =  int_S w f~ v dA   for all v,

because the Dirichlet form is conformally invariant and the surface measure
pulls back to w dA.  The stiffness matrix is therefore the round-sphere
matrix; the surface enters only the load vector (through w f~) and the
mean-zero constraint (through w).

Discrete problem in broken Bernstein-Bezier coefficients c = Z c^:

    (Z^T K Z) c^  +  g mu  =  Z^T F,      g^T c^ = 0,
    g = Z^T M_w c_1,   c_1 = broken coefficients of the constant 1,
    F_alpha = sum_T int_T w f~ B_alpha dA.

Manufactured solutions: u~ is prescribed on the sphere as an explicit
function of the ambient coordinates (so u = u~ o Phi^{-1} is a function on
M), and f = -Delta_M u = -w^{-1} Delta_S u~ is computed symbolically.
Errors are reported in L^2(M) = w-weighted L^2(S^2) and in the H^1(M)
seminorm = round-sphere Dirichlet seminorm (exact identities).

Reuses: sphsplines (mesh, smoothness matrix, quadrature, basis, assembly),
prolong (null-space matrix), spheroid / revsurf / rbc (conformal weights).
"""
import math, sys, time
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as sy

from sphsplines import (icosphere, meshsize_sphere, smoothness_sphere,
                        assemble_sphere, refquad, triquad_sphere, bern_eval,
                        grad_eval, interp_sphere)
from prolong import column_order, build_prolongation
from spheroid import SpheroidWeight
from revsurf import RevolutionWeight
from rbc import CurveWeight


# --------------------------------------------------------------------------
# manufactured data: u~ on S^2 given by an ambient expression F(x1,x2,x3)
# --------------------------------------------------------------------------
def manufactured(expr):
    """Return callables ut(pts), gradS_ut(pts), lapS_ut(pts) for the
    restriction of the ambient expression expr to the unit sphere.
    Surface gradient: (I - x x^T) grad F.  Laplace-Beltrami on the unit
    sphere: Delta_S F = Delta F - x^T Hess(F) x - 2 x . grad F  at |x| = 1."""
    x = sy.symbols('x1 x2 x3')
    F = sy.sympify(expr)
    g = [sy.diff(F, xi) for xi in x]
    H = [[sy.diff(F, xi, xj) for xj in x] for xi in x]
    xdotg = sum(x[i] * g[i] for i in range(3))
    xHx = sum(x[i] * H[i][j] * x[j] for i in range(3) for j in range(3))
    lapF = sum(H[i][i] for i in range(3))
    lapS = sy.simplify(lapF - xHx - 2 * xdotg)
    gS = [sy.simplify(g[i] - x[i] * xdotg) for i in range(3)]
    fF = sy.lambdify(x, F, 'numpy')
    fL = sy.lambdify(x, lapS, 'numpy')
    fG = [sy.lambdify(x, gi, 'numpy') for gi in gS]
    ut = lambda p: np.broadcast_to(fF(p[0], p[1], p[2]), p.shape[1:]).astype(float)
    lap = lambda p: np.broadcast_to(fL(p[0], p[1], p[2]), p.shape[1:]).astype(float)
    grad = lambda p: np.vstack([np.broadcast_to(fg(p[0], p[1], p[2]), p.shape[1:]) for fg in fG]).astype(float)
    return ut, grad, lap


# --------------------------------------------------------------------------
# load vector and error integrals
# --------------------------------------------------------------------------
def load_vector(v, t, d, weight, lap_ut, q=12):
    """F_alpha = sum_T int_T w f~ B_alpha dA with f~ = -w^{-1} Delta_S u~,
    i.e. the integrand is -(Delta_S u~) B_alpha.  Written out with w to
    show where a general right-hand side f on M enters: replace
    (w f~) by w(pts) * f(Phi(pts))."""
    lam, wref = refquad(q)
    m = (d + 1) * (d + 2) // 2
    Fv = np.zeros(m * t.shape[0])
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, W = triquad_sphere(v1, v2, v3, lam, wref)
        B = bern_eval(d, sb)                                  # m x nq
        wq = weight(pts) if weight is not None else np.ones(pts.shape[1])
        ft = -lap_ut(pts) / wq                                # f~ = f o Phi
        Fv[kk * m:(kk + 1) * m] = B @ (W * wq * ft)
    return Fv


def errors(v, t, d, c, weight, ut, grad_ut, q=12):
    """L^2(M) error (w-weighted on S^2), H^1(M) seminorm error (unweighted),
    and the w-weighted mean of the exact solution (to fix the constant)."""
    lam, wref = refquad(q)
    m = (d + 1) * (d + 2) // 2
    # first pass: weighted mean of u~ and of u_h so both are mean-zero
    num_e = num_h = den = 0.0
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, W = triquad_sphere(v1, v2, v3, lam, wref)
        wq = weight(pts) if weight is not None else np.ones(pts.shape[1])
        B = bern_eval(d, sb)
        uh = c[kk * m:(kk + 1) * m] @ B
        num_e += np.sum(W * wq * ut(pts)); num_h += np.sum(W * wq * uh)
        den += np.sum(W * wq)
    me, mh = num_e / den, num_h / den
    e0 = e1 = 0.0
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, W = triquad_sphere(v1, v2, v3, lam, wref)
        wq = weight(pts) if weight is not None else np.ones(pts.shape[1])
        B = bern_eval(d, sb)
        cT = c[kk * m:(kk + 1) * m]
        uh = cT @ B
        e0 += np.sum(W * wq * ((ut(pts) - me) - (uh - mh)) ** 2)
        Vinv = la.inv(np.column_stack([v1, v2, v3]))
        g = grad_eval(d, sb, Vinv)                            # 3 x m x nq
        dot = np.einsum('cq,cmq->mq', pts, g)
        gt = g - pts[:, None, :] * dot[None, :, :]
        gh = np.einsum('m,cmq->cq', cT, gt)
        e1 += np.sum(W * np.sum((grad_ut(pts) - gh) ** 2, axis=0))
    return math.sqrt(e0), math.sqrt(e1)


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------
def solve_poisson(v, t, d, r, weight, lap_ut, q=12, cache=None):
    """Returns broken coefficient vector c of the spline solution, plus
    (dim, timing) info.  cache: dict to reuse Z, K across geometries."""
    m = (d + 1) * (d + 2) // 2
    key = (d, r, t.shape[0])
    t0 = time.time()
    if cache is not None and key in cache:
        Z, K = cache[key]
    else:
        J = smoothness_sphere(v, t, d, r)                       # Step 2
        Z, D, Fr = build_prolongation(J, m * t.shape[0], column_order(v, t, d))
        _, K = assemble_sphere(v, t, d, q=q, weight=None)       # Step 3 (K)
        if cache is not None:
            cache[key] = (Z, K)
    Mw, _ = assemble_sphere(v, t, d, q=q, weight=weight)        # Step 3 (M_w)
    Fv = load_vector(v, t, d, weight, lap_ut, q=q)              # Step 3 (F)
    c1 = interp_sphere(v, t, d, lambda p: np.ones(p.shape[1]))  # constant 1
    A = (Z.T @ K @ Z).tocsr()
    g = np.asarray(Z.T @ (Mw @ c1)).ravel()
    b = np.asarray(Z.T @ Fv).ravel()
    n = A.shape[0]
    Aug = sp.bmat([[A, sp.csr_matrix(g[:, None])],
                   [sp.csr_matrix(g[None, :]), None]], format='csc')   # Step 4
    sol = spla.spsolve(Aug, np.concatenate([b, [0.0]]))
    chat = sol[:n]
    c = np.asarray(Z @ chat).ravel()
    return c, dict(dim=n, time=time.time() - t0, resid=la.norm(A @ chat + g * sol[n] - b) / la.norm(b))


# --------------------------------------------------------------------------
# geometries
# --------------------------------------------------------------------------
def geometries():
    G = {}
    G['sphere'] = None
    G['spheroid'] = SpheroidWeight(1.0, 1.5)
    R = lambda p: 1.0 - 0.75 * np.sin(p) ** 2
    Rp = lambda p: -1.5 * np.sin(p) * np.cos(p)
    G['dumbbell'] = RevolutionWeight(R, Rp)
    G['rbc'] = CurveWeight()
    return G


if __name__ == '__main__':
    # exact solution on the sphere, pushed to M by u = u~ o Phi^{-1}
    EXPR = 'exp(x1) * x2 + x3**3 * x1'      # not a spherical polynomial of any even degree
    ut, grad_ut, lap_ut = manufactured(EXPR)
    cases = [(2, 0, [1, 2, 3]), (4, 0, [1, 2, 3]), (6, 1, [1, 2])]
    geos = geometries()
    cache = {}
    rows = []
    print(f"exact solution u~ = {EXPR}")
    for name, wgt in geos.items():
        print(f"=== {name} ===")
        for d, r, levels in cases:
            prev = None
            for lev in levels:
                v, t = icosphere(lev)
                h = meshsize_sphere(v, t)
                c, info = solve_poisson(v, t, d, r, wgt, lap_ut, cache=cache)
                e0, e1 = errors(v, t, d, c, wgt, ut, grad_ut)
                r0 = r1 = float('nan')
                if prev:
                    r0 = math.log(prev[0] / e0) / math.log(prev[2] / h)
                    r1 = math.log(prev[1] / e1) / math.log(prev[2] / h)
                rows.append(dict(geo=name, d=d, r=r, lev=lev, h=h, dim=info['dim'],
                                 e0=e0, e1=e1, r0=r0, r1=r1, resid=info['resid']))
                print(f" d={d} r={r} lev={lev} h={h:.3f} dim={info['dim']:6d} "
                      f"L2(M) {e0:.3e} rate {r0:5.2f}  H1(M) {e1:.3e} rate {r1:5.2f} "
                      f"resid {info['resid']:.1e}  {info['time']:.1f}s")
                prev = (e0, e1, h)
    np.save('poisson_rows.npy', rows, allow_pickle=True)


# --------------------------------------------------------------------------
# Test 2: right-hand side prescribed on the physical surface, f(p) = p_x
# (the x-coordinate of the point p in M). Reference: dense spherical-harmonic
# Galerkin solve in the azimuthal-order-one sector (zonal weight).
# --------------------------------------------------------------------------
from spheroid import _legendre_norm

def sh_reference_poisson(weightTheta, fTheta, L=80, nG=600):
    """Solve a(u,v) = int w f~ v for f~ = fTheta(Theta) cos(phi) in the basis
    Pbar_l^1(cos Theta) cos(phi), l = 1..L. Returns U (coefficients)."""
    x, gw = np.polynomial.legendre.leggauss(nG)
    Th = np.arccos(x)
    P = _legendre_norm(L, 1, x)                     # l = 1..L
    Fl = P @ (weightTheta(Th) * fTheta(Th) * gw)    # common factor pi dropped
    ll = np.arange(1, L + 1, dtype=float)
    return Fl / (ll * (ll + 1.0))

def make_reference(geo, L=80):
    """Returns callables uref(pts), grad_uref(pts) on the sphere for f = p_x."""
    if geo is None:
        return (lambda p: 0.5 * p[0],
                lambda p: 0.5 * (np.vstack([np.ones(p.shape[1]), np.zeros(p.shape[1]), np.zeros(p.shape[1])])
                                 - p * p[0][None, :]))
    wTh = lambda Th: geo.w_of_Theta(Th)
    def fTh(Th):
        pts = np.vstack([np.sin(Th), np.zeros_like(Th), np.cos(Th)])
        return geo.embed(pts)[0]                    # p_x at phi = 0
    U = sh_reference_poisson(wTh, fTh, L=L)
    def uref(p):
        z = np.clip(p[2], -1, 1); rho = np.hypot(p[0], p[1])
        cphi = np.where(rho > 1e-14, p[0] / np.maximum(rho, 1e-14), 1.0)
        P = _legendre_norm(L, 1, z)
        return (U @ P) * cphi
    def grad_uref(p, h=1e-5):
        # central differences along two tangent directions, then assemble
        g = np.zeros_like(p)
        for i in range(3):
            e = np.zeros(3); e[i] = 1.0
            pp = p + h * e[:, None]; pp /= la.norm(pp, axis=0)
            pm = p - h * e[:, None]; pm /= la.norm(pm, axis=0)
            g[i] = (uref(pp) - uref(pm)) / (2 * h)
        # the FD gradient of the radially extended function is tangential
        return g - p * np.sum(p * g, axis=0)[None, :]
    return uref, grad_uref

def load_vector_physical(v, t, d, weight, f_phys, embed, q=12):
    """F_alpha = sum_T int_T w(x) f(Phi(x)) B_alpha(x) dA(x)."""
    lam, wref = refquad(q)
    m = (d + 1) * (d + 2) // 2
    Fv = np.zeros(m * t.shape[0])
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        pts, sb, W = triquad_sphere(v1, v2, v3, lam, wref)
        B = bern_eval(d, sb)
        wq = weight(pts) if weight is not None else np.ones(pts.shape[1])
        fq = f_phys(embed(pts) if embed is not None else pts)
        Fv[kk * m:(kk + 1) * m] = B @ (W * wq * fq)
    return Fv

def run_test2():
    geos = geometries()
    cases = [(2, 0, [1, 2, 3]), (4, 0, [1, 2, 3]), (6, 1, [1, 2])]
    f_phys = lambda P: P[0]
    cache = {}; rows = []
    print("Test 2: f(p) = p_x on M, reference from spherical harmonics (L=80)")
    for name, wgt in geos.items():
        uref, guref = make_reference(wgt)
        embed = (lambda p: p) if wgt is None else wgt.embed
        print(f"=== {name} ===")
        for d, r, levels in cases:
            prev = None
            for lev in levels:
                v, t = icosphere(lev); h = meshsize_sphere(v, t)
                m = (d + 1) * (d + 2) // 2
                key = (d, r, t.shape[0])
                if key not in cache:
                    J = smoothness_sphere(v, t, d, r)
                    Z, D, Fr = build_prolongation(J, m * t.shape[0], column_order(v, t, d))
                    _, K = assemble_sphere(v, t, d, q=12)
                    cache[key] = (Z, K)
                Z, K = cache[key]
                Mw, _ = assemble_sphere(v, t, d, q=12, weight=wgt)
                Fv = load_vector_physical(v, t, d, wgt, f_phys, embed)
                c1 = interp_sphere(v, t, d, lambda p: np.ones(p.shape[1]))
                A = (Z.T @ K @ Z).tocsr(); g = np.asarray(Z.T @ (Mw @ c1)).ravel()
                b = np.asarray(Z.T @ Fv).ravel(); n = A.shape[0]
                Aug = sp.bmat([[A, sp.csr_matrix(g[:, None])], [sp.csr_matrix(g[None, :]), None]], format='csc')
                sol = spla.spsolve(Aug, np.concatenate([b, [0.0]]))
                c = np.asarray(Z @ sol[:n]).ravel()
                e0, e1 = errors(v, t, d, c, wgt, uref, guref)
                r0 = r1 = float('nan')
                if prev:
                    r0 = math.log(prev[0] / e0) / math.log(prev[2] / h)
                    r1 = math.log(prev[1] / e1) / math.log(prev[2] / h)
                rows.append(dict(geo=name, d=d, r=r, lev=lev, h=h, dim=n, e0=e0, e1=e1, r0=r0, r1=r1))
                print(f" d={d} r={r} lev={lev} h={h:.3f} dim={n:6d} L2(M) {e0:.3e} rate {r0:5.2f}  H1(M) {e1:.3e} rate {r1:5.2f}")
                prev = (e0, e1, h)
    np.save('poisson_rows2.npy', rows, allow_pickle=True)
    return rows

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'test2':
    run_test2()

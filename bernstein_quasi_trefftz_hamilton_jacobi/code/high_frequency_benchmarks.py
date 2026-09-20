from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator

from eikonal_c0_pminus1 import (
    multiindices2, bernstein_values, bernstein_gradients,
    triangle_quadrature, structured_tri_mesh, tri_geometry,
    build_c0_assembly,
)

Array = np.ndarray


def scaled_structured_tri_mesh(n: int, bounds=((0.0, 1.0), (0.0, 1.0))):
    verts, tris = structured_tri_mesh(n)
    (xmin, xmax), (ymin, ymax) = bounds
    verts = verts.copy()
    verts[:, 0] = xmin + (xmax - xmin) * verts[:, 0]
    verts[:, 1] = ymin + (ymax - ymin) * verts[:, 1]
    return verts, tris


def boundary_edges(tris: Array):
    counts = {}
    for tri in tris:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = tuple(sorted((int(a), int(b))))
            counts[e] = counts.get(e, 0) + 1
    return [e for e, c in counts.items() if c == 1]


def bernstein_1d_matrix(p: int, ts: Array) -> Array:
    from math import comb
    B = np.zeros((len(ts), p + 1), float)
    for i, t in enumerate(ts):
        for k in range(p + 1):
            B[i, k] = comb(p, k) * (t ** k) * ((1.0 - t) ** (p - k))
    return B


class RadialS1:
    """Potter--Cameron analytic point-source test s1, adapted to 2D.

    u = cos(r)+r-1, s=1-sin(r), source at the origin.
    """
    source = np.array([0.0, 0.0])

    def u(self, x: Array) -> Array:
        r = np.linalg.norm(x - self.source, axis=1)
        return np.cos(r) + r - 1.0

    def grad(self, x: Array) -> Array:
        z = x - self.source
        r = np.linalg.norm(z, axis=1)
        g = np.zeros_like(z)
        m = r > 1e-14
        fac = (1.0 - np.sin(r[m])) / r[m]
        g[m] = z[m] * fac[:, None]
        # direction is singular at source; quadrature never samples exactly there
        return g

    def n2(self, x: Array) -> Array:
        r = np.linalg.norm(x - self.source, axis=1)
        return (1.0 - np.sin(r)) ** 2


class RadialS2:
    """Potter--Cameron analytic point-source test s2, adapted to 2D."""
    source = np.array([0.0, 0.0])

    def u(self, x: Array) -> Array:
        r2 = np.sum((x - self.source) ** 2, axis=1)
        return 0.5 * r2

    def grad(self, x: Array) -> Array:
        return x - self.source

    def n2(self, x: Array) -> Array:
        return np.sum((x - self.source) ** 2, axis=1)


class LinearSpeed:
    """Analytic linear-speed benchmark of Potter--Cameron section 5.3.

    We use one fixed global speed field
        1/s(x) = a + v dot x,
    with a=1 by default.  The paper's analytic point-source formula is then
    applied to arbitrary source locations in the same medium.
    """
    def __init__(self, source=(0.0, 0.0), a=1.0, v=(0.5, 0.25)):
        self.source = np.asarray(source, float)
        self.a = float(a)
        self.v = np.asarray(v, float)
        self.vnorm = float(np.linalg.norm(self.v))
        if self.vnorm == 0:
            raise ValueError("v must be nonzero")
        if self.a + np.min(self.v) < 0:
            raise ValueError("parameters may make speed negative")
        self.si = float(self.slowness(self.source[None, :])[0])

    def slowness(self, x: Array) -> Array:
        return 1.0 / (self.a + x @ self.v)

    def n2(self, x: Array) -> Array:
        s = self.slowness(x)
        return s * s

    def u(self, x: Array) -> Array:
        dx = x - self.source
        s = self.slowness(x)
        arg = 1.0 + 0.5 * self.si * s * (self.vnorm ** 2) * np.sum(dx * dx, axis=1)
        arg = np.maximum(arg, 1.0)
        return np.arccosh(arg) / self.vnorm

    def grad(self, x: Array) -> Array:
        # Accurate analytic-gradient evaluation by differentiating the closed form.
        dx = x - self.source
        r2 = np.sum(dx * dx, axis=1)
        s = self.slowness(x)
        ds = -(s * s)[:, None] * self.v[None, :]
        C = 0.5 * self.si * (self.vnorm ** 2)
        z = 1.0 + C * s * r2
        dz = C * (ds * r2[:, None] + 2.0 * s[:, None] * dx)
        den = self.vnorm * np.sqrt(np.maximum(z * z - 1.0, 1e-30))
        g = dz / den[:, None]
        # At the source the direction is singular; return zero only for safety.
        g[np.linalg.norm(dx, axis=1) < 1e-13] = 0.0
        return g


class RadialFactor:
    def __init__(self, source, scale=1.0):
        self.source = np.asarray(source, float)
        self.scale = float(scale)

    def u(self, x: Array) -> Array:
        return self.scale * np.linalg.norm(x - self.source, axis=1)

    def grad(self, x: Array) -> Array:
        z = x - self.source
        r = np.linalg.norm(z, axis=1)
        g = np.zeros_like(z)
        m = r > 1e-14
        g[m] = self.scale * z[m] / r[m, None]
        return g


class ZeroFactor:
    def u(self, x: Array) -> Array:
        return np.zeros(len(x))

    def grad(self, x: Array) -> Array:
        return np.zeros_like(x)


class FactoredC0EikonalPminus1:
    """Global C0 p-1 residual-moment solver for u=T+tau_h.

    The unknown is the smooth polynomial correction tau_h.  All element moments
    against P_{p-1} are minimized in nonlinear least squares.  For exact
    benchmarks the boundary trace of tau is interpolated in the 1D Bernstein
    basis on each boundary edge.
    """

    def __init__(self, n: int, p: int, problem, factor=None,
                 bounds=((0.0, 1.0), (0.0, 1.0)), quad_order=None):
        self.n = int(n); self.p = int(p); self.problem = problem
        self.factor = factor if factor is not None else ZeroFactor()
        self.bounds = bounds
        self.verts, self.tris = scaled_structured_tri_mesh(n, bounds)
        self.asm = build_c0_assembly(self.verts, self.tris, p)
        self.inds = multiindices2(p); self.Nd = len(self.inds)
        self.r = p - 1; self.test_inds = multiindices2(self.r); self.Nr = len(self.test_inds)
        self.qorder = quad_order or max(8, 2*p + 3)
        ref, wref = triangle_quadrature(self.qorder)
        self.ref = ref; self.wref = wref
        lam = np.column_stack([1-ref[:, 0]-ref[:, 1], ref[:, 0], ref[:, 1]])
        self.B = bernstein_values(p, lam)
        self.Q = bernstein_values(self.r, lam)
        self.elements = []
        for K, tri in enumerate(self.tris):
            V = self.verts[tri]
            _, gl = tri_geometry(V)
            J = np.column_stack((V[1]-V[0], V[2]-V[0]))
            det = abs(np.linalg.det(J))
            pts = V[0] + ref @ J.T
            G = bernstein_gradients(p, lam, gl)
            W = wref * det
            self.elements.append((pts, G, W))
        self.fixed, self.fixed_vals = self._boundary_coefficients()
        allidx = np.arange(len(self.asm.keys)); mask = np.ones(len(allidx), bool); mask[self.fixed] = False
        self.free = allidx[mask]
        self.free_pos = {int(g): i for i, g in enumerate(self.free)}

    def tau_exact(self, x: Array) -> Array:
        return self.problem.u(x) - self.factor.u(x)

    def grad_tau_exact(self, x: Array) -> Array:
        return self.problem.grad(x) - self.factor.grad(x)

    def _boundary_coefficients(self):
        b_edges = boundary_edges(self.tris)
        fixed_vals = {}
        # vertices
        bverts = sorted(set(v for e in b_edges for v in e))
        for v in bverts:
            key = ('v', int(v))
            if key in self.asm.keys:
                gid = self.asm.keys.index(key)
                fixed_vals[gid] = float(self.tau_exact(self.verts[[v]])[0])
        # edge Bernstein coefficients from interpolation at equispaced nodes
        ts = np.linspace(0.0, 1.0, self.p + 1)
        B1 = bernstein_1d_matrix(self.p, ts)
        for a, b in b_edges:
            va, vb = self.verts[a], self.verts[b]
            pts = (1-ts[:, None])*va + ts[:, None]*vb
            vals = self.tau_exact(pts)
            coeff = la.solve(B1, vals)
            for k in range(1, self.p):
                key = ('e', int(a), int(b), k)  # a<b from boundary_edges
                try:
                    gid = self.asm.keys.index(key)
                except ValueError:
                    continue
                fixed_vals[gid] = float(coeff[k])
        fixed = np.array(sorted(fixed_vals), int)
        vals = np.array([fixed_vals[g] for g in fixed], float)
        return fixed, vals

    def initial_coefficients(self):
        c = self.tau_exact(self.asm.dof_coords)
        c[self.fixed] = self.fixed_vals
        return c

    def unpack(self, z):
        c = self._base.copy(); c[self.free] = z; return c

    def residual_and_jac(self, z, want_jac=True):
        c = self.unpack(z)
        nrows = len(self.tris) * self.Nr
        R = np.zeros(nrows)
        Jmat = np.zeros((nrows, len(self.free))) if want_jac else None
        for K, tri in enumerate(self.tris):
            gids = self.asm.l2g[K]; ck = c[gids]
            pts, G, W = self.elements[K]
            gtau = np.einsum('qjd,j->qd', G, ck, optimize=True)
            gh = gtau + self.factor.grad(pts)
            Fq = 0.5 * (np.sum(gh*gh, axis=1) - self.problem.n2(pts))
            rows = slice(K*self.Nr, (K+1)*self.Nr)
            R[rows] = self.Q.T @ (W * Fq)
            if want_jac:
                adv = np.einsum('qd,qjd->qj', gh, G, optimize=True)
                Jloc = self.Q.T @ (W[:, None] * adv)
                for j, g in enumerate(gids):
                    pos = self.free_pos.get(int(g))
                    if pos is not None:
                        Jmat[rows, pos] += Jloc[:, j]
        return R, Jmat

    def solve(self, max_nfev=100, verbose=0):
        self._base = self.initial_coefficients()
        z0 = self._base[self.free].copy()
        fun = lambda z: self.residual_and_jac(z, False)[0]
        jac = lambda z: self.residual_and_jac(z, True)[1]
        t0 = time.perf_counter()
        sol = least_squares(fun, z0, jac=jac, method='trf', x_scale='jac', max_nfev=max_nfev,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=verbose)
        elapsed = time.perf_counter() - t0
        return self.unpack(sol.x), sol, elapsed

    def errors(self, c, order=None):
        order = order or max(10, 2*self.p + 4)
        ref, wref = triangle_quadrature(order)
        lam = np.column_stack([1-ref[:, 0]-ref[:, 1], ref[:, 0], ref[:, 1]])
        B = bernstein_values(self.p, lam)
        L2 = H1 = Nu = Ng = Linf = 0.0
        for K, tri in enumerate(self.tris):
            V = self.verts[tri]; _, gl = tri_geometry(V)
            J = np.column_stack((V[1]-V[0], V[2]-V[0])); det = abs(np.linalg.det(J))
            pts = V[0] + ref @ J.T; W = wref * det
            G = bernstein_gradients(self.p, lam, gl)
            ck = c[self.asm.l2g[K]]
            tauh = B @ ck; gtau = np.einsum('qjd,j->qd', G, ck, optimize=True)
            uh = self.factor.u(pts) + tauh
            gh = self.factor.grad(pts) + gtau
            ue = self.problem.u(pts); ge = self.problem.grad(pts)
            de = uh - ue
            L2 += np.sum(W*de**2); H1 += np.sum(W*np.sum((gh-ge)**2, axis=1))
            Nu += np.sum(W*ue**2); Ng += np.sum(W*np.sum(ge**2, axis=1))
            Linf = max(Linf, float(np.max(np.abs(de))))
        return math.sqrt(L2/max(Nu, 1e-30)), math.sqrt(H1/max(Ng, 1e-30)), Linf

    def summary(self, c, sol, seconds):
        eL2, eH1, einf = self.errors(c)
        self._base = c.copy()
        R, _ = self.residual_and_jac(c[self.free], False)
        return dict(n=self.n, ntri=len(self.tris), p=self.p, test_degree=self.p-1,
                    full_c0_dofs=len(self.asm.keys), fixed_boundary_dofs=len(self.fixed),
                    free_dofs=len(self.free), residual_rows=len(self.tris)*self.Nr,
                    retained_local_dim=self.p+1,
                    relL2=eL2, relH1=eH1, absLinf=einf,
                    residual_rms=float(la.norm(R)/math.sqrt(len(R))),
                    success=bool(sol.success), nfev=int(sol.nfev), seconds=float(seconds))


def solve_radial_s1(n=5, p=4):
    prob = RadialS1(); fac = RadialFactor(prob.source, scale=1.0)
    model = FactoredC0EikonalPminus1(n, p, prob, fac, bounds=((-1, 1), (-1, 1)))
    c, sol, sec = model.solve()
    return model.summary(c, sol, sec), c, model


def solve_linear_speed(source=(0.0, 0.0), n=5, p=4, v=(0.5, 0.25)):
    prob = LinearSpeed(source=source, v=v)
    fac = RadialFactor(prob.source, scale=prob.si)
    model = FactoredC0EikonalPminus1(n, p, prob, fac, bounds=((0, 1), (0, 1)))
    c, sol, sec = model.solve()
    return model.summary(c, sol, sec), c, model


def evaluate_model(model: FactoredC0EikonalPminus1, c: Array, pts: Array):
    U = np.full(len(pts), np.nan)
    for K, tri in enumerate(model.tris):
        V = model.verts[tri]
        A = np.column_stack((V[1]-V[0], V[2]-V[0]))
        rs = np.linalg.solve(A, (pts - V[0]).T)
        l1, l2 = rs[0], rs[1]; l0 = 1-l1-l2
        mask = (l0 >= -1e-10) & (l1 >= -1e-10) & (l2 >= -1e-10)
        if np.any(mask):
            lam = np.column_stack([l0[mask], l1[mask], l2[mask]])
            B = bernstein_values(model.p, lam)
            U[np.where(mask)[0]] = model.factor.u(pts[mask]) + B @ c[model.asm.l2g[K]]
    return U


def two_source_linear_speed(n=5, p=4, v=(0.5, 0.25), gridN=251):
    s1, c1, m1 = solve_linear_speed((0.0, 0.0), n=n, p=p, v=v)
    s2, c2, m2 = solve_linear_speed((0.8, 0.0), n=n, p=p, v=v)
    xx = np.linspace(0, 1, gridN); yy = np.linspace(0, 1, gridN)
    X, Y = np.meshgrid(xx, yy); pts = np.column_stack([X.ravel(), Y.ravel()])
    U1 = evaluate_model(m1, c1, pts); U2 = evaluate_model(m2, c2, pts)
    U = np.minimum(U1, U2)
    exact1 = m1.problem.u(pts); exact2 = m2.problem.u(pts); exact = np.minimum(exact1, exact2)
    err = U - exact
    relL2 = float(np.linalg.norm(err) / np.linalg.norm(exact))
    absLinf = float(np.max(np.abs(err)))
    return dict(p=p, n=n, relL2=relL2, absLinf=absLinf,
                branch1_seconds=s1['seconds'], branch2_seconds=s2['seconds'],
                total_seconds=s1['seconds']+s2['seconds'],
                full_c0_dofs_per_branch=s1['full_c0_dofs'], retained_local_dim=p+1), (X, Y, U, exact), (s1, s2)


# --- Marmousi support ----------------------------------------------------

def fast_sweeping(slowness: Array, dx: float, dy: float, source_ij: Tuple[int, int], max_iter=100, tol=1e-10):
    """First-order 2D fast-sweeping reference solver on a Cartesian grid.

    This is intentionally a reference/initialization routine, not the qT method.
    """
    s = np.asarray(slowness, float)
    ny, nx = s.shape
    U = np.full((ny, nx), np.inf, float)
    sj, si = source_ij
    U[sj, si] = 0.0

    # isotropic formula for unequal spacings: solve ((u-a)/dx)^2+((u-b)/dy)^2=s^2
    def update(j, i):
        if j == sj and i == si:
            return 0.0
        a = min(U[j, i-1] if i > 0 else np.inf, U[j, i+1] if i+1 < nx else np.inf)
        b = min(U[j-1, i] if j > 0 else np.inf, U[j+1, i] if j+1 < ny else np.inf)
        q = s[j, i]
        vals = []
        if np.isfinite(a): vals.append(a + q*dx)
        if np.isfinite(b): vals.append(b + q*dy)
        best = min(vals) if vals else np.inf
        if np.isfinite(a) and np.isfinite(b):
            A = 1/dx**2 + 1/dy**2
            B = -2*(a/dx**2 + b/dy**2)
            C = a*a/dx**2 + b*b/dy**2 - q*q
            disc = max(B*B - 4*A*C, 0.0)
            root = (-B + math.sqrt(disc))/(2*A)
            if root >= max(a, b): best = min(best, root)
        return best

    orders = [
        (range(ny), range(nx)),
        (range(ny-1, -1, -1), range(nx)),
        (range(ny), range(nx-1, -1, -1)),
        (range(ny-1, -1, -1), range(nx-1, -1, -1)),
    ]
    for _ in range(max_iter):
        old = U.copy()
        for jr, ir in orders:
            for j in jr:
                for i in ir:
                    cand = update(j, i)
                    if cand < U[j, i]: U[j, i] = cand
        if np.nanmax(np.abs(U-old)[np.isfinite(U-old)]) < tol:
            break
    return U


class GridSlowness:
    def __init__(self, velocity: Array, width_km=9.2, depth_km=3.0):
        self.velocity = np.asarray(velocity, float)
        if self.velocity.ndim != 2: raise ValueError('velocity must be 2D')
        self.slowness = 1.0 / self.velocity
        ny, nx = self.velocity.shape
        self.x = np.linspace(0.0, width_km, nx)
        self.y = np.linspace(0.0, depth_km, ny)
        self._interp = RegularGridInterpolator((self.y, self.x), self.slowness, bounds_error=False, fill_value=None)

    def n2(self, pts: Array) -> Array:
        s = self._interp(np.column_stack([pts[:,1], pts[:,0]]))
        return s*s


def load_velocity(path: str) -> Array:
    p = Path(path)
    if p.suffix == '.npy': return np.load(p)
    if p.suffix == '.npz':
        d = np.load(p)
        for key in ('velocity', 'vp', 'v', 'model'):
            if key in d: return d[key]
        return d[d.files[0]]
    raise ValueError('Marmousi loader currently accepts .npy or .npz velocity arrays')

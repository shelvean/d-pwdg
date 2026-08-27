"""Search-cost scaling experiment: global skeleton-residual direction search
(one DG assembly+solve per nonlinear trial) versus local Cauchy-trace
variable projection (local least squares per trial, one final DG solve),
on three refinement levels of the exact annulus a=0.5 < r < 1.

Levels: (n_theta, n_r) = (8,1), (16,2), (32,4)  ->  8, 32, 128 elements.
Field:  outgoing Herglotz-DtN reference field at k=16, |m|<=70 (the same
        density as in the manuscript); Dirichlet trace on r=a, circular
        DtN on r=1 with g_R = 0 exactly.
Search: three real plane-wave directions shared by all elements, Nelder-
        Mead from the same equispaced start with identical tolerances for
        both methods.  Everything in IEEE binary64.
Output: data/timing_levels.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.optimize import minimize

k = 16.0
a, R = 0.5, 1.0
alpha = beta = delta = 0.5
NdtN = 70
QPW = 3                       # three shared directions
LEVELS = [(8, 1), (16, 2), (32, 4)]
NQ_LINE, NQ_ARC_IN, NQ_ARC_BND = 40, 50, 80
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- exact field
def herglotz_dtn_coeffs():
    nq = 4096
    phi = 2 * np.pi * np.arange(nq) / nq
    g = np.exp(1.55 * np.cos(phi - 0.48) + 0.22 * np.cos(2 * phi + 0.35)) \
        * np.exp(1j * (0.28 * np.sin(3 * phi + 0.20) + 0.10 * np.cos(5 * phi - 0.40)))
    ghat = np.fft.fft(g) / nq                       # ghat[m] for m mod nq
    ms = np.arange(-NdtN, NdtN + 1)
    gm = ghat[ms % nq]
    return ms, 2 * np.pi * (1j ** ms) * jv(ms, k * a) * gm / hankel1(ms, k * a)

MS, CM = herglotz_dtn_coeffs()

def u_exact(x):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    return (hankel1(MS[None, :], k * r[:, None])
            * np.exp(1j * np.outer(th, MS))) @ CM

def grad_exact(x):
    r = np.hypot(x[:, 0], x[:, 1]); th = np.arctan2(x[:, 1], x[:, 0])
    E = np.exp(1j * np.outer(th, MS))
    ur = (k * h1vp(MS[None, :], k * r[:, None], 1) * E) @ CM
    ut = (hankel1(MS[None, :], k * r[:, None]) * E) @ (1j * MS * CM) / r
    er = np.c_[np.cos(th), np.sin(th)]; et = np.c_[-np.sin(th), np.cos(th)]
    return ur[:, None] * er + ut[:, None] * et

# ------------------------------------------------------------------- meshing
class Mesh:
    """Curved annulus mesh with nt congruent sectors and nr radial layers.
    All face quadrature geometry and exact-field data are precomputed so a
    direction trial pays only for basis evaluations."""

    def __init__(self, nt, nr):
        self.nt, self.nr = nt, nr
        self.ne = nt * nr
        dr = (R - a) / nr
        self.cen = np.zeros((self.ne, 2)); self.hK = np.zeros(self.ne)
        for i in range(nr):
            r0, r1 = a + i * dr, a + (i + 1) * dr
            for j in range(nt):
                t0 = 2 * np.pi * j / nt; tm = t0 + np.pi / nt
                K = i * nt + j
                self.cen[K] = 0.5 * (r0 + r1) * np.array([np.cos(tm), np.sin(tm)])
                self.hK[K] = 0.55 * np.hypot(r1 - r0, r1 * 2 * np.pi / nt)
        zl, wl = leggauss(NQ_LINE)
        self.faces = []                     # interior faces: (x, w, n, K-, K+)
        for i in range(nr):                 # radial interior segments
            r0, r1 = a + i * dr, a + (i + 1) * dr
            for j in range(nt):
                t = 2 * np.pi * j / nt
                e = np.array([np.cos(t), np.sin(t)])
                x = (0.5 * ((r1 - r0) * zl + (r1 + r0)))[:, None] * e
                w = 0.5 * (r1 - r0) * wl
                n = np.array([-np.sin(t), np.cos(t)])  # from K- toward K+
                Km, Kp = i * nt + (j - 1) % nt, i * nt + j
                self.faces.append((x, w, np.tile(n, (NQ_LINE, 1)), Km, Kp))
        za, wa = leggauss(NQ_ARC_IN)
        for i in range(1, nr):              # circular interior arcs
            r = a + i * dr
            for j in range(nt):
                t0, t1 = 2 * np.pi * j / nt, 2 * np.pi * (j + 1) / nt
                th = 0.5 * ((t1 - t0) * za + (t1 + t0))
                x = r * np.c_[np.cos(th), np.sin(th)]
                w = 0.5 * (t1 - t0) * r * wa
                n = np.c_[np.cos(th), np.sin(th)]      # from ring i-1 outward
                self.faces.append((x, w, n, (i - 1) * nt + j, i * nt + j))
        zb, wb = leggauss(NQ_ARC_BND)
        self.inner = []                     # (x, w, n, K, g_D)
        for j in range(nt):
            t0, t1 = 2 * np.pi * j / nt, 2 * np.pi * (j + 1) / nt
            th = 0.5 * ((t1 - t0) * zb + (t1 + t0))
            x = a * np.c_[np.cos(th), np.sin(th)]
            w = 0.5 * (t1 - t0) * a * wb
            self.inner.append((x, w, -np.c_[np.cos(th), np.sin(th)], j, u_exact(x)))
        xs, ws, ths, Ks = [], [], [], []    # outer DtN ring, gathered globally
        for j in range(nt):
            t0, t1 = 2 * np.pi * j / nt, 2 * np.pi * (j + 1) / nt
            th = 0.5 * ((t1 - t0) * zb + (t1 + t0))
            xs.append(R * np.c_[np.cos(th), np.sin(th)])
            ws.append(0.5 * (t1 - t0) * R * wb)
            ths.append(th); Ks.append((nr - 1) * nt + j)
        self.ox = np.vstack(xs); self.ow = np.concatenate(ws)
        self.oth = np.concatenate(ths); self.oK = Ks
        ms = np.arange(-NdtN, NdtN + 1)
        self.oE = np.exp(-1j * np.outer(self.oth, ms))          # analysis
        self.oS = np.exp(1j * np.outer(self.oth, ms))           # synthesis
        self.omult = k * h1vp(ms, k * R, 1) / hankel1(ms, k * R)
        # per-element local Cauchy-trace modal data for the local objective
        self.loc = []
        nq = 256; t = 2 * np.pi * np.arange(nq) / nq; wq = 2 * np.pi / nq
        for K in range(self.ne):
            h = self.hK[K]
            M = min(45, int(np.ceil(k * h)) + 18)
            mm = np.arange(-M, M + 1)
            xq = self.cen[K] + h * np.c_[np.cos(t), np.sin(t)]
            nrm = np.c_[np.cos(t), np.sin(t)]
            uv = u_exact(xq); un = np.einsum('qd,qd->q', grad_exact(xq), nrm)
            tau2 = 2 * np.pi * h * k * (jv(mm, k * h) ** 2 + jvp(mm, k * h) ** 2)
            tau = np.sqrt(tau2)
            Ph = np.exp(1j * np.outer(t, mm))
            ip = wq * h * (k * (Ph.conj().T @ uv)
                           + (1 / k) * ((k * jvp(mm, k * h) / jv(mm, k * h))[:, None]
                                        * Ph.conj().T @ un).ravel() * 0)  # placeholder
            # trace inner product <u, phi_m> done properly below
            phi_v = jv(mm, k * h) * Ph            # values of modes on the circle
            phi_n = k * jvp(mm, k * h) * Ph       # normal derivatives
            ip = wq * h * (k * (phi_v.conj().T @ uv) + (1 / k) * (phi_n.conj().T @ un))
            self.loc.append((mm, tau, ip / tau))  # equilibrated coefficients b_m

# --------------------------------------------------------------- DG machinery
def basis(mesh, K, x, d):
    ph = np.exp(1j * k * ((x - mesh.cen[K]) @ d.T))
    return ph, 1j * k * ph[:, :, None] * d[None, :, :]

def assemble(mesh, thetas):
    d = np.c_[np.cos(thetas), np.sin(thetas)]
    N = mesh.ne * QPW
    Kmat = np.zeros((N, N), complex); f = np.zeros(N, complex)
    for x, w, n, Km, Kp in mesh.faces:
        for Kt, sgn_t in ((Km, 1.0), (Kp, -1.0)):
            Vt, Gt = basis(mesh, Kt, x, d)
            jUt = sgn_t * Vt[:, :, None] * n[:, None, :]
            jGt = sgn_t * np.einsum('qpd,qd->qp', Gt, n)
            it = slice(Kt * QPW, (Kt + 1) * QPW)
            for Ks_, sgn_s in ((Km, 1.0), (Kp, -1.0)):
                Vs, Gs = basis(mesh, Ks_, x, d)
                jUs = sgn_s * Vs[:, :, None] * n[:, None, :]
                jGs = sgn_s * np.einsum('qpd,qd->qp', Gs, n)
                isl = slice(Ks_ * QPW, (Ks_ + 1) * QPW)
                A = np.einsum('q,qi,qj->ij', w, np.conj(jGt), 0.5 * Vs) \
                    - np.einsum('q,qid,qjd->ij', w, np.conj(jUt), 0.5 * Gs) \
                    - 1j * alpha * k * np.einsum('q,qid,qjd->ij', w, np.conj(jUt), jUs) \
                    + (beta / (1j * k)) * np.einsum('q,qi,qj->ij', w, np.conj(jGt), jGs)
                Kmat[it, isl] += A
    for x, w, n, K, gd in mesh.inner:
        V, G = basis(mesh, K, x, d)
        DN = np.einsum('qpd,qd->qp', G, n)
        i = slice(K * QPW, (K + 1) * QPW)
        Kmat[i, i] += -np.einsum('q,qi,qj->ij', w, np.conj(V), DN) \
                      - 1j * alpha * k * np.einsum('q,qi,qj->ij', w, np.conj(V), V)
        f[i] += -np.einsum('q,q,qi->i', w, gd, np.conj(DN) + alpha * 1j * k * np.conj(V))
    Q = len(mesh.ow)
    Vg = np.zeros((Q, N), complex); Dg = np.zeros((Q, N), complex)
    nb = NQ_ARC_BND
    for jj, K in enumerate(mesh.oK):
        sl = slice(jj * nb, (jj + 1) * nb)
        x = mesh.ox[sl]
        nrm = np.c_[np.cos(mesh.oth[sl]), np.sin(mesh.oth[sl])]
        V, G = basis(mesh, K, x, d)
        Vg[sl, K * QPW:(K + 1) * QPW] = V
        Dg[sl, K * QPW:(K + 1) * QPW] = np.einsum('qpd,qd->qp', G, nrm)
    C = (mesh.oE.T @ (mesh.ow[:, None] * Vg)) / (2 * np.pi)
    TV = mesh.oS @ (mesh.omult[:, None] * C)
    Res = Dg - TV
    W = mesh.ow[:, None]
    Kmat += Dg.conj().T @ (W * Vg) - Vg.conj().T @ (W * TV) \
            + delta / (1j * k) * (Res.conj().T @ (W * Res))
    return Kmat, f, d

def skeleton_J(mesh, thetas, c):
    d = np.c_[np.cos(thetas), np.sin(thetas)]
    J = 0.0
    for x, w, n, Km, Kp in mesh.faces:
        Vm, Gm = basis(mesh, Km, x, d); Vp, Gp = basis(mesh, Kp, x, d)
        um = Vm @ c[Km * QPW:(Km + 1) * QPW]; up = Vp @ c[Kp * QPW:(Kp + 1) * QPW]
        gm = np.einsum('qpd,p->qd', Gm, c[Km * QPW:(Km + 1) * QPW])
        gp = np.einsum('qpd,p->qd', Gp, c[Kp * QPW:(Kp + 1) * QPW])
        J += k * np.sum(w * np.abs(um - up) ** 2)
        J += (1 / k) * np.sum(w * np.abs(np.einsum('qd,qd->q', gm - gp, n)) ** 2)
    for x, w, n, K, gd in mesh.inner:
        V, _ = basis(mesh, K, x, d)
        J += k * np.sum(w * np.abs(V @ c[K * QPW:(K + 1) * QPW] - gd) ** 2)
    Q = len(mesh.ow); N = mesh.ne * QPW
    Vg = np.zeros((Q, N), complex); Dg = np.zeros((Q, N), complex)
    nb = NQ_ARC_BND
    for jj, K in enumerate(mesh.oK):
        sl = slice(jj * nb, (jj + 1) * nb)
        nrm = np.c_[np.cos(mesh.oth[sl]), np.sin(mesh.oth[sl])]
        V, G = basis(mesh, K, mesh.ox[sl], d)
        Vg[sl, K * QPW:(K + 1) * QPW] = V
        Dg[sl, K * QPW:(K + 1) * QPW] = np.einsum('qpd,qd->qp', G, nrm)
    C = (mesh.oE.T @ (mesh.ow[:, None] * (Vg @ c)[:, None])) / (2 * np.pi)
    TV = mesh.oS @ (mesh.omult[:, None] * C)
    J += (1 / k) * np.sum(mesh.ow * np.abs(Dg @ c - TV.ravel()) ** 2)
    return J

def l2_error(mesh, thetas, c):
    d = np.c_[np.cos(thetas), np.sin(thetas)]
    z, wz = leggauss(40); zt, wt = leggauss(40)
    dr = (R - a) / mesh.nr
    num = den = 0.0
    for i in range(mesh.nr):
        r0, r1 = a + i * dr, a + (i + 1) * dr
        r = 0.5 * ((r1 - r0) * z + (r1 + r0)); wr = 0.5 * (r1 - r0) * wz
        for j in range(mesh.nt):
            K = i * mesh.nt + j
            t0, t1 = 2 * np.pi * j / mesh.nt, 2 * np.pi * (j + 1) / mesh.nt
            th = 0.5 * ((t1 - t0) * zt + (t1 + t0)); wth = 0.5 * (t1 - t0) * wt
            RR, TT = np.meshgrid(r, th, indexing='ij')
            W2 = (np.outer(wr, wth) * RR).ravel()
            x = np.c_[(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()]
            V, _ = basis(mesh, K, x, d)
            uh = V @ c[K * QPW:(K + 1) * QPW]
            ue = u_exact(x)
            num += np.sum(W2 * np.abs(uh - ue) ** 2)
            den += np.sum(W2 * np.abs(ue) ** 2)
    return np.sqrt(num / den)

# ------------------------------------------------------------- the two methods
START = np.array([0.5, 0.5 + 2 * np.pi / 3, 0.5 + 4 * np.pi / 3])
NMOPT = dict(method='Nelder-Mead',
             options=dict(xatol=1e-4, fatol=1e-10, maxfev=400))

def run_global_residual(mesh):
    evals = [0]
    def obj(th):
        evals[0] += 1
        Kmat, f, _ = assemble(mesh, th)
        c = np.linalg.solve(Kmat, f)
        return skeleton_J(mesh, th, c)
    t0 = time.perf_counter()
    res = minimize(obj, START, **NMOPT)
    t_sel = time.perf_counter() - t0
    t1 = time.perf_counter()
    Kmat, f, _ = assemble(mesh, res.x)
    c = np.linalg.solve(Kmat, f)
    t_dg = time.perf_counter() - t1
    return dict(evals=evals[0], t_sel=t_sel, t_dg=t_dg,
                err=l2_error(mesh, res.x, c), th=np.sort(res.x % (2 * np.pi)))

def run_local_trace(mesh):
    evals = [0]
    def obj(th):
        evals[0] += 1
        r2 = 0.0
        for mm, tau, b in mesh.loc:
            A = tau[:, None] * (1j ** mm)[:, None] * np.exp(-1j * np.outer(mm, th))
            cK, res, *_ = np.linalg.lstsq(A, b, rcond=None)
            r = b - A @ cK
            r2 += np.real(np.vdot(r, r))
        return r2
    t0 = time.perf_counter()
    res = minimize(obj, START, **NMOPT)
    t_sel = time.perf_counter() - t0
    t1 = time.perf_counter()
    Kmat, f, _ = assemble(mesh, res.x)
    c = np.linalg.solve(Kmat, f)
    t_dg = time.perf_counter() - t1
    return dict(evals=evals[0], t_sel=t_sel, t_dg=t_dg,
                err=l2_error(mesh, res.x, c), th=np.sort(res.x % (2 * np.pi)))

# ------------------------------------------------------------------------ run
if __name__ == '__main__':
    rows = []
    for nt, nr in LEVELS:
        print(f'== level ({nt},{nr}): {nt*nr} elements, {nt*nr*QPW} dofs', flush=True)
        mesh = Mesh(nt, nr)
        A = run_global_residual(mesh)
        B = run_local_trace(mesh)
        row = dict(nt=nt, nr=nr, ne=nt * nr, dofs=nt * nr * QPW,
                   evals_A=A['evals'], sel_A=A['t_sel'], dg_A=A['t_dg'],
                   tot_A=A['t_sel'] + A['t_dg'], err_A=A['err'],
                   per_trial_A=A['t_sel'] / A['evals'],
                   evals_B=B['evals'], sel_B=B['t_sel'], dg_B=B['t_dg'],
                   tot_B=B['t_sel'] + B['t_dg'], err_B=B['err'],
                   per_trial_B=B['t_sel'] / B['evals'],
                   ratio=(A['t_sel'] + A['t_dg']) / (B['t_sel'] + B['t_dg']),
                   th_A=';'.join(f'{np.degrees(t):.3f}' for t in A['th']),
                   th_B=';'.join(f'{np.degrees(t):.3f}' for t in B['th']))
        rows.append(row)
        print(row, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / 'data' / 'timing_levels.csv', index=False)
    print(pd.DataFrame(rows).to_string())

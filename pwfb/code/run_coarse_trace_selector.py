"""Selection from estimated trace data (review point: the decisive
experiments give the selector exact analytic traces; test the adaptive
loop with traces estimated from a coarse numerical solution).

Benchmark: centered outgoing Hankel field on the 8-sector annulus at
k = 16 with inner Dirichlet and outer truncated DtN (Section 7.5).
Protocol, all binary64:
  0. coarse solve: p0 equispaced plane waves plus the J_0 mode;
  1. estimate the local modal coefficients a_m of the solution about the
     element center from its Cauchy data on the inscribed circle
     (radius h_in = 0.24, entirely inside the element);
  2. run the unchanged stability-aware selector on the estimated modal
     tail (same tau weights at h = 0.44, same ESPRIT init, same
     variable-projection polish, same rank filter);
  3. solve with the selected space; record error and selection;
  4. repeat 1-3 once from the improved solution (cycle 2).
Control: the exact-trace selection at the same budget (Table 7 row).

Output: data/coarse_trace_selector.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag, cholesky, eigh, solve_triangular, svdvals, svd
from scipy.optimize import least_squares
from scipy.special import jv, jvp, hankel1, h1vp

k = 16.0; a = 0.5; R = 1.0; ne = 8
alpha = beta = delta = 0.5
h = 0.44; rc = 0.75; h_in = 0.24
tau_rank = 1e-12; NdtN = 70
M_EST = 40                     # estimated modal window |m| <= M_EST
Q_CAND = [0, 2, 4, 6]
ROOT = Path(__file__).resolve().parents[1]

cent_th = (np.arange(ne) + 0.5) * 2 * np.pi / ne
CEN = rc * np.c_[np.cos(cent_th), np.sin(cent_th)]
ms_est = np.arange(-M_EST, M_EST + 1)
tau_sel = np.sqrt(2 * np.pi * h * k *
                  (jv(ms_est, k * h) ** 2 + jvp(ms_est, k * h) ** 2))

# exact modal coefficients about CEN[0] (Graf): a_m = (-1)^m H_m(k rc)
a_exact = ((-1.0) ** ms_est) * hankel1(ms_est, k * rc)
gamma_exact = (1j) ** (-ms_est) * a_exact

# ----------------------------------------------------- hybrid PWDG machinery
def basis_vals(K, rel_angles, M, x):
    parts, grads = [], []
    if len(rel_angles):
        d = np.c_[np.cos(rel_angles + cent_th[K]), np.sin(rel_angles + cent_th[K])]
        ph = np.exp(1j * k * ((x - CEN[K]) @ d.T))
        parts.append(ph); grads.append(1j * k * ph[:, :, None] * d[None, :, :])
    modes = np.arange(-M, M + 1)
    y = x - CEN[K]; rho = np.hypot(y[:, 0], y[:, 1])
    phi = np.arctan2(y[:, 1], y[:, 0]) - cent_th[K]
    rr = np.where(rho > 1e-14, rho, 1e-14)
    E = np.exp(1j * np.outer(phi, modes)); Z = k * rho[:, None]
    J = jv(modes[None, :], Z); JP = jvp(modes[None, :], Z, 1)
    V = J * E
    er = np.c_[np.cos(phi + cent_th[K]), np.sin(phi + cent_th[K])]
    et = np.c_[-np.sin(phi + cent_th[K]), np.cos(phi + cent_th[K])]
    G = (k * JP * E)[:, :, None] * er[:, None, :] \
        + ((1j * modes[None, :] / rr[:, None]) * V)[:, :, None] * et[:, None, :]
    parts.append(V); grads.append(G)
    return np.concatenate(parts, axis=1), np.concatenate(grads, axis=1)

def local_gram(rel_angles, M, nq=4096):
    t = 2 * np.pi * np.arange(nq) / nq
    x = CEN[0] + h * np.c_[np.cos(t + cent_th[0]), np.sin(t + cent_th[0])]
    n = np.c_[np.cos(t + cent_th[0]), np.sin(t + cent_th[0])]
    V, G = basis_vals(0, rel_angles, M, x)
    DN = np.einsum('qpd,qd->qp', G, n)
    w = h * 2 * np.pi / nq
    Gram = w * (k * (V.conj().T @ V) + (1 / k) * (DN.conj().T @ DN))
    Gram = (Gram + Gram.conj().T) / 2
    d = np.sqrt(np.maximum(np.real(np.diag(Gram)), 1e-300)); D = np.diag(1 / d)
    lam, U = eigh(D @ Gram @ D); keep = lam > tau_rank * lam.max()
    T = D @ U[:, keep] @ np.diag(1 / np.sqrt(lam[keep]))
    return T, int(keep.sum())

def edge_quad_line(x0, x1, nq=100):
    z, w = leggauss(nq)
    return .5 * ((1 - z)[:, None] * x0 + (1 + z)[:, None] * x1), \
           .5 * np.linalg.norm(x1 - x0) * w

def arc_quad(r, t0, t1, nq=128):
    z, w = leggauss(nq); th = .5 * ((t1 - t0) * z + (t1 + t0))
    return th, r * np.c_[np.cos(th), np.sin(th)], .5 * (t1 - t0) * r * w

def assemble_raw(rel_angles, M, nq_line=100, nq_arc=128):
    p = len(rel_angles) + 2 * M + 1; N = ne * p
    Kmat = np.zeros((N, N), complex); f = np.zeros(N, complex)
    for j in range(ne):
        th = 2 * np.pi * j / ne
        x0 = a * np.array([np.cos(th), np.sin(th)])
        x1 = R * np.array([np.cos(th), np.sin(th)])
        x, w = edge_quad_line(x0, x1, nq_line)
        KL, KR = (j - 1) % ne, j
        nL = np.array([-np.sin(th), np.cos(th)]); nR = -nL
        cache = {Ks: (basis_vals(Ks, rel_angles, M, x), ns)
                 for Ks, ns in ((KL, nL), (KR, nR))}
        for Ks in (KL, KR):
            (VK, GK), nside = cache[Ks]
            idxK = slice(Ks * p, (Ks + 1) * p)
            jumpUK = VK[:, :, None] * nside[None, None, :]
            jumpGK = np.einsum('qpd,d->qp', GK, nside)
            for Ls in (KL, KR):
                (VL, GL), ntest = cache[Ls]
                idxL = slice(Ls * p, (Ls + 1) * p)
                jumpUL = VL[:, :, None] * ntest[None, None, :]
                jumpGL = np.einsum('qpd,d->qp', GL, ntest)
                Kmat[idxL, idxK] += \
                    np.einsum('q,qi,qj->ij', w, np.conj(jumpGL), .5 * VK) \
                    - np.einsum('q,qid,qjd->ij', w, np.conj(jumpUL), .5 * GK) \
                    - 1j * alpha * k * np.einsum('q,qid,qjd->ij', w, np.conj(jumpUL), jumpUK) \
                    + (beta / (1j * k)) * np.einsum('q,qi,qj->ij', w, np.conj(jumpGL), jumpGK)
    for K in range(ne):
        t0, t1 = 2 * np.pi * K / ne, 2 * np.pi * (K + 1) / ne
        th, x, w = arc_quad(a, t0, t1, nq_arc)
        n = -np.c_[np.cos(th), np.sin(th)]
        V, G = basis_vals(K, rel_angles, M, x)
        DN = np.einsum('qpd,qd->qp', G, n)
        idx = slice(K * p, (K + 1) * p)
        Kmat[idx, idx] += -np.einsum('q,qi,qj->ij', w, np.conj(V), DN) \
                          - 1j * alpha * k * np.einsum('q,qi,qj->ij', w, np.conj(V), V)
        gd = hankel1(0, k * a) * np.ones(len(w), complex)
        f[idx] += -np.einsum('q,q,qi->i', w, gd, np.conj(DN) + alpha * 1j * k * np.conj(V))
    Q = ne * nq_arc
    Vg = np.zeros((Q, N), complex); Dg = np.zeros((Q, N), complex)
    wg = np.zeros(Q); thg = np.zeros(Q); q0 = 0
    for K in range(ne):
        t0, t1 = 2 * np.pi * K / ne, 2 * np.pi * (K + 1) / ne
        th, x, w = arc_quad(R, t0, t1, nq_arc)
        n = np.c_[np.cos(th), np.sin(th)]
        V, G = basis_vals(K, rel_angles, M, x)
        DN = np.einsum('qpd,qd->qp', G, n)
        slq = slice(q0, q0 + nq_arc); sli = slice(K * p, (K + 1) * p)
        Vg[slq, sli] = V; Dg[slq, sli] = DN; wg[slq] = w; thg[slq] = th
        q0 += nq_arc
    ms = np.arange(-NdtN, NdtN + 1)
    E = np.exp(-1j * np.outer(thg, ms))
    C = (E.T @ (wg[:, None] * Vg)) / (2 * np.pi)
    mult = k * h1vp(ms, k * R, 1) / hankel1(ms, k * R)
    TV = np.exp(1j * np.outer(thg, ms)) @ (mult[:, None] * C)
    Res = Dg - TV; W = wg[:, None]
    Kmat += Dg.conj().T @ (W * Vg) - Vg.conj().T @ (W * TV) \
            + delta / (1j * k) * (Res.conj().T @ (W * Res))
    return Kmat, f

def solve_space(rel_angles, M):
    p = len(rel_angles) + 2 * M + 1
    T = block_diag(*([local_gram(rel_angles, M)[0]] * ne))
    Kraw, fraw = assemble_raw(rel_angles, M)
    Kc = T.conj().T @ Kraw @ T; fc = T.conj().T @ fraw
    G = (Kc.conj().T - Kc) / (2j); G = (G + G.conj().T) / 2
    L = cholesky(G, lower=True, check_finite=False)
    B = solve_triangular(L.conj().T, np.eye(L.shape[0]), lower=False,
                         check_finite=False)
    Kb = B.conj().T @ Kc @ B
    sgr = svdvals(Kb)
    y = np.linalg.solve(Kb, B.conj().T @ fc)
    coef = T @ (B @ y)
    z, wz = leggauss(70); zt, wt = leggauss(70)
    num = den = 0.0
    for K in range(ne):
        rr = .5 * ((R - a) * z + (R + a)); wr = .5 * (R - a) * wz
        tA, tB = 2 * np.pi * K / ne, 2 * np.pi * (K + 1) / ne
        th = .5 * ((tB - tA) * zt + (tB + tA)); wth = .5 * (tB - tA) * wt
        RR, TT = np.meshgrid(rr, th, indexing='ij')
        W2 = (np.outer(wr, wth) * RR).ravel()
        x = np.c_[(RR * np.cos(TT)).ravel(), (RR * np.sin(TT)).ravel()]
        V, _ = basis_vals(K, rel_angles, M, x)
        uh = V @ coef[K * p:(K + 1) * p]
        ue = hankel1(0, k * np.hypot(x[:, 0], x[:, 1]))
        num += np.sum(W2 * np.abs(uh - ue) ** 2)
        den += np.sum(W2 * np.abs(ue) ** 2)
    return coef, np.sqrt(num / den), sgr[0] / sgr[-1]

# ----------------------------------------- modal estimation from a solution
def estimate_gamma(rel_angles, M, coef):
    """Modal coefficients a_m about CEN[0] from the Cauchy data of the
    numerical solution on the inscribed circle of element 0."""
    nq = 1024
    t = 2 * np.pi * np.arange(nq) / nq
    x = CEN[0] + h_in * np.c_[np.cos(t + cent_th[0]), np.sin(t + cent_th[0])]
    n = np.c_[np.cos(t + cent_th[0]), np.sin(t + cent_th[0])]
    p = len(rel_angles) + 2 * M + 1
    V, G = basis_vals(0, rel_angles, M, x)
    c0 = coef[:p]
    uv = V @ c0
    un = np.einsum('qpd,p,qd->q', G, c0, n)
    tau_in2 = 2 * np.pi * h_in * k * (jv(ms_est, k * h_in) ** 2
                                      + jvp(ms_est, k * h_in) ** 2)
    Ph = np.exp(1j * np.outer(t + cent_th[0] - cent_th[0], ms_est))
    phi_v = jv(ms_est, k * h_in) * Ph
    phi_n = k * jvp(ms_est, k * h_in) * Ph
    wq = 2 * np.pi / nq
    ip = wq * h_in * (k * (phi_v.conj().T @ uv) + (1 / k) * (phi_n.conj().T @ un))
    a_hat = ip / tau_in2
    return (1j) ** (-ms_est) * a_hat


def estimate_gamma_boundary(rel_angles, M, coef):
    """Modal coefficients a_m about CEN[0] by least squares against the
    Cauchy data of the numerical solution on the element boundary
    (two arcs and two radial edges), in the equilibrated trace metric."""
    t0, t1 = 0.0, 2 * np.pi / ne
    pts, wts, nrms = [], [], []
    for r_, sgn in ((a, -1.0), (R, 1.0)):
        th, x, w = arc_quad(r_, t0, t1, 128)
        pts.append(x); wts.append(w)
        nrms.append(sgn * np.c_[np.cos(th), np.sin(th)])
    for th_e, sgn in ((t0, -1.0), (t1, 1.0)):
        x0 = a * np.array([np.cos(th_e), np.sin(th_e)])
        x1 = R * np.array([np.cos(th_e), np.sin(th_e)])
        x, w = edge_quad_line(x0, x1, 100)
        n = sgn * np.array([-np.sin(th_e), np.cos(th_e)])
        pts.append(x); wts.append(w); nrms.append(np.tile(n, (len(w), 1)))
    x = np.vstack(pts); w = np.concatenate(wts); n = np.vstack(nrms)
    p = len(rel_angles) + 2 * M + 1
    V, G = basis_vals(0, rel_angles, M, x)
    c0 = coef[:p]
    uv = V @ c0
    un = np.einsum('qpd,p,qd->q', G, c0, n)
    y = x - CEN[0]; rho = np.hypot(y[:, 0], y[:, 1])
    phi = np.arctan2(y[:, 1], y[:, 0])
    phi_rel = phi - cent_th[0]           # modal phase relative to the
    E = np.exp(1j * np.outer(phi_rel, ms_est))  # center direction
    Vm = jv(ms_est[None, :], k * rho[:, None]) * E
    er = np.c_[np.cos(phi), np.sin(phi)]
    et = np.c_[-np.sin(phi), np.cos(phi)]
    nr = np.einsum('qd,qd->q', n, er)
    nt = np.einsum('qd,qd->q', n, et)
    DNm = (k * jvp(ms_est[None, :], k * rho[:, None], 1) * E) * nr[:, None]           + ((1j * ms_est[None, :] / rho[:, None]) * Vm) * nt[:, None]
    sw = np.sqrt(w)
    A = np.vstack([np.sqrt(k) * sw[:, None] * Vm,
                   (1 / np.sqrt(k)) * sw[:, None] * DNm])
    b = np.concatenate([np.sqrt(k) * sw * uv, (1 / np.sqrt(k)) * sw * un])
    # no column equilibration: the natural trace-norm decay of high modes
    # on the element boundary is what keeps unresolvable modes out of the
    # fit.  rcond truncates them; a hard trust window zeroes the rest.
    colnorm = np.linalg.norm(A, axis=0)
    a_hat = np.linalg.lstsq(A, b, rcond=1e-10)[0]
    a_hat[colnorm < 1e-8 * colnorm.max()] = 0.0
    return (1j) ** (-ms_est) * a_hat

# --------------------------------------------------------------- selector
def esprit_init(gamma, M, q, N=24):
    i0 = np.searchsorted(ms_est, M + 1)
    seq = gamma[i0:i0 + N]
    L = len(seq) // 2 + 1; Kc_ = len(seq) - L + 1
    H = np.empty((L, Kc_), complex)
    for r in range(L):
        H[r, :] = seq[r:r + Kc_]
    U, s, _ = svd(H, full_matrices=False)
    Uq = U[:, :q]
    S = np.linalg.pinv(Uq[:-1, :]) @ Uq[1:, :]
    z = np.linalg.eigvals(S)
    return np.sort((-np.angle(z) + np.pi) % (2 * np.pi) - np.pi)

def fit_candidate(gamma, p, q):
    M = (p - q - 1) // 2
    mask = np.abs(ms_est) > M
    mt = ms_est[mask]; wt = tau_sel[mask]; b = wt * gamma[mask]
    tn = np.linalg.norm(tau_sel * gamma)
    if q == 0:
        return M, np.array([]), np.linalg.norm(b) / tn
    th0 = esprit_init(gamma, M, q)
    def solve_c(th):
        A = wt[:, None] * np.exp(-1j * np.outer(mt, th))
        c = np.linalg.lstsq(A, b, rcond=1e-13)[0]
        return c, A @ c - b
    res = least_squares(lambda th: np.r_[solve_c(th)[1].real,
                                         solve_c(th)[1].imag],
                        th0, method='trf', max_nfev=400,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    th = (res.x + np.pi) % (2 * np.pi) - np.pi
    c, r = solve_c(th)
    scale = np.max(np.abs(c)) if len(c) else 0.0
    if scale > 0:
        keep = np.abs(c) > 1e-10 * scale
        th = th[keep]
        if keep.sum() != q and keep.sum():
            c, r = solve_c(th)
        elif not keep.sum():
            r = -b
    return M, th, np.linalg.norm(r) / tn

def select(gamma, p):
    admiss = []
    for q in Q_CAND:
        M, th, e = fit_candidate(gamma, p, q)
        T, rk = local_gram(th, M)
        if rk == len(th) + 2 * M + 1:
            admiss.append((e, th, M))
    e, th, M = min(admiss, key=lambda z: z[0])
    return th, M, e

# ------------------------------------------------------------------ driver
if __name__ == '__main__':
    rows = []
    P = 21
    ang0 = 2 * np.pi * np.arange(9) / 9
    coef0, err0, _ = solve_space(ang0, 0)
    print(f'coarse solve (9 PW + J0): E_L2 = {err0:.3e}', flush=True)
    rows.append(dict(stage='coarse solve', estimator='', qPW='', MFB=0,
                     E_L2=err0, tail_pred='', modal_data_err='',
                     angles_deg=''))
    for name, est in (('inscribed circle', estimate_gamma),
                      ('element boundary', estimate_gamma_boundary)):
        state = (ang0, 0, coef0)
        for cycle in (1, 2):
            gam = est(*state)
            tail_err = np.linalg.norm(tau_sel * (gam - gamma_exact)) \
                       / np.linalg.norm(tau_sel * gamma_exact)
            th, M, e = select(gam, P)
            coef, err, kgr = solve_space(th, M)
            ang = ';'.join(f'{d:.4f}' for d in np.rad2deg(np.sort(th)))
            print(f'{name}, cycle {cycle}: data err {tail_err:.2e}; '
                  f'selected qPW={len(th)}, M={M}, pred {e:.3e}; '
                  f'E_L2 = {err:.4e}, kGR = {kgr:.3f}, angles [{ang}]',
                  flush=True)
            rows.append(dict(stage=f'cycle {cycle}', estimator=name,
                             qPW=len(th), MFB=M, E_L2=err, tail_pred=e,
                             modal_data_err=tail_err, angles_deg=ang,
                             kappa_GR=kgr))
            state = (th, M, coef)
    th, M, e = select(gamma_exact, P)
    coef, err, kgr = solve_space(th, M)
    ang = ';'.join(f'{d:.4f}' for d in np.rad2deg(np.sort(th)))
    print(f'control (exact trace): qPW={len(th)}, M={M}, pred {e:.3e}; '
          f'E_L2 = {err:.4e}, kGR = {kgr:.3f}, angles [{ang}]', flush=True)
    rows.append(dict(stage='control', estimator='exact trace', qPW=len(th),
                     MFB=M, E_L2=err, tail_pred=e, modal_data_err=0.0,
                     angles_deg=ang, kappa_GR=kgr))
    pd.DataFrame(rows).to_csv(ROOT / 'data' / 'coarse_trace_selector.csv',
                              index=False)
    print('wrote data/coarse_trace_selector.csv')

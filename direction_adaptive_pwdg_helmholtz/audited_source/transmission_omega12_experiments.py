"""Reproduce the principal omega=12 transmission experiments in the manuscript.

The script records:
  * Part A analytic-Jacobian frequency continuation on N=2 and N=4 meshes;
  * Part B simultaneous coefficient-direction continuation on N=2;
  * the omega=12 uniform conditioning sweep.

The deliberately unsuccessful direct omega=12 starts are reported separately in
``data/transmission_omega12_direct_starts.csv``; the principal algorithms use
frequency continuation because the target-frequency landscape is nonconvex.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.linalg import cholesky

from exact import Transmission
from pwdg import PWDG
from direction_adaptive_automatic import l2_error, physical_skeleton_residual, graph_data
from wirtinger_pwdg_tests import (make_family_state, physical_skeleton_residual_vector,
                                  all_analytic, fams_trans)
from joint_ls_tests import (build_state, complex_residual_and_Kz, pack, unpack,
                            real_residual, real_jac)

OUT = Path(__file__).resolve().parent.parent / "generated"
OUT.mkdir(exist_ok=True)

STARTS_A = {
    69: np.array([[np.deg2rad(251.786), 0.0],
                  [np.deg2rad(103.766), 0.0],
                  [np.deg2rad(65.676), 0.0]]),
    29: np.array([[np.deg2rad(263.749), -0.6095],
                  [np.deg2rad(45.757), 0.2579],
                  [np.deg2rad(351.014), 1.2004]]),
}


def part_a_stage(N, exact, z0, nq):
    x0 = np.asarray(z0, float).ravel()
    fams = fams_trans(N)

    def unpack_z(x):
        z = np.asarray(x, float).reshape(-1, 2).copy()
        z[:, 0] %= 2 * np.pi
        return z

    lb, ub = [], []
    for j in range(len(z0)):
        lb += [x0[2*j] - 2*np.pi, -1.8]
        ub += [x0[2*j] + 2*np.pi, 1.8]

    def fun(x):
        z = unpack_z(x)
        s, _ = make_family_state(N, exact, z, fams, nq)
        return physical_skeleton_residual_vector(s)

    def jac(x):
        z = unpack_z(x)
        s, fb = make_family_state(N, exact, z, fams, nq)
        return all_analytic(s, fb, z)[1]

    rr = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf",
                       x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                       max_nfev=120)
    z = unpack_z(rr.x)
    s, _ = make_family_state(N, exact, z, fams, max(nq, 24))
    return z, s, rr


def run_part_a():
    stages, finals = [], []
    for N in (2, 4):
        for angle in (69, 29):
            z = STARTS_A[angle].copy()
            cumulative = 0
            for omega in (1, 2, 4, 6, 8, 10, 12):
                z, s, rr = part_a_stage(N, Transmission(omega, angle), z,
                                        16 if N == 2 else 14)
                cumulative += rr.nfev
                stages.append([N, angle, omega, rr.nfev, cumulative,
                               l2_error(s, 20), physical_skeleton_residual(s)**2,
                               *np.degrees(z[:, 0]) % 360, *z[:, 1]])
            gd = graph_data(s.A)
            finals.append([N, angle, s.ndof, cumulative, l2_error(s, 20),
                           physical_skeleton_residual(s)**2,
                           gd["cond_raw"], gd["cond_gr"],
                           *np.degrees(z[:, 0]) % 360, *z[:, 1]])
    pd.DataFrame(stages, columns=["N","angle","omega","stage_nfev","cum_nfev",
        "l2","J","theta1","theta2","theta3","eta1","eta2","eta3"]).to_csv(
        OUT / "transmission_omega12_analytic_continuation_stages.csv", index=False)
    pd.DataFrame(finals, columns=["N","angle","ndof","cum_nfev","l2","J",
        "cond_raw","cond_gr","theta1","theta2","theta3","eta1","eta2","eta3"]).to_csv(
        OUT / "transmission_omega12_analytic_final.csv", index=False)


def part_b_stage(exact, z0, c0=None, nq=12):
    s, D, b, _, _, m = build_state(2, exact, z0, "transmission", nq)
    if c0 is None:
        c0 = la.lstsq(D, b, lapack_driver="gelsy")[0]
    N = len(c0)
    x0 = pack(c0, z0)
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[-m:] = -1.8
    ub[-m:] = 1.8

    def fun(x):
        c, z = unpack(x, N, m)
        return real_residual(complex_residual_and_Kz(2, exact, z, c,
                                                      "transmission", nq)[3])

    def jac(x):
        c, z = unpack(x, N, m)
        _, D, _, _, Kz = complex_residual_and_Kz(2, exact, z, c,
                                                  "transmission", nq)
        return real_jac(D, Kz)

    rr = least_squares(fun, x0, jac=jac, bounds=(lb, ub), method="trf",
                       x_scale="jac", ftol=1e-13, xtol=1e-13, gtol=1e-13,
                       max_nfev=600)
    c, z = unpack(rr.x, N, m)
    s, D, b, r, Kz = complex_residual_and_Kz(2, exact, z, c,
                                              "transmission", max(nq, 20))
    s.u = c
    return rr, c, z, s, float(np.vdot(r, r).real)


def run_part_b():
    rows = []
    for angle in (69, 29):
        z = STARTS_A[angle][:, 0] + 1j * STARTS_A[angle][:, 1]
        c = None
        cumulative = 0
        for omega in (0.25, 0.5, 1, 2, 4, 6, 8, 10, 12):
            rr, c, z, s, J = part_b_stage(Transmission(omega, angle), z, c)
            cumulative += rr.nfev
            rows.append([angle, omega, rr.nfev, cumulative, J, l2_error(s, 18),
                         *np.degrees(z.real) % 360, *z.imag])
    pd.DataFrame(rows, columns=["angle","omega","stage_nfev","cum_nfev","J","l2",
        "theta1","theta2","theta3","eta1","eta2","eta3"]).to_csv(
        OUT / "joint_transmission_omega12_continuation.csv", index=False)


def run_conditioning():
    rows = []
    exact = Transmission(12, 69)
    for p in (3,5,7,9,11,13,15,17,19,21,23,25,27,29):
        solver = PWDG(4, exact, p=p, nq=30)
        A, _ = solver.assemble()
        A = A.toarray()
        G = (A - A.conj().T)/(2j)
        G = (G + G.conj().T)/2
        ev = np.linalg.eigvalsh(G)
        raw = np.linalg.cond(A)
        try:
            L = cholesky(G, lower=True, check_finite=False)
            B = np.linalg.solve(L.conj().T, np.eye(L.shape[0]))
            gr = np.linalg.cond(B.conj().T @ A @ B)
        except Exception:
            gr = np.nan
        rows.append([p, A.shape[0], raw, gr, ev[0]])
    pd.DataFrame(rows, columns=["p","Nh","kappa2_A","kappa_GR","min_eig_G"]).to_csv(
        OUT / "uniform_conditioning_graph_riesz_omega12.csv", index=False)


if __name__ == "__main__":
    run_part_a()
    run_part_b()
    run_conditioning()

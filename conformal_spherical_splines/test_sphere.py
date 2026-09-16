"""
test_sphere.py: verification of sphsplines on the round sphere.

T1 geometry: Euler relations of a closed triangulation, quadrature area 4pi
T2 reproduction: constant (even d) and the spherical harmonic xy interpolate
   exactly, satisfy the smoothness rows, and lie in ker K / solve the pencil
T3 eigenvalues: parity fingerprint (even-l clusters exact, odd-l converge
   from above) at d = 2, 4 (r = 0) and d = 6 (r = 1)
"""
import math
import numpy as np
import numpy.linalg as la

from sphsplines import (icosphere, meshsize_sphere, fast_meshdata,
                        smoothness_sphere, assemble_sphere, interp_sphere,
                        eval_on_triangles, sphere_pencil, laplace_eigs,
                        exact_sphere_spectrum, refquad, triquad_sphere)

np.set_printoptions(precision=3, suppress=False)
PASS = lambda ok: "PASS" if ok else "FAIL"


def t1_geometry():
    print("== T1: mesh and quadrature ==")
    for lev in [0, 1, 2]:
        v, t = icosphere(lev)
        e, b, te, tv, ev = fast_meshdata(t, v.shape[0])
        V, E, N = v.shape[0], e.shape[0], t.shape[0]
        euler = (N == 2 * V - 4) and (E == 3 * V - 6) and (b.size == 0)
        lam, w = refquad(16)
        area = 0.0
        for kk in range(N):
            _, _, W = triquad_sphere(v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]],
                                     lam, w)
            area += W.sum()
        err = abs(area - 4 * math.pi) / (4 * math.pi)
        print(f" level {lev}: V={V} E={E} N={N} |mesh|={meshsize_sphere(v,t):.3f}"
              f"  Euler {PASS(euler)}  area rel err {err:.2e} {PASS(err < 1e-12)}")


def t2_reproduction(d=4, lev=1, r=1):
    print(f"== T2: reproduction, d={d}, r={r}, level {lev} ==")
    v, t = icosphere(lev)
    m = (d + 1) * (d + 2) // 2

    Z, Kz, Mz, S, M, K = sphere_pencil(v, t, d, r, q=12)

    # constant function (d even): in B_d, hence a global spline
    c1 = interp_sphere(v, t, d, lambda p: np.ones(p.shape[1]))
    vals, _ = eval_on_triangles(v, t, d, c1)
    e_eval = np.abs(vals - 1.0).max()
    e_smooth = np.abs(S @ c1).max() if S.shape[0] else 0.0
    e_stiff = np.abs(Z.T @ (K @ c1)).max()
    mass1 = c1 @ (M @ c1)
    print(f" const: eval {e_eval:.2e} {PASS(e_eval<1e-12)}"
          f"  |S c| {e_smooth:.2e} {PASS(e_smooth<1e-10)}"
          f"  |K c| {e_stiff:.2e} {PASS(e_stiff<1e-10)}"
          f"  c'Mc-4pi {abs(mass1-4*math.pi):.2e}")

    # spherical harmonic Y in Y_2: f = xy, eigenvalue 6
    f = lambda p: p[0] * p[1]
    c2 = interp_sphere(v, t, d, f)
    vals, pts = eval_on_triangles(v, t, d, c2)
    e_eval = np.abs(vals - f(pts)).max()
    e_smooth = np.abs(S @ c2).max() if S.shape[0] else 0.0
    res = Z.T @ (K @ c2 - 6.0 * (M @ c2))
    e_pencil = np.abs(res).max() / np.abs(Z.T @ (M @ c2)).max()
    print(f" xy   : eval {e_eval:.2e} {PASS(e_eval<1e-12)}"
          f"  |S c| {e_smooth:.2e} {PASS(e_smooth<1e-10)}"
          f"  pencil resid {e_pencil:.2e} {PASS(e_pencil<1e-9)}")


def spectrum_table(d, r, levels, k, q=12):
    print(f"== T3: eigenvalues, d={d}, r={r} ==")
    exact = exact_sphere_spectrum(k)
    prev_err = None
    for lev in levels:
        v, t = icosphere(lev)
        h = meshsize_sphere(v, t)
        Z, Kz, Mz, S, M, K = sphere_pencil(v, t, d, r, q=q)
        mu = laplace_eigs(Kz, Mz, k=k)
        err = np.abs(mu - exact)
        # split by parity of l: index ranges of each cluster
        lmax = 0
        cnt = 0
        clusters = []
        while cnt < k:
            n = min(2 * lmax + 1, k - cnt)
            clusters.append((lmax, slice(cnt, cnt + n)))
            cnt += n
            lmax += 1
        even_max = max(err[s].max() for (l, s) in clusters if l % 2 == 0)
        odd = [(l, err[s].max()) for (l, s) in clusters if l % 2 == 1]
        odd_max = max(e for (_, e) in odd) if odd else 0.0
        one_sided = (mu - exact).min()
        line = (f" level {lev}: h={h:.3f} dim={Z.shape[1]}"
                f"  even-l max err {even_max:.2e}"
                f"  odd-l max err {odd_max:.2e}"
                f"  min(mu_h - mu) {one_sided:+.1e}")
        if prev_err is not None and odd_max > 0:
            rate = np.log(prev_err / odd_max) / np.log(prev_h / h)
            line += f"  odd rate {rate:.2f} (theory {2*d})"
        print(line)
        prev_err, prev_h = odd_max, h
        if lev == levels[-1]:
            print("   mu_h  :", np.array2string(mu[:min(k, 16)], precision=6))
            print("   exact :", np.array2string(exact[:min(k, 16)], precision=6))


if __name__ == "__main__":
    t1_geometry()
    t2_reproduction(d=4, lev=1, r=1)
    t2_reproduction(d=6, lev=1, r=1)
    spectrum_table(d=2, r=0, levels=[1, 2, 3], k=16)
    spectrum_table(d=4, r=0, levels=[1, 2], k=16)
    spectrum_table(d=6, r=1, levels=[1, 2], k=50)

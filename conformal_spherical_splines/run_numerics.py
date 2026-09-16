"""
run_numerics.py: produce all numerical results for the paper.

  A. round sphere: parity fingerprint and odd-cluster rates, d = 2, 4, 6
  B. spheroid (a=1, c=1.5): weight/area error check, SH reference with
     self-convergence estimate, spline spectra and errors vs reference
  C. eigenvectors for the Mayavi figures

Everything is saved to results_numerics.npz for the figure and table steps.
"""
import math
import numpy as np
from scipy.sparse.linalg import eigsh

from sphsplines import (icosphere, meshsize_sphere, sphere_pencil,
                        laplace_eigs, exact_sphere_spectrum)
from spheroid import SpheroidWeight, sh_reference

OUT = {}


def eigpairs(Kz, Mz, k):
    Ks = ((Kz + Kz.T) * 0.5).tocsc()
    Ms = ((Mz + Mz.T) * 0.5).tocsc()
    vals, vecs = eigsh(Ks, k=k, M=Ms, sigma=-1.0, which='LM')
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def cluster_errors(mu, exact, k):
    """max error over even-l and odd-l clusters of the round sphere."""
    err = np.abs(mu - exact)
    even, odd, cnt, l = 0.0, 0.0, 0, 0
    while cnt < k:
        n = min(2 * l + 1, k - cnt)
        e = err[cnt:cnt + n].max()
        if l % 2 == 0:
            even = max(even, e)
        else:
            odd = max(odd, e)
        cnt += n
        l += 1
    return even, odd


# ---------------------------------------------------------------- A: sphere
print("=== A: round sphere ===")
sphere_rows = []
cases = [(2, 0, [1, 2, 3], 16), (4, 0, [1, 2], 16), (6, 1, [1, 2], 50)]
sphere_eigvec = None
for (d, r, levels, k) in cases:
    exact = exact_sphere_spectrum(k)
    prev = None
    for lev in levels:
        v, t = icosphere(lev)
        h = meshsize_sphere(v, t)
        Z, Kz, Mz, S, M, K = sphere_pencil(v, t, d, r, q=12)
        if d == 6 and lev == 2:
            mu, y = eigpairs(Kz, Mz, k)
            sphere_eigvec = (v, t, d, (Z @ y[:, 44]))   # inside mu=42 cluster
        else:
            mu = laplace_eigs(Kz, Mz, k=k)
        even, odd = cluster_errors(mu, exact, k)
        onesided = float((mu - exact).min())
        rate = (math.log(prev[0] / odd) / math.log(prev[1] / h)
                if prev else float('nan'))
        sphere_rows.append(dict(d=d, r=r, lev=lev, h=h, dim=Z.shape[1],
                                even=even, odd=odd, rate=rate,
                                onesided=onesided))
        print(f" d={d} r={r} lev={lev} h={h:.3f} dim={Z.shape[1]} "
              f"even {even:.2e} odd {odd:.2e} rate {rate:.2f} "
              f"1-sided {onesided:+.1e}")
        prev = (odd, h)
OUT['sphere_rows'] = sphere_rows

# -------------------------------------------------------------- B: spheroid
print("=== B: spheroid a=1, c=1.5 ===")
sw = SpheroidWeight(1.0, 1.5)

# B1: area error check (validates the conformal weight independently)
from sphsplines import refquad, triquad_sphere
v, t = icosphere(2)
lam, wq = refquad(14)
area_q = 0.0
for kk in range(t.shape[0]):
    pts, sb, W = triquad_sphere(v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]],
                                lam, wq)
    area_q += (W * sw(pts)).sum()
A_exact = sw.area_exact()
area_err = abs(area_q - A_exact) / A_exact
print(f" area: quadrature {area_q:.12f} exact {A_exact:.12f} "
      f"rel err {area_err:.2e}")
OUT['area'] = dict(quad=area_q, exact=A_exact, err=area_err)

# B2: SH reference with self-convergence estimate
KREF = 50
ref60 = sh_reference(sw.w_of_Theta, L=60, nG=400, k=KREF)
ref50 = sh_reference(sw.w_of_Theta, L=50, nG=400, k=KREF)
ref_acc = np.abs(ref60 - ref50).max()
print(f" SH reference: first values {np.round(ref60[:8], 8)}")
print(f" SH self-convergence (L=50 vs 60), max diff: {ref_acc:.2e}")
mu1bar = ref60[1] * A_exact / (4 * math.pi)
print(f" normalized mu_1 * Area/(4 pi) = {mu1bar:.6f}  (Hersch bound 2)")
OUT['ref'] = ref60
OUT['ref_acc'] = ref_acc
OUT['mu1bar'] = mu1bar

# B3: spline spectra on the spheroid
spheroid_rows = []
spheroid_eigvec = None
cases = [(2, 0, [1, 2, 3], 16), (4, 0, [1, 2], 16), (6, 1, [1, 2], 30)]
for (d, r, levels, k) in cases:
    prev = None
    for lev in levels:
        v, t = icosphere(lev)
        h = meshsize_sphere(v, t)
        Z, Kz, Mz, S, M, K = sphere_pencil(v, t, d, r, q=12, weight=sw)
        if d == 6 and lev == 2:
            mu, y = eigpairs(Kz, Mz, k)
            spheroid_eigvec = (v, t, d, (Z @ y[:, 6]))   # 7th eigenfunction
        else:
            mu = laplace_eigs(Kz, Mz, k=k)
        err = np.abs(mu - ref60[:k]).max()
        onesided = float((mu - ref60[:k]).min())
        rate = (math.log(prev[0] / err) / math.log(prev[1] / h)
                if prev else float('nan'))
        spheroid_rows.append(dict(d=d, r=r, lev=lev, h=h, dim=Z.shape[1],
                                  err=err, rate=rate, onesided=onesided))
        print(f" d={d} r={r} lev={lev} h={h:.3f} dim={Z.shape[1]} "
              f"max err {err:.2e} rate {rate:.2f} 1-sided {onesided:+.1e}")
        prev = (err, h)
OUT['spheroid_rows'] = spheroid_rows

np.savez('results_numerics.npz',
         sphere_rows=np.array(sphere_rows, dtype=object),
         spheroid_rows=np.array(spheroid_rows, dtype=object),
         ref=ref60, ref_acc=ref_acc, mu1bar=mu1bar,
         area=np.array([area_q, A_exact, area_err]),
         sphere_vec_v=sphere_eigvec[0], sphere_vec_t=sphere_eigvec[1],
         sphere_vec_d=sphere_eigvec[2], sphere_vec_c=sphere_eigvec[3],
         spheroid_vec_v=spheroid_eigvec[0], spheroid_vec_t=spheroid_eigvec[1],
         spheroid_vec_d=spheroid_eigvec[2], spheroid_vec_c=spheroid_eigvec[3])
print("saved results_numerics.npz")

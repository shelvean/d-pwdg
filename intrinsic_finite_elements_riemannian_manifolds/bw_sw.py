"""Seifert-Weber coexact 1-form spectrum on the Bernstein-Whitney complex.
usage: python3 bw_sw.py r [level] [nev] [sigma]"""
import sys, time, numpy as np, scipy.sparse.linalg as sla
import hypforms
from sw_forms import seifert_weber
from bw_complex import BWComplex
r = int(sys.argv[1]); level = int(sys.argv[2]) if len(sys.argv) > 2 else 0
nev = int(sys.argv[3]) if len(sys.argv) > 3 else 12
sig = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
t0 = time.perf_counter()
cells, pairings, _ = seifert_weber(level)
X = BWComplex(cells, pairings, r, hypforms.Q, forms=(0, 1, 2, 3))
D0, D1, D2 = X.assemble_d(0), X.assemble_d(1), X.assemble_d(2)
t1 = time.perf_counter()
M1, M2 = X.assemble_mass(1), X.assemble_mass(2)
t2 = time.perf_counter()
dd1 = abs(D1 @ D0).max(); dd2 = abs(D2 @ D1).max()
A = (D1.T @ M2 @ D1).tocsc()
lam = np.sort(sla.eigsh(A, M=M1.tocsc(), k=nev, sigma=sig, which='LM', tol=1e-13,
                        return_eigenvectors=False))
t3 = time.perf_counter()
co = lam[lam > 1.0]
print(f'r={r} level={level} dims={tuple(X.dims[k] for k in range(4))} '
      f'|D1D0|={dd1:.1e} |D2D1|={dd2:.1e} complex {t1-t0:.0f}s masses {t2-t1:.0f}s solve {t3-t2:.0f}s')
print('coexact:', np.array2string(co, precision=11, max_line_width=170))
print(f'mean of lowest six {co[:6].mean():.12f}')
np.save(f'run/bw_r{r}_L{level}.npy', lam)

"""Seifert-Weber coexact spectrum with the vectorized local mass (fastmass.py).
usage: python3 sw_fast.py r level [equilibrate 0/1]"""
import os, sys, time
r, level = int(sys.argv[1]), int(sys.argv[2])
os.environ['RIEMANNFEM_EQUILIBRATE'] = sys.argv[3] if len(sys.argv) > 3 else '1'
import numpy as np, scipy.sparse.linalg as sla
from sw_forms import seifert_weber, SparseEntityComplex
from fastmass import local_hodge_mass_fast
from aad import local_hodge_mass_aad_cell
import hypforms
t0 = time.perf_counter()
cells, pairings, metrics = seifert_weber(level)
C = SparseEntityComplex(len(cells), r, pairings); q = r + 4
t1 = time.perf_counter()
loc = {k: [local_hodge_mass_aad_cell(C.spaces[k], V, hypforms.Q, q=q) for V in cells] for k in (1, 2)}
t2 = time.perf_counter()
M1 = C.assemble_mass(1, loc[1]); M2 = C.assemble_mass(2, loc[2])
A = (C.D[1].T @ M2 @ C.D[1]).tocsc()
lam = np.sort(sla.eigsh(A, M=M1.tocsc(), k=14, sigma=1.5, which='LM', tol=1e-12,
                        return_eigenvectors=False))
t3 = time.perf_counter()
co = lam[lam > 1.0]
print(f'r={r} level={level} dims={C.dimensions} complex {t1-t0:.0f}s  masses {t2-t1:.0f}s  solve {t3-t2:.0f}s')
print('coexact:', np.array2string(co, precision=11, max_line_width=160))
print(f'mean of lowest six {co[:6].mean():.11f}')
np.save(f'run/swfast_r{r}_L{level}.npy', lam)

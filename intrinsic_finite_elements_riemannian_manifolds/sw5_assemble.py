import glob, pickle, time, numpy as np, scipy.sparse as sp
from sw_forms import seifert_weber, SparseEntityComplex
r, level = 5, 1
t = time.perf_counter()
cells, pairings, metrics = seifert_weber(level)
C = SparseEntityComplex(len(cells), r, pairings)
files = sorted(glob.glob(f'run/swblk_r{r}_L{level}_*.pkl'), key=lambda f: int(f.split('_')[-2]))
loc = {1: [], 2: []}
for f in files:
    d = pickle.load(open(f, 'rb')); loc[1] += d[1]; loc[2] += d[2]
assert len(loc[1]) == len(cells)
M1 = C.assemble_mass(1, loc[1]).tocsc(); M2 = C.assemble_mass(2, loc[2])
A = (C.D[1].T @ M2 @ C.D[1]).tocsc()
sp.save_npz('run/sw5_A.npz', A); sp.save_npz('run/sw5_M1.npz', M1)
print('dims', C.dimensions, 'A nnz', A.nnz, 'M1 nnz', M1.nnz, round(time.perf_counter() - t), 's')

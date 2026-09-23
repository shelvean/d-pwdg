import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
A = sp.load_npz('run/sw5_A.npz'); M1 = sp.load_npz('run/sw5_M1.npz')
K = (A - 1.5 * M1).tocsc()
t = time.perf_counter()
lu = sla.splu(K, permc_spec=sys.argv[1], options=dict(SymmetricMode=True))
print(sys.argv[1], 'nnz L+U', lu.L.nnz + lu.U.nnz, 'MB', (lu.L.nnz + lu.U.nnz) * 12 / 1e6, round(time.perf_counter() - t), 's')

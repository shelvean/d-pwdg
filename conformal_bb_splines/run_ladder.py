"""Table 'tab:ladder': regular 4g-gons with opposite sides paired, g = 2..8,
on S^0_4 at levels 1 and 2 (level 2 fails at g = 8 by the absolute matching
tolerance, as stated in the paper)."""
import numpy as np, json, time
from scipy.sparse.linalg import eigsh
from hypsurf import assemble_n
rows = []
for g in range(2, 9):
    n = 4*g
    for level, d, nq in [(1, 4, 24), (2, 4, 16)]:
        t0 = time.time()
        try: out = assemble_n(n, level, d, nquad=nq)
        except Exception as e: print(g, level, 'fail', e); break
        M, K, area, nt, nd = out[:5]
        v = np.sort(eigsh(K, k=5, M=M, sigma=-1.0, which='LM', return_eigenvectors=False).real)
        rows.append(dict(g=g, n=n, level=level, d=d, nt=int(nt), nd=int(nd), area=float(area), area_err=float(abs(area-4*np.pi*(g-1))), mu0=float(v[0]), mu1=float(v[1]), mu2=float(v[2]), t=time.time()-t0))
        print(rows[-1], flush=True)
json.dump(rows, open('genus_ladder.json', 'w'), indent=1)

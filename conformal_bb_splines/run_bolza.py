"""Table 'tab:bolza' (Bolza convergence) and 'tab:bolzaspec' (first ten
distinct eigenvalues with multiplicities); also the odd-degree zero-mode
numbers quoted after Proposition 'exactfacts'."""
import numpy as np, json, hypmesh, hypbb
from scipy.sparse.linalg import eigsh
LAM1 = 3.8388872588421995185866224504354645970819150157
rows = []
for d in (2, 4, 6):
    for L in (1, 2, 3):
        nq = {1: 44, 2: 30, 3: 20}[L]
        M, K, area, nt, nd = hypmesh.assemble(L, d, nquad=nq)
        if nd > 20000: continue
        v = np.sort(eigsh(K, k=6, M=M, sigma=-1.0, which="LM", return_eigenvectors=False).real)
        rows.append(dict(d=d, L=L, nt=int(nt), nd=int(nd), areaerr=float(abs(area-4*np.pi)), l0=float(v[0]), l1=float(v[1]), rel=float(abs(v[1]-LAM1)/LAM1)))
        print(rows[-1], flush=True)
json.dump(rows, open('bolza_conv.json', 'w'), indent=1)
# spectrum with multiplicities at d = 6, level 2
M, K, area, nt, nd = hypmesh.assemble(2, 6, nquad=30)
vals = np.sort(eigsh(K, k=45, M=M, sigma=-1.0, which="LM", return_eigenvectors=False).real)
cl = []; cur = [vals[0]]
for v in vals[1:]:
    if abs(v-cur[-1]) < 2e-4*max(1, abs(v)): cur.append(v)
    else: cl.append(cur); cur = [v]
cl.append(cur)
json.dump([dict(mu=float(np.mean(c)), mult=len(c), spread=float(max(c)-min(c))) for c in cl], open('bolza_spectrum.json', 'w'), indent=1)
for c in cl: print(f"{np.mean(c):.9f} mult {len(c)} spread {max(c)-min(c):.1e}")
# odd degree: constants not in the space, zero mode polluted
orig = hypbb.constant_coeffs
hypmesh.constant_coeffs = lambda T, d: (orig(T, d) if d % 2 == 0 else np.zeros(len(hypbb.bern_indices(d))))
for d in (3, 4, 5, 6):
    M, K, area, nt, nd = hypmesh.assemble(1, d, nquad=30)
    v = np.sort(eigsh(K, k=4, M=M, sigma=-1.0, which='LM', return_eigenvectors=False).real)
    print(f"Bolza L=1 d={d}: mu0 = {v[0]:.3e}")

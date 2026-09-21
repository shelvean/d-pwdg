"""Step-3: Laplace-Beltrami eigenvalues of the Bolza surface, against Strohmaier-Uski."""

import numpy as np
from scipy.sparse.linalg import eigsh
from hypmesh import assemble, global_constant

# Strohmaier-Uski high-precision reference, multiplicity 3
LAM1 = 3.8388872588421995185866224504354645970819150157

print("=" * 78)
print("Bolza surface, Laplace-Beltrami spectrum on S^0_d, exact geodesic geometry")
print(f"reference lambda_1 = {LAM1:.16f}  (multiplicity 3)")
print("=" * 78)
print(f"{'d':>2} {'L':>2} {'ntri':>6} {'ndof':>7} {'area err':>10} "
      f"{'lam_0':>12} {'lam_1':>18} {'rel err':>10}")

results = {}
for d in (2, 4, 6):
    for L in (1, 2, 3):
        nq = {1: 44, 2: 30, 3: 20}[L]
        M, K, area, nt, nd = assemble(L, d, nquad=nq)
        if nd > 20000:
            continue
        vals = eigsh(K, k=8, M=M, sigma=-1.0, which="LM", return_eigenvectors=False)
        vals = np.sort(vals.real)
        l0, l1 = vals[0], vals[1]
        rel = abs(l1 - LAM1) / LAM1
        results.setdefault(d, []).append((L, nd, rel))
        print(f"{d:>2} {L:>2} {nt:>6} {nd:>7} {abs(area-4*np.pi):>10.2e} "
              f"{l0:>12.2e} {l1:>18.12f} {rel:>10.2e}")
    print()

print("=" * 78)
print("observed convergence rates in h (h halves per level)")
print("=" * 78)
for d, rows in results.items():
    for (L1, _, e1), (L2, _, e2) in zip(rows, rows[1:]):
        print(f"    d={d}  L {L1}->{L2}:  rate {np.log2(e1/e2):.2f}   (2d = {2*d})")

print()
print("=" * 78)
print("multiplicity of lambda_1 (expected 3)")
print("=" * 78)
M, K, area, nt, nd = assemble(3, 4, nquad=20)
vals = np.sort(eigsh(K, k=10, M=M, sigma=-1.0, which="LM",
                     return_eigenvectors=False).real)
for i, v in enumerate(vals):
    print(f"    lambda_{i} = {v:.12f}")

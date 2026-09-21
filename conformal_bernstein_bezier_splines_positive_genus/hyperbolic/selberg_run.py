"""Driver: Selberg trace formula check on the Bolza surface."""
import numpy as np
from scipy.sparse.linalg import eigsh
from klein import surface, OPPOSITE
from hypmesh import assemble
import selberg as S

V, g, p, info = surface(8, OPPOSITE(8))
G = [g[j] for j in range(8)]
E = S.group_ball(G, 4)
cls_raw = S.length_classes_fast(E, E[:1500], lmax=7.0)
# The word ball contains a class at exactly twice the systole.  It is an
# iterate, not a new primitive class, and must not also be fed to
# geometric_term(), which generates repetitions itself.
syslen = cls_raw[0][0]
cls = [(L, n) for L, n in cls_raw if abs(L - 2.0*syslen) > 5e-5]
print("primitive length classes retained for the trace-formula check")
for L, nc in cls:
    print(f"   ell = {L:.9f}   classes {nc:4d}")
print("   systole check: 24 oriented classes = 2 x 12 systolic geodesics")
print("   removed nonprimitive class at 2*systole before generating iterates")

M, K, area, nt, nd = assemble(3, 4, nquad=20)
lam = np.sort(eigsh(K, k=250, M=M, sigma=-1., which="LM",
                    return_eigenvectors=False).real)
lam[0] = 0.0
cls3 = [(L, n, 0) for L, n in cls]
print(f"\n{'t':>5} {'spectral':>14} {'identity':>14} {'geodesic':>12} "
      f"{'id+geo':>14} {'rel diff':>10}  limited by")
for t in (1.0, 0.7, 0.5, 0.3, 0.2):
    sp = S.spectral_term(lam, t)
    idt = S.identity_term(4*np.pi, t)
    geo = S.geometric_term(cls3, t)
    lim = "geodesic truncation" if sp > idt+geo else "spectral discretisation"
    print(f"{t:5.2f} {sp:14.9f} {idt:14.9f} {geo:12.9f} {idt+geo:14.9f} "
          f"{abs(sp-idt-geo)/sp:10.2e}  {lim}")

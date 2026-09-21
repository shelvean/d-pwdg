"""Tables 'tab:dtorus' (spline eigenvalues on the discocyte torus), the
closed-form assembly comparison quoted in the text, and 'tab:isocmp' (equal-
degree comparison with flat-facet and isoparametric surface FEM)."""
import numpy as np, json, time
import scipy.sparse.linalg as spla
from scipy.integrate import quad
from flatquot import Quotient
from flatexact import assemble_exact
from iso_fem import eigs
from dtorus import U, weight, rho, zz, t_of, speed, _ts
rows = []
for d, r, n in [(2, 0, 8), (2, 0, 16), (4, 0, 8), (4, 0, 16), (6, 1, 8), (6, 1, 12), (10, 1, 6)]:
    Q = Quotient('torus', n, n, d, r, a=U, b=2*np.pi); vals, vecs, Z = Q.eigen(k=9, w=weight, q=d+4)
    K, M, F, c1 = Q.assemble(w=weight, q=d+4); area = float(c1@(M@c1))
    rows.append(dict(d=d, r=r, n=n, dim=int(Z.shape[1]), area=area, eigs=[float(x) for x in vals[:9]]))
    print(f"spline ({d},{r}) n={n:2d} area {area:.10f} mu1 {vals[1]:.10f}", flush=True)
A = quad(lambda t: 2*np.pi*rho(t)*speed(t), 0, 2*np.pi, limit=200)[0]; lam = np.log(rho(_ts))
json.dump(dict(rows=rows, area_exact=A, U=U, osc=float(lam.max()-lam.min())), open('dtorus_table.json', 'w'), indent=1)
ex = []
for d, r, n in [(4, 0, 8), (4, 0, 16), (6, 1, 8), (6, 1, 12)]:
    Q = Quotient('torus', n, n, d, r, a=U, b=2*np.pi); K, M, F, c1 = assemble_exact(Q, w=weight, p=d+2); Z = Q.nullspace()
    v = np.sort(spla.eigsh((Z.T@K@Z).tocsc(), k=9, M=(Z.T@M@Z).tocsc(), sigma=-1.0, which='LM')[0])
    ex.append(dict(d=d, r=r, n=n, p=d+2, area=float(c1@(M@c1)), eigs=[float(x) for x in v]))
    print(f"closed-form ({d},{r}) n={n:2d} area {ex[-1]['area']:.10f} mu1 {v[1]:.10f}", flush=True)
json.dump(ex, open('dtorus_exact.json', 'w'), indent=1)
def demb(u, phi):
    t = t_of(u); return np.array([rho(t)*np.cos(phi), rho(t)*np.sin(phi), zz(t)])
REF = rows[-1]['eigs'][1]; sw = []
for k in [2, 4, 6]:
    for geom in [1, k]:
        for n, m in [(24, 12), (48, 24), (96, 48)]:
            if k == 6 and (n, m) == (96, 48): continue
            nv, area, v = eigs(demb, U, 2*np.pi, n, m, k, q=k+3, geom_k=geom)
            sw.append(dict(k=k, geom=geom, n=n, m=m, nv=nv, area=area, arel=float(abs(area-A)/A), mu1=float(v[1]), rel=float(abs(v[1]-REF)/REF)))
            print(f"P{k} geom{geom} {n}x{m} nv={nv} area rel {sw[-1]['arel']:.1e} mu1 rel {sw[-1]['rel']:.1e}", flush=True)
json.dump(sw, open('iso_sweep.json', 'w'), indent=1)

"""Eigenvalues of the distorted torus with the spline method, against the
Fourier reference of dtorus.py."""
import numpy as np, json, math
from flatquot import Quotient
from dtorus import U, weight, reference, rho, t_of, speed
from scipy.integrate import quad
ref = reference(nmodes=40)
ref2 = reference(nmodes=60)
print('reference self-conv', np.abs(ref[:10]-ref2[:10]).max())
rows=[]
for d, r in [(2,0),(4,0),(6,1)]:
    prev=None
    for n in [4, 8, 16]:
        if d == 6 and n == 16: continue
        Q = Quotient('torus', n, n, d, r, a=U, b=2*np.pi)
        vals, vecs, Z = Q.eigen(k=12, w=weight, q=d+4)
        # area check
        K, M, F, c1 = Q.assemble(w=weight, q=d+4)
        area = float(c1 @ (M @ c1))
        err = np.abs(vals[1:9]-ref2[1:9]).max()/ref2[8]
        p = float('nan') if prev is None else math.log(prev/err)/math.log(2)
        rows.append(dict(d=d,r=r,n=n,dim=int(Z.shape[1]),area=area,
                         eigs=[float(v) for v in vals[:9]],err=float(err),rate=p))
        print(f"({d},{r}) n={n:2d} dim={Z.shape[1]:6d} area {area:.9f} mu1 {vals[1]:.9f} mu3 {vals[3]:.9f} err {err:.2e} rate {p:.2f}", flush=True)
        prev=err
A=quad(lambda t: 2*np.pi*rho(t)*speed(t),0,2*np.pi,limit=200)[0]
json.dump(dict(ref=[float(v) for v in ref2], rows=rows, area_exact=A, U=U), open('dtorus.json','w'), indent=1)
print('exact area', A)

"""Poisson Test 1 for a prescribed conformal metric on the sphere with no
embedding: lambda(x) = a (x1^2 - x2^2) + b x3^2 (Hersch centering holds by
symmetry), w = exp(2 lambda). Same manufactured solution as Test 1."""
import json, os, time, numpy as np
from poisson_surface import manufactured, solve_poisson, errors
from sphsplines import icosphere, meshsize_sphere
a, b = 0.6, -0.4
def lam(p): return a * (p[0]**2 - p[1]**2) + b * p[2]**2
def wgt(p): return np.exp(2.0 * lam(p))
ut, grad_ut, lap_ut = manufactured('exp(x1) * x2 + x3**3 * x1')
out = 'poisson_prescribed.jsonl'; cache = {}
for d, r, levels in [(2, 0, [1, 2, 3, 4]), (4, 0, [1, 2, 3, 4]), (6, 1, [1, 2, 3, 4])]:
    for lev in levels:
        v, t = icosphere(lev); h = meshsize_sphere(v, t); t0 = time.time()
        c, info = solve_poisson(v, t, d, r, wgt, lap_ut, cache=cache)
        e0, e1 = errors(v, t, d, c, wgt, ut, grad_ut)
        row = dict(geo='prescribed', d=d, r=r, lev=lev, h=h, dim=info['dim'], e0=e0, e1=e1, resid=info['resid'])
        open(out, 'a').write(json.dumps(row) + '\n')
        print(d, r, lev, f"{e0:.3e} {e1:.3e} {time.time()-t0:.1f}s", flush=True)
# oscillation of lambda on a fine sample
th = np.linspace(0, np.pi, 400); ph = np.linspace(0, 2*np.pi, 800)
TH, PH = np.meshgrid(th, ph); P = np.array([np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)]).reshape(3, -1)
L = lam(P); print('osc lambda', 0.5*(L.max()-L.min())*2/2, L.max()-L.min())

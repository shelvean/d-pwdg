"""Poisson Test 1 on the flat and conformal torus and on the conformal Klein
bottle: manufactured u, f = -exp(-2 lambda) Delta u, rates in L2(M) and H1."""
import numpy as np, json, math, sys
from flatquot import Quotient
pi = np.pi
def run(kind, lam, u, grad_u, lap_u, degs=((2,0),(4,0),(6,1)), ns=(4,8,16,32), out=None):
    w = (lambda X: np.exp(2 * lam(X))) if lam is not None else None
    f = lambda X: -(np.exp(-2 * lam(X)) if lam is not None else 1.0) * lap_u(X)
    rows = []
    for d, r in degs:
        prev = None
        for n in ns:
            if d == 6 and n == 32: continue
            Q = Quotient(kind, n, n, d, r)
            c, nu = Q.poisson(f, w=w, q=d + 4)
            e0, e1 = Q.errors(c, u, grad_u, w=w)
            h = 1.0 / n
            p0 = p1 = float('nan')
            if prev: p0 = math.log(prev[1] / e0) / math.log(prev[0] / h); p1 = math.log(prev[2] / e1) / math.log(prev[0] / h)
            rows.append(dict(kind=kind, d=d, r=r, n=n, dim=int(Q.nullspace().shape[1]), e0=e0, e1=e1, p0=p0, p1=p1, nu=abs(nu)))
            print(f"{kind:6s} ({d},{r}) n={n:2d} dim={rows[-1]['dim']:6d} L2 {e0:.3e} ({p0:.2f})  H1 {e1:.3e} ({p1:.2f})  nu {abs(nu):.1e}", flush=True)
            prev = (h, e0, e1)
    if out: json.dump(rows, open(out, 'w'), indent=1)
    return rows

# torus: u periodic, not a polynomial
u  = lambda X: np.sin(2*pi*X[0]) * np.cos(4*pi*X[1]) + 0.5*np.cos(2*pi*(X[0]+X[1]))
gu = lambda X: np.array([2*pi*np.cos(2*pi*X[0])*np.cos(4*pi*X[1]) - pi*np.sin(2*pi*(X[0]+X[1])),
                         -4*pi*np.sin(2*pi*X[0])*np.sin(4*pi*X[1]) - pi*np.sin(2*pi*(X[0]+X[1]))])
lu = lambda X: -(4*pi**2 + 16*pi**2)*np.sin(2*pi*X[0])*np.cos(4*pi*X[1]) - 0.5*8*pi**2*np.cos(2*pi*(X[0]+X[1]))
lam_t = lambda X: 0.3*np.cos(2*pi*X[0]) + 0.2*np.sin(2*pi*X[1])   # periodic conformal factor
# Klein bottle: glide-invariant u and lambda (u(1-x, y+1) = u(x, y))
uk  = lambda X: np.sin(2*pi*X[0])*np.sin(pi*X[1]) + np.cos(2*pi*X[0])*np.cos(2*pi*X[1])
guk = lambda X: np.array([2*pi*np.cos(2*pi*X[0])*np.sin(pi*X[1]) - 2*pi*np.sin(2*pi*X[0])*np.cos(2*pi*X[1]),
                          pi*np.sin(2*pi*X[0])*np.cos(pi*X[1]) - 2*pi*np.cos(2*pi*X[0])*np.sin(2*pi*X[1])])
luk = lambda X: -(4*pi**2+pi**2)*np.sin(2*pi*X[0])*np.sin(pi*X[1]) - 8*pi**2*np.cos(2*pi*X[0])*np.cos(2*pi*X[1])
lam_k = lambda X: 0.3*np.cos(2*pi*X[0])*np.cos(2*pi*X[1])
if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'torus'): run('torus', None, u, gu, lu, out='poisson_torus_flat.json')
    if which in ('all', 'ctorus'): run('torus', lam_t, u, gu, lu, out='poisson_torus_conf.json')
    if which in ('all', 'klein'): run('klein', lam_k, uk, guk, luk, out='poisson_klein_conf.json')

"""Laplace-Beltrami eigenvalues on the Mobius band [0,1]^2 / (x,y)~(x+1,1-y),
free (Neumann) sides, flat metric and conformal metric e^{2 lambda} with a
glide-invariant lambda. Reference: Fourier-Neumann Galerkin on the double
cover [0,2]x[0,1] restricted to the glide-invariant subspace (m+n even)."""
import numpy as np, json, math, sys
from flatquot import Quotient
pi = np.pi
lam = lambda X: 0.3*np.cos(2*pi*X[0])*np.cos(2*pi*(X[1]-0.5))   # invariant under (x,y)->(x+1,1-y)
w = lambda X: np.exp(2*lam(X))

def reference(w, M=20, Nn=20, q=80, k=16):
    # real basis on the double cover: cos(pi m x) cos(n pi y), sin(pi m x) cos(n pi y), m>=0 (sin: m>=1), n>=0, m+n even
    xg, xw = np.polynomial.legendre.leggauss(q); x = (xg+1); xw = xw   # [0,2]
    yg, yw = np.polynomial.legendre.leggauss(q); y = 0.5*(yg+1); yw = 0.5*yw  # [0,1]
    X, Y = np.meshgrid(x, y, indexing='ij'); Wq = np.outer(xw, yw)
    W = w(np.array([X.ravel() % 1.0, Y.ravel()])).reshape(X.shape)   # w is 1-periodic in x on the cover? lambda has period 1 in x: yes
    basis = []; lam_diag = []
    for m in range(0, M+1):
        for n in range(0, Nn+1):
            if (m+n) % 2: continue
            cy = np.cos(n*pi*Y)
            basis.append(np.cos(pi*m*X)*cy); lam_diag.append(pi**2*(m*m+n*n))
            if m > 0: basis.append(np.sin(pi*m*X)*cy); lam_diag.append(pi**2*(m*m+n*n))
    B = np.array([b.ravel() for b in basis])            # (nb, nq)
    Mw = (B * (Wq.ravel()*W.ravel())) @ B.T
    M0 = (B * Wq.ravel()) @ B.T
    K = np.diag(lam_diag) @ M0                          # stiffness = eigenvalue times flat mass (basis orthogonal)
    K = 0.5*(K+K.T)
    from scipy.linalg import eigh
    vals = eigh(K, Mw, eigvals_only=True)
    return np.sort(vals)[:k]

if __name__ == '__main__':
    ref = reference(w, M=24, Nn=24)
    ref2 = reference(w, M=30, Nn=30)
    print('reference (M=24):', np.round(ref[:10], 8)); print('self-convergence', np.abs(ref[:10]-ref2[:10]).max())
    rows = []
    for d, r in [(2,0),(4,0),(6,1)]:
        prev = None
        for n in [4, 8, 16, 32]:
            if d == 6 and n == 32: continue
            Q = Quotient('mobius', n, n, d, r)
            vals, _, Z = Q.eigen(k=12, w=w, q=d+4)
            err = np.abs(vals[1:9] - ref2[1:9]).max() / np.abs(ref2[8])
            p = float('nan') if prev is None else math.log(prev/err)/math.log(2)
            rows.append(dict(d=d, r=r, n=n, dim=int(Z.shape[1]), eigs=[float(v) for v in vals[:9]], err=float(err), rate=p))
            print(f"({d},{r}) n={n:2d} dim={Z.shape[1]:6d} mu1..3 {vals[1]:.6f} {vals[2]:.6f} {vals[3]:.6f} maxrelerr(1..8) {err:.2e} rate {p:.2f}", flush=True)
            prev = err
    json.dump(dict(ref=[float(v) for v in ref2], rows=rows), open('mobius_eig.json','w'), indent=1)

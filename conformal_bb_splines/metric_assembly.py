"""Spline assembly on the flat torus with a general metric tensor given at the
quadrature points (orthogonal or not): the stiffness form int sqrt(g) g^{ij}
d_i u d_j v and the mass form int sqrt(g) u v. Reduces to flatquot.assemble
with w = sqrt(g) when the metric is conformal."""
import numpy as np, scipy.sparse as sp
from flatquot import tri_quadrature, bernstein, bernstein_grad

def assemble_metric(Q, metric, q=None):
    d = Q.d; q = q or (d+4); P, W = tri_quadrature(q)
    rows=[]; cols=[]; kv=[]; mv=[]
    for t, T in enumerate(Q.tris):
        V = Q.V[list(T)]; A = np.vstack([V.T, np.ones(3)]); gb = np.linalg.inv(A)[:, :2]
        area = 0.5*abs(np.linalg.det(np.array([V[1]-V[0], V[2]-V[0]])))
        b = np.vstack([1-P[:, 0]-P[:, 1], P[:, 0], P[:, 1]]); X = V[0][:, None]*b[0]+V[1][:, None]*b[1]+V[2][:, None]*b[2]
        B = bernstein(d, b); G = bernstein_grad(d, b, gb)             # (N,2,nq)
        E, F, Gm = np.array([metric(x, y) for x, y in X.T]).T
        det = E*Gm-F*F; sq = np.sqrt(det)
        ginv = np.array([[Gm/det, -F/det], [-F/det, E/det]])           # (2,2,nq)
        wq = W*2*area
        Kt = np.einsum('iax,abx,jbx,x->ij', G, ginv, G, wq*sq); Mt = np.einsum('ix,jx,x->ij', B, B, wq*sq)
        sl = np.arange(t*Q.Nloc, (t+1)*Q.Nloc); ii, jj = np.meshgrid(sl, sl, indexing='ij')
        rows += list(ii.ravel()); cols += list(jj.ravel()); kv += list(Kt.ravel()); mv += list(Mt.ravel())
    K = sp.csr_matrix((kv, (rows, cols)), shape=(Q.N, Q.N)); M = sp.csr_matrix((mv, (rows, cols)), shape=(Q.N, Q.N))
    return K, M

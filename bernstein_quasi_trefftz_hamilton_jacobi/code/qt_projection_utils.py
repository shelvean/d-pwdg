from __future__ import annotations
import time, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from eikonal_c0_pminus1 import triangle_quadrature, bernstein_values

def project_unknown(model, value_fun):
    """L2 projection of a scalar target into the model's assembled C0 Bernstein space.

    value_fun(points) returns the quantity represented by the model coefficients
    (u itself for unfactored HJ, tau=u-T for additive factorization, psi for
    multiplicative factorization). Prescribed model.fixed values are imposed exactly.
    """
    p=model.p; ref,wref=triangle_quadrature(max(12,2*p+6))
    lam=np.column_stack([1-ref[:,0]-ref[:,1],ref[:,0],ref[:,1]])
    B=bernstein_values(p,lam); nd=len(model.asm.keys)
    rows=[]; cols=[]; vals=[]; rhs=np.zeros(nd)
    for K,tri in enumerate(model.tris):
        V=model.verts[tri]; J=np.column_stack((V[1]-V[0],V[2]-V[0])); det=abs(np.linalg.det(J))
        pts=V[0]+ref@J.T; W=wref*det; target=value_fun(pts)
        M=B.T@(W[:,None]*B); b=B.T@(W*target); g=model.asm.l2g[K]
        rows.append(np.repeat(g,len(g))); cols.append(np.tile(g,len(g))); vals.append(M.ravel()); np.add.at(rhs,g,b)
    A=sp.coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(nd,nd)).tocsr()
    c=np.zeros(nd); c[model.fixed]=model.fixed_vals; free=model.free
    b=rhs[free]-A[free][:,model.fixed]@c[model.fixed]
    t=time.perf_counter(); c[free]=spsolve(A[free][:,free],b)
    return c,time.perf_counter()-t

"""Isoparametric surface finite elements of degree k = 1,2,3 on a surface given
by an exact parametrization of a periodic rectangle. Lagrange nodes are placed
on the exact surface, the geometry is the degree-k interpolant of those nodes,
and the tangential stiffness and mass matrices are assembled by quadrature on
the reference triangle. Used on the tube torus and on the distorted torus."""
import numpy as np, itertools
import scipy.sparse as sp, scipy.sparse.linalg as spla

def lagrange_basis(k):
    """nodes (barycentric i,j) and the monomial-Vandermonde basis on the
    reference triangle {(x,y): x,y>=0, x+y<=1}"""
    nodes = [(i/k, j/k) for i in range(k+1) for j in range(k+1-i)]
    mons  = [(a, b) for d in range(k+1) for a in range(d+1) for b in [d-a]]
    V = np.array([[x**a*y**b for a, b in mons] for x, y in nodes])
    C = np.linalg.inv(V)                       # columns: coefficients of each basis fn
    def N(x, y):
        m = np.array([x**a*y**b for a, b in mons]); return C.T @ m
    def dN(x, y):
        dx = np.array([a*x**(a-1)*y**b if a > 0 else 0.0 for a, b in mons])
        dy = np.array([b*x**a*y**(b-1) if b > 0 else 0.0 for a, b in mons])
        return np.vstack([C.T @ dx, C.T @ dy])          # (2, nloc)
    return np.array(nodes), N, dN

def tri_quad(order):
    x, w = np.polynomial.legendre.leggauss(order)
    x = 0.5*(x+1); w = 0.5*w
    P, W = [], []
    for i in range(order):
        for j in range(order):
            P.append((x[i], x[j]*(1-x[i]))); W.append(w[i]*w[j]*(1-x[i]))
    return np.array(P), np.array(W)

def assemble(emb, A, B, n, m, k, q=None, geom_k=None):
    """emb(u,v) -> R^3 on the periodic rectangle [0,A]x[0,B]"""
    q = q or (k+2)
    geom_k = geom_k or k                 # geometry degree: k (isoparametric) or 1 (flat facets)
    nodes, N, dN = lagrange_basis(k)
    gnodes, gN, gdN = lagrange_basis(geom_k)
    gdNq = np.array([gdN(*p) for p in QP]) if False else None
    QP, QW = tri_quad(q)
    Nq = np.array([N(*p) for p in QP])              # (nq, nloc)
    dNq = np.array([dN(*p) for p in QP])            # (nq, 2, nloc)
    gdNq = np.array([gdN(*p) for p in QP])          # (nq, 2, ngeom)
    gid = {}; XYZ = []
    def node_id(u, v):
        key = (round(u % A, 9), round(v % B, 9))
        if key not in gid:
            gid[key] = len(XYZ); XYZ.append(emb(key[0], key[1]))
        return gid[key]
    I=[]; J=[]; Kv=[]; Mv=[]
    hu, hv = A/n, B/m
    for a in range(n):
        for b in range(m):
            c00 = (a*hu, b*hv); c10 = ((a+1)*hu, b*hv)
            c01 = (a*hu, (b+1)*hv); c11 = ((a+1)*hu, (b+1)*hv)
            for tri in [(c00, c10, c11), (c00, c11, c01)]:
                p0, p1, p2 = [np.array(t) for t in tri]
                loc = []
                for (xi, eta) in nodes:
                    p = p0 + xi*(p1-p0) + eta*(p2-p0)
                    loc.append(node_id(p[0], p[1]))
                Xl = np.array([XYZ[i] for i in loc])          # (nloc,3)
                # geometry nodes (degree geom_k) on the surface
                Xg = np.array([emb(*(p0 + xi*(p1-p0) + eta*(p2-p0))) for (xi, eta) in gnodes])
                Ke = np.zeros((len(loc), len(loc))); Me = np.zeros_like(Ke)
                for iq in range(len(QP)):
                    t = gdNq[iq] @ Xg                         # (2,3) tangents of the geometry
                    g = t @ t.T
                    det = np.linalg.det(g)
                    if det <= 0: continue
                    ginv = np.linalg.inv(g); jac = np.sqrt(det)*QW[iq]
                    G = dNq[iq]                               # (2,nloc)
                    Ke += jac*(G.T @ ginv @ G)
                    Me += jac*np.outer(Nq[iq], Nq[iq])
                for ii in range(len(loc)):
                    for jj in range(len(loc)):
                        I.append(loc[ii]); J.append(loc[jj]); Kv.append(Ke[ii,jj]); Mv.append(Me[ii,jj])
    nv = len(XYZ)
    K = sp.csr_matrix((Kv,(I,J)), shape=(nv,nv)); M = sp.csr_matrix((Mv,(I,J)), shape=(nv,nv))
    return K, M, nv

def eigs(emb, A, B, n, m, k, nev=8, q=None, geom_k=None):
    K, M, nv = assemble(emb, A, B, n, m, k, q, geom_k)
    area = M.sum()
    vals = np.sort(spla.eigsh(K.tocsc(), k=nev, M=M.tocsc(), sigma=-1e-6, which='LM',
                              return_eigenvectors=False).real)
    return nv, float(area), vals

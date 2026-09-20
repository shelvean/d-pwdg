"""
metriccompare.py -- one degree-d scalar assembler in reference coordinates, with the metric
supplied in three ways, so that the geometry is the only thing that changes between runs.

    mode "model"  : the pullback of the model metric at the quadrature points, so the cell
                    geometry carries no approximation error.
    mode "regge"  : the constant metric determined by the six geodesic edge lengths of the
                    cell, which is the piecewise-flat metric of the same cell data,
                        G_kl = (l_{0k}^2 + l_{0l}^2 - l_{kl}^2)/2 .
    mode "tensor" : the model metric composed with a symmetric positive field A that is
                    invariant under the identifications, giving a metric other than the model
                    one, with the stiffness weight A^{-1} and the volume weight (det A)^{1/2}.

The polynomial space, the mesh, the identifications and the quadrature are the same in every
mode, so a difference between two runs is a difference of geometry alone.
"""
import sys, time, itertools, numpy as np
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from sphereforms import multi_indices, duffy3, bern
import sphereforms as sf

EPS = 1e-9

def geodesic_lengths(V, kappa):
    Q = np.diag([float(kappa), 1.0, 1.0, 1.0])
    L = np.zeros((4, 4))
    for i, j in itertools.combinations(range(4), 2):
        c = V[:, i] @ Q @ V[:, j]
        L[i, j] = L[j, i] = np.arccos(np.clip(c, -1, 1)) if kappa > 0 else np.arccosh(max(1.0, -c))
    return L

def metric_at(V, bq, kappa, mode, A=None):
    Q = np.diag([float(kappa), 1.0, 1.0, 1.0])
    if mode == "regge":
        L = geodesic_lengths(V, kappa); G = np.zeros((3, 3))
        for k in range(3):
            for l in range(3):
                G[k, l] = (L[0, k+1]**2 + L[0, l+1]**2 - L[k+1, l+1]**2)/2
        return np.broadcast_to(G, (len(bq), 3, 3)).copy()
    y = bq @ V.T; rho = np.sqrt(kappa*np.einsum("qi,ij,qj->q", y, Q, y))
    dy = (V[:, 1:] - V[:, [0]]).T
    dr = kappa*np.einsum("ki,ij,qj->qk", dy, Q, y)/rho[:, None]
    dx = (dy[None] - dr[:, :, None]*(y/rho[:, None])[:, None, :])/rho[:, None, None]
    G = np.einsum("qki,ij,qlj->qkl", dx, Q, dx)
    if mode == "tensor":
        x = y/rho[:, None]
        for q in range(len(bq)):
            M = A(x[q])                       # symmetric positive, in the model tangent frame
            G[q] = G[q] @ M
    return G

def build(cells, group, kappa, d, mode="model", A=None, nq=None, neig=20):
    MI = multi_indices(d); nloc = len(MI)
    pts = np.concatenate([(V@(MI.T/d)).T for V in cells])
    pts = pts/np.sqrt(np.abs(kappa*np.einsum("qi,ij,qj->q", pts,
          np.diag([float(kappa), 1.0, 1.0, 1.0]), pts)))[:, None]
    parent = np.arange(len(pts))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    tree = cKDTree(pts)
    for g in group:
        for i, nb in enumerate(tree.query_ball_point(pts@g.T, EPS)):
            for j in nb:
                a, b = find(i), find(int(j))
                if a != b: parent[b] = a
    _, gid = np.unique([find(i) for i in range(len(pts))], return_inverse=True); ndof = gid.max()+1
    nq = nq or d+4; bq, wq = duffy3(nq); val, grad = bern(d, MI, bq)
    DLm = np.array([[-1., -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    gref = np.einsum("qai,ik->qak", grad, DLm)    # derivatives with respect to (r,s,t)
    I, J, MV, KV = [], [], [], []; vol = 0.0
    gidc = gid.reshape(len(cells), nloc)
    for c, V in enumerate(cells):
        G = metric_at(np.asarray(V), bq, kappa, mode, A)
        Gi = np.linalg.inv(G); mu = wq*np.sqrt(np.abs(np.linalg.det(G))); vol += mu.sum()
        Me = val.T@(mu[:, None]*val)
        Ke = np.einsum("qai,qij,q,qbj->ab", gref, Gi, mu, gref)
        g = gidc[c]
        I.append(np.repeat(g, nloc)); J.append(np.tile(g, nloc))
        MV.append(Me.ravel()); KV.append(Ke.ravel())
    I, J = np.concatenate(I), np.concatenate(J)
    M = sp.coo_matrix((np.concatenate(MV), (I, J)), shape=(ndof, ndof)).tocsc()
    K = sp.coo_matrix((np.concatenate(KV), (I, J)), shape=(ndof, ndof)).tocsc()
    lam = np.sort(spla.eigsh(K, M=M, k=min(neig, ndof-2), sigma=-1.0, which="LM",
                             ncv=4*min(neig, ndof-2), return_eigenvectors=False))
    return dict(ndof=ndof, vol=float(vol), lam=lam)

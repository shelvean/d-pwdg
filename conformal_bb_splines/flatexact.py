"""Closed-form assembly on the flat quotients: stiffness and mass matrices by
the Bernstein product and integration formulas, with the conformal factor
(and the product w f for the load) replaced by their C^0 Bernstein
interpolants of degree p on the same mesh. No quadrature anywhere; the only
approximation is w -> w_hat, which is the setting of the weight-perturbation
proposition with delta = ||w_hat/w - 1||_inf = O(h^{p+1})."""
import numpy as np, scipy.sparse as sp
from math import comb, factorial
from flatquot import multi_indices, bernstein

def mcomb(a, b):
    """product over components of binom(a_i, b_i)"""
    return comb(a[0], b[0]) * comb(a[1], b[1]) * comb(a[2], b[2])

def interpolant(Q, func, p):
    """degree-p Bernstein coefficients of the C^0 interpolant of func on each
    triangle (broken numbering): returns (T, N_p) array"""
    idx = multi_indices(p); Np = len(idx)
    # collocation matrix at domain points is the same for every triangle
    Bc = np.array([[ (factorial(p)/(factorial(i)*factorial(j)*factorial(k)))
                     * (a/p)**i * (b/p)**j * (c/p)**k for (i,j,k) in idx] for (a,b,c) in idx])
    Binv = np.linalg.inv(Bc)
    C = np.zeros((len(Q.tris), Np))
    for t, T in enumerate(Q.tris):
        P = Q.V[list(T)]
        X = np.array([(i*P[0] + j*P[1] + k*P[2])/p for (i,j,k) in idx]).T
        C[t] = Binv @ func(X)
    return C

def assemble_exact(Q, w=None, f=None, p=None):
    d = Q.d; p = p if p is not None else d + 2
    idx = multi_indices(d); Nl = len(idx); idxp = multi_indices(p)
    W = interpolant(Q, w, p) if w is not None else None
    Fw = interpolant(Q, (lambda X: w(X)*f(X)) if w is not None else f, p) if f is not None else None
    # precompute combinatorial tensors
    cM = np.zeros((Nl, Nl, len(idxp)))          # int B_a B_b B_g / |T|
    cM0 = np.zeros((Nl, Nl))                     # int B_a B_b / |T|
    for a_, a in enumerate(idx):
        for b_, b in enumerate(idx):
            ab = tuple(a[i]+b[i] for i in range(3))
            cM0[a_, b_] = mcomb(ab, a) / comb(2*d, d) / comb(2*d+2, 2)
            base = mcomb(ab, a) / comb(2*d, d) / comb(2*d+p, p) / comb(2*d+p+2, 2)
            for g_, g in enumerate(idxp):
                abg = tuple(ab[i]+g[i] for i in range(3))
                cM[a_, b_, g_] = base * mcomb(abg, g)
    # stiffness combinatorics: int B^{d-1}_{a-e_i} B^{d-1}_{b-e_j} / |T|
    idxm = multi_indices(d-1); posm = {m: k for k, m in enumerate(idxm)}
    cK = np.zeros((Nl, Nl, 3, 3))
    for a_, a in enumerate(idx):
        for i in range(3):
            if a[i] == 0: continue
            am = tuple(a[k]-(k==i) for k in range(3))
            for b_, b in enumerate(idx):
                for j in range(3):
                    if b[j] == 0: continue
                    bm = tuple(b[k]-(k==j) for k in range(3))
                    s = tuple(am[k]+bm[k] for k in range(3))
                    cK[a_, b_, i, j] = mcomb(s, am) / comb(2*d-2, d-1) / comb(2*d, 2)
    rows=[]; cols=[]; kv=[]; mv=[]; Fv = np.zeros(Q.N)
    for t, T in enumerate(Q.tris):
        P = Q.V[list(T)]
        A = np.vstack([P.T, np.ones(3)]); gb = np.linalg.inv(A)[:, :2]   # grad b_i
        G = gb @ gb.T
        area = 0.5*abs(np.linalg.det(np.array([P[1]-P[0], P[2]-P[0]])))
        Kt = d*d*area*np.einsum('abij,ij->ab', cK, G)
        Mt = area*(cM @ W[t] if W is not None else cM0)
        sl = np.arange(t*Nl, (t+1)*Nl)
        ii, jj = np.meshgrid(sl, sl, indexing='ij')
        rows += list(ii.ravel()); cols += list(jj.ravel()); kv += list(Kt.ravel()); mv += list(Mt.ravel())
        if Fw is not None:
            # int B_a * (wf)_hat = sum_g c_g int B^d_a B^p_g
            cF = np.array([[mcomb(tuple(a[i]+g[i] for i in range(3)), a)/comb(d+p, d)/comb(d+p+2, 2)
                            for g in idxp] for a in idx])
            Fv[sl] = area*(cF @ Fw[t])
    K = sp.csr_matrix((kv, (rows, cols)), shape=(Q.N, Q.N)); M = sp.csr_matrix((mv, (rows, cols)), shape=(Q.N, Q.N))
    return K, M, Fv, np.ones(Q.N)

"""
sphereforms.py  --  homogeneous Bernstein-Bezier C^0 splines on spherical
space forms S^3/Gamma, Gamma a subgroup of the binary icosahedral group 2I
acting by left quaternion multiplication on the geodesic 600-cell.

Space: on a geodesic tetrahedron with vertex matrix V (columns in R^4),
b = V^{-1} x, basis B_alpha(b) = d!/alpha! b^alpha, |alpha| = d (homogeneous).
Quotient: B-coefficients are constant on Gamma-orbits of domain points.
Forms (kappa = +1):  m = int F_a F_b,  k = int grad F_a . grad F_b - d^2 F_a F_b,
chart y = V b (sum b = 1), x = y/|y|, dvol = |det V| |y|^{-4} db.
"""
import itertools, math, sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull, cKDTree

PHI = (1 + 5**0.5)/2

def qmul(a, b):
    a0,a1,a2,a3 = a.T if a.ndim > 1 else a; b0,b1,b2,b3 = b
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2,
                     a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0], -1)

def left_matrix(g):            # x -> g x as a 4x4 orthogonal matrix
    return np.stack([qmul(g, e) for e in np.eye(4)], 1)

def icosians():
    pts = []
    for i in range(4):
        for s in (1, -1):
            e = np.zeros(4); e[i] = s; pts.append(e)
    for s in itertools.product((0.5, -0.5), repeat=4): pts.append(np.array(s))
    even = [p for p in itertools.permutations(range(4))
            if sum(p[i] > p[j] for i in range(4) for j in range(i+1, 4)) % 2 == 0]
    base = np.array([PHI, 1, 1/PHI, 0])/2
    for p in even:
        for s in itertools.product((1, -1), repeat=3):
            v = base.copy(); v[:3] *= s; w = np.zeros(4); w[list(p)] = v; pts.append(w)
    return np.unique(np.round(np.array(pts), 12), axis=0)

def generated(gens, G):
    tree = cKDTree(G); S = {tree.query(np.array([1., 0, 0, 0]))[1]}; frontier = list(S)
    while frontier:
        new = []
        for i in frontier:
            for g in gens:
                j = tree.query(qmul(g, G[i]))[1]
                if j not in S: S.add(j); new.append(j)
        frontier = new
    return G[sorted(S)]

def subgroup(name, G):
    re = G[:, 0]
    pick = lambda c: G[np.argmin(np.abs(re - c) + 1e-3*np.arange(len(G))/len(G))]
    if name == "S3":   return G[[np.argmax(re)]]
    if name == "RP3":  return generated([np.array([-1., 0, 0, 0])], G)
    if name == "L4":   return generated([pick(0.0)], G)
    if name == "L6":   return generated([pick(0.5)], G)
    if name == "L10":  return generated([pick(PHI/2)], G)
    if name == "Q8":   return G[np.isclose(np.abs(G).max(1), 1)]
    if name == "2T":   return G[np.isclose(np.abs(G).max(1), 1) | np.isclose(np.abs(G).max(1), .5)]
    if name == "PDS":  return G
    raise ValueError(name)

def exact_spectrum(Gam, kmax=40):
    th = np.arccos(np.clip(Gam[:, 0], -1, 1)); out = []
    for k in range(kmax + 1):
        chi = np.where(np.sin(th) < 1e-9, (k+1)*np.sign(np.cos(th))**k,
                       np.sin((k+1)*th)/np.where(np.sin(th) < 1e-9, 1, np.sin(th)))
        m = (k+1)*int(round(chi.sum()/len(Gam)))
        if m: out.append((k, k*(k+2), m))
    return out

def multi_indices(d):
    return np.array([(a, b, c, d-a-b-c) for a in range(d, -1, -1)
                     for b in range(d-a, -1, -1) for c in range(d-a-b, -1, -1)])

def duffy3(nq):
    z, w = np.polynomial.legendre.leggauss(nq); z = (z+1)/2; w = w/2
    U, Vv, W = np.meshgrid(z, z, z, indexing="ij")
    wt = (w[:, None, None]*w[None, :, None]*w[None, None, :]*(1-U)**2*(1-Vv)).ravel()
    b1 = U; b2 = (1-U)*Vv; b3 = (1-U)*(1-Vv)*W; b0 = 1-b1-b2-b3
    return np.stack([b0, b1, b2, b3], -1).reshape(-1, 4), wt

def bern(d, A, b):
    coef = np.array([math.factorial(d)/np.prod([math.factorial(int(t)) for t in a]) for a in A])
    P = np.stack([b**e for e in range(d+1)], 0)            # P[e, q, i]
    pw = lambda i, e: P[np.clip(e, 0, d)][:, :, i].T        # (nq, nloc)
    val = coef*np.prod([P[A[:, i], :, i].T for i in range(4)], 0)
    grad = np.zeros(b.shape[:1] + (len(A), 4))
    for i in range(4):
        t = coef*A[:, i]*P[np.clip(A[:, i]-1, 0, d), :, i].T
        for j in range(4):
            if j != i: t = t*P[A[:, j], :, j].T
        grad[:, :, i] = t
    return val, grad

def refine(cells):
    out = []
    for V in cells:
        v = [V[:, i] for i in range(4)]
        m = {(i, j): (v[i]+v[j])/np.linalg.norm(v[i]+v[j]) for i in range(4) for j in range(i+1, 4)}
        for i in range(4):
            out.append(np.stack([v[i]] + [m[tuple(sorted((i, j)))] for j in range(4) if j != i], 1))
        dg = min([((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))],
                 key=lambda p: np.linalg.norm(m[p[0]]-m[p[1]]))
        a, c = dg; rest = [e for e in m if e not in dg]
        for e in rest:
            for f in rest:
                if e < f and len(set(e) & set(f)) == 1 and \
                   tuple(sorted(set(e) ^ set(f))) not in dg:
                    pass
        # octahedron split about the diagonal (a,c): four tets (m_a, m_c, p, q), p,q adjacent on the equator
        eq = [e for e in m if e not in (a, c)]
        order = [eq[0]]
        while len(order) < 4:
            for e in eq:
                if e not in order and len(set(e) & set(order[-1])) == 1: order.append(e); break
        for t in range(4):
            out.append(np.stack([m[a], m[c], m[order[t]], m[order[(t+1) % 4]]], 1))
    return out

def build(name="PDS", d=6, level=0, nq=None, neig=40, verbose=True):
    G = icosians(); assert len(G) == 120
    Gam = subgroup(name, G); L = [left_matrix(g) for g in Gam]
    hull = ConvexHull(G); tets = np.sort(hull.simplices, 1); tree = cKDTree(G)
    seen = set(); reps = []
    for t in map(tuple, tets):
        if t in seen: continue
        reps.append(t)
        for Lg in L: seen.add(tuple(sorted(tree.query((Lg@G[list(t)].T).T)[1])))
    cells = []
    for t in reps:
        V = G[list(t)].T.copy()
        if np.linalg.det(V) < 0: V[:, [0, 1]] = V[:, [1, 0]]
        cells.append(V)
    for _ in range(level): cells = refine(cells)
    A = multi_indices(d); nloc = len(A)
    pts = np.concatenate([(V@(A.T/d)).T for V in cells]); pts /= np.linalg.norm(pts, axis=1)[:, None]
    parent = np.arange(len(pts))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    pt = cKDTree(pts)
    for Lg in L:
        img = pts@Lg.T
        for i, nb in enumerate(pt.query_ball_point(img, 1e-9)):
            for j in nb:
                a, b_ = find(i), find(j)
                if a != b_: parent[b_] = a
    roots = np.array([find(i) for i in range(len(pts))]); _, gid = np.unique(roots, return_inverse=True)
    ndof = gid.max()+1
    nq = nq or d+4; bq, wq = duffy3(nq); val, grad = bern(d, A, bq)
    VV = np.asarray(cells); C = len(cells); ei = np.array([0., 1, 0, 0]); nqp = len(bq)
    blk = max(1, int(4e7 // (nqp*nloc*3)))          # cells per block, to bound peak memory
    I, J, MV, KV, XV = [], [], [], [], []; vol = 0.0
    gidc = gid.reshape(C, nloc)
    for s0 in range(0, C, blk):
        Vb = VV[s0:s0+blk]; cb = len(Vb)
        Y = np.einsum("qi,cji->cqj", bq, Vb); R = np.linalg.norm(Y, axis=2)
        W = wq[None, :]*np.abs(np.linalg.det(Vb))[:, None]/R**4; vol += float(W.sum())
        F = val[None, :, :]/R[:, :, None]**d                          # (c,q,nloc)
        GF = np.einsum("qai,cij->cqaj", grad, np.linalg.inv(Vb))/R[:, :, None, None]**(d-1)
        XI = qmul((Y/R[:, :, None]).reshape(-1, 4), ei).reshape(cb, nqp, 4)
        D = np.einsum("cqaj,cqj->cqa", GF, XI)
        # reshape so that BLAS sees one batched matrix product per integral
        Me = (F*W[:, :, None]).transpose(0, 2, 1) @ F
        Gm = GF.transpose(0, 2, 1, 3).reshape(cb, nloc, -1)
        Ke = ((GF*W[:, :, None, None]).transpose(0, 2, 1, 3).reshape(cb, nloc, -1)
              @ Gm.transpose(0, 2, 1)) - d*d*Me
        Xe = (D*W[:, :, None]).transpose(0, 2, 1) @ D
        g = gidc[s0:s0+blk]
        I.append(np.repeat(g, nloc, axis=1).ravel()); J.append(np.tile(g, (1, nloc)).ravel())
        MV.append(Me.ravel()); KV.append(Ke.ravel()); XV.append(Xe.ravel())
    I, J = np.concatenate(I), np.concatenate(J)
    M = sp.coo_matrix((np.concatenate(MV), (I, J)), shape=(ndof, ndof)).tocsc()
    K = sp.coo_matrix((np.concatenate(KV), (I, J)), shape=(ndof, ndof)).tocsc()
    Kxi = sp.coo_matrix((np.concatenate(XV), (I, J)), shape=(ndof, ndof)).tocsc()
    neig = min(neig, ndof-2)
    lam = np.sort(spla.eigsh(K, M=M, k=neig, sigma=-1.0, which="LM", return_eigenvectors=False))
    return dict(K=K, M=M, Kxi=Kxi, Gam=Gam, name=name, order=len(Gam), cells=len(cells), ndof=ndof, vol=vol,
                vol_exact=2*np.pi**2/len(Gam), lam=lam, exact=exact_spectrum(Gam))

def report(R, nclusters=4):
    print(f"\n{R['name']}: |Gamma|={R['order']}, cells={R['cells']}, dofs={R['ndof']}, "
          f"volume error {abs(R['vol']-R['vol_exact']):.2e}")
    pos = 0
    for k, mu, m in R["exact"][:nclusters]:
        if pos+m > len(R["lam"]): break
        c = R["lam"][pos:pos+m]; pos += m
        err = np.abs(c-mu).max()/(mu if mu else 1)
        print(f"   k={k:2d}  mu={mu:4d}  mult={m:3d}  computed [{c.min():.8f}, {c.max():.8f}]  "
              f"{'abs' if mu == 0 else 'rel'} err {err:.2e}")

if __name__ == "__main__":
    name = sys.argv[1]; d = int(sys.argv[2]); lev = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    t = time.time(); R = build(name, d, lev, neig=int(sys.argv[4]) if len(sys.argv) > 4 else 40)
    report(R); print(f"   [{time.time()-t:.0f}s]")

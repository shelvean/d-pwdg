"""
hypforms.py -- homogeneous Bernstein-Bezier C^0 splines on the Seifert-Weber dodecahedral space.
Model: H^3 = {x in R^{3,1}: <x,x> = -1, x_0 > 0}, <x,y> = -x_0 y_0 + x_1 y_1 + x_2 y_2 + x_3 y_3.
Fundamental domain: regular dodecahedron, dihedral angle 2 pi/5, centered at e_0.
Face pairing g_f = (boost by twice the inradius along f) o (rotation by 3 pi/5 about f), same hand for all f.
Forms (kappa = -1): m = int F_a F_b,  k = int grad F_a^T Q grad F_b + d^2 F_a F_b,
chart y = V b (sum b = 1), r = sqrt(-<y,y>), x = y/r, dvol = |det V| r^-4 db.
"""
import sys, time, itertools, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from sphereforms import multi_indices, duffy3, bern

PHI = (1+5**.5)/2; Q = np.diag([-1., 1, 1, 1]); THETA = 2*np.pi/5; TWIST = 3*np.pi/5
mdot = lambda x, y: x@Q@y
def mnorm(y): return y/np.sqrt(-mdot(y, y))

def dodecahedron():
    V = [np.array(s, float) for s in itertools.product((1, -1), repeat=3)]
    for s1, s2 in itertools.product((1, -1), repeat=2):
        for sh in range(3): V.append(np.roll([0, s1/PHI, s2*PHI], sh))
    F = []
    for s1, s2 in itertools.product((1, -1), repeat=2):
        for sh in range(3): F.append(np.roll([0, s1*PHI, s2*1.], sh))
    V = np.array(V); V /= np.linalg.norm(V, axis=1)[:, None]; F = np.array(F); F /= np.linalg.norm(F, axis=1)[:, None]
    return V, F

def geometry():
    U, Fd = dodecahedron()
    a = np.arcsinh(np.sqrt((1/np.sqrt(5) + np.cos(THETA))/(1 - 1/np.sqrt(5))))       # inradius
    cuf = np.max(U@Fd[0]); rho = np.arctanh(np.tanh(a)/cuf)                            # circumradius
    hv = np.array([np.r_[np.cosh(rho), np.sinh(rho)*u] for u in U])
    hf = np.array([np.r_[np.cosh(a), np.sinh(a)*f] for f in Fd])
    faces = []
    for f in Fd:
        idx = np.where(U@f > cuf - 1e-9)[0]; e1 = np.cross(f, [1, 2, 3.]); e1 /= np.linalg.norm(e1); e2 = np.cross(f, e1)
        faces.append(idx[np.argsort(np.arctan2(U[idx]@e2, U[idx]@e1))])
    return a, rho, U, Fd, hv, hf, faces

def pairing(f, a):
    c, s = np.cosh(2*a), np.sinh(2*a); B = np.eye(4); P = np.outer(f, f)
    B[0, 0] = c; B[0, 1:] = s*f; B[1:, 0] = s*f; B[1:, 1:] = np.eye(3) + (c-1)*P
    Kx = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]])
    R = np.eye(4); R[1:, 1:] = np.eye(3) + np.sin(TWIST)*Kx + (1-np.cos(TWIST))*Kx@Kx
    return B@R

def refine(cells):
    out = []
    for V in cells:
        v = [V[:, i] for i in range(4)]; m = {(i, j): mnorm(v[i]+v[j]) for i in range(4) for j in range(i+1, 4)}
        for i in range(4): out.append(np.stack([v[i]] + [m[tuple(sorted((i, j)))] for j in range(4) if j != i], 1))
        a, c = min([((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))], key=lambda p: -mdot(m[p[0]], m[p[1]]))
        eq = [e for e in m if e not in (a, c)]; order = [eq[0]]
        while len(order) < 4:
            for e in eq:
                if e not in order and len(set(e) & set(order[-1])) == 1: order.append(e); break
        for t in range(4): out.append(np.stack([m[a], m[c], m[order[t]], m[order[(t+1) % 4]]], 1))
    return out

def build(d, level, neig=30, nq=None, verbose=True):
    a, rho, U, Fd, hv, hf, faces = geometry(); O = np.array([1., 0, 0, 0])
    gens = [np.eye(4)]
    for f in Fd: gens.append(pairing(f, a))
    # each generator maps the face at -f onto the face at +f, vertex to vertex
    tv = cKDTree(hv)
    for f, g in zip(Fd, gens[1:]):
        src = faces[int(np.argmax(Fd@(-f)))]; dist, j = tv.query(hv[src]@g.T)
        assert dist.max() < 1e-10 and set(j) == set(faces[int(np.argmax(Fd@f))])
    cells = []
    for k, idx in enumerate(faces):
        for i in range(5): cells.append(np.stack([O, hf[k], hv[idx[i]], hv[idx[(i+1) % 5]]], 1))
    for _ in range(level): cells = refine(cells)
    A = multi_indices(d); nloc = len(A)
    pts = np.concatenate([(V@(A.T/d)).T for V in cells]); parent = np.arange(len(pts)); pt = cKDTree(pts)
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for g in gens:
        for i, nb in enumerate(pt.query_ball_point(pts@g.T, 1e-8)):
            for j in nb:
                x, y = find(i), find(j)
                if x != y: parent[y] = x
    roots = np.array([find(i) for i in range(len(pts))]); _, gid = np.unique(roots, return_inverse=True); ndof = gid.max()+1
    # topology of the glued dodecahedron: classes of its 20 vertices and 30 edge midpoints
    vid = [gid[pt.query(v)[1]] for v in hv]
    info = dict(vertex_classes=len(set(vid)))
    nq = nq or d+6; bq, wq = duffy3(nq); val, grad = bern(d, A, bq)
    VV = np.asarray(cells); C = len(cells); nqp = len(bq); nloc = len(A)
    blk = max(1, int(4e7 // (nqp*nloc*3)))          # cells per block, to bound peak memory
    I, J, MV, KV = [], [], [], []; vol = 0.0; rng = [1e9, 0]
    gidc = gid.reshape(C, nloc); dQ = np.diag(Q)
    for s0 in range(0, C, blk):
        Vb = VV[s0:s0+blk]; cb = len(Vb)
        Y = np.einsum("qi,cji->cqj", bq, Vb); R = np.sqrt(-np.einsum("cqi,ij,cqj->cq", Y, Q, Y))
        W = wq[None, :]*np.abs(np.linalg.det(Vb))[:, None]/R**4; vol += float(W.sum())
        rng = [min(rng[0], R.min()), max(rng[1], (R.max(axis=1)/R.min(axis=1)).max())]
        F = val[None, :, :]/R[:, :, None]**d
        GF = np.einsum("qai,cij->cqaj", grad, np.linalg.inv(Vb))/R[:, :, None, None]**(d-1)
        Me = (F*W[:, :, None]).transpose(0, 2, 1) @ F
        GW = (GF*dQ)*W[:, :, None, None]
        Ke = (GW.transpose(0, 2, 1, 3).reshape(cb, nloc, -1) @
              GF.transpose(0, 2, 1, 3).reshape(cb, nloc, -1).transpose(0, 2, 1)) + d*d*Me
        g = gidc[s0:s0+blk]
        I.append(np.repeat(g, nloc, axis=1).ravel()); J.append(np.tile(g, (1, nloc)).ravel())
        MV.append(Me.ravel()); KV.append(Ke.ravel())
    I, J = np.concatenate(I), np.concatenate(J)
    M = sp.coo_matrix((np.concatenate(MV), (I, J)), shape=(ndof, ndof)).tocsc()
    K = sp.coo_matrix((np.concatenate(KV), (I, J)), shape=(ndof, ndof)).tocsc()
    lam = np.sort(spla.eigsh(K, M=M, k=neig, sigma=-1.0, which="LM", ncv=4*neig, return_eigenvectors=False))
    return dict(a=a, rho=rho, cells=len(cells), ndof=ndof, vol=vol, lam=lam, rmin=rng[0], wratio=rng[1]**4, **info)

if __name__ == "__main__":
    d, L = int(sys.argv[1]), int(sys.argv[2]); t = time.time(); R = build(d, L)
    print(f"d={d} L={L}: cells={R['cells']} unknowns={R['ndof']} inradius={R['a']:.6f} circumradius={R['rho']:.6f} "
          f"volume={R['vol']:.10f} vertex classes={R['vertex_classes']} max cell weight ratio={R['wratio']:.1f} [{time.time()-t:.0f}s]")
    print("   ", np.array2string(R["lam"][:22], precision=6, max_line_width=130))

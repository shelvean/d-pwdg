"""Laplace-Beltrami eigenvalues on the two embedded surfaces of the figures,
by P1 surface finite elements on the triangulated surface (Dziuk): the tube
torus of tube_torus.py and the pretzel of pretzel.py. Reports the first
eigenvalues at three mesh sizes and the area, with Richardson-style
extrapolation of mu_1."""
import numpy as np, json
import scipy.sparse as sp, scipy.sparse.linalg as spla

def p1_matrices(V, F):
    n = len(V); I=[]; J=[]; Kv=[]; Mv=[]
    for f in F:
        P = V[f]; e1 = P[1]-P[0]; e2 = P[2]-P[0]
        nrm = np.cross(e1, e2); A = 0.5*np.linalg.norm(nrm)
        if A < 1e-14: continue
        # cotangent stiffness
        for a in range(3):
            b, c = (a+1) % 3, (a+2) % 3
            u = P[b]-P[a]; v = P[c]-P[a]
            cot = (u@v)/ (2*A*2)          # cot at vertex a / 2 handled below
        # standard cotan weights
        L = np.zeros((3,3))
        for a in range(3):
            b, c = (a+1) % 3, (a+2) % 3
            u = P[b]-P[a]; v = P[c]-P[a]
            cot_a = (u@v)/np.linalg.norm(np.cross(u, v))
            L[b, c] -= 0.5*cot_a; L[c, b] -= 0.5*cot_a
            L[b, b] += 0.5*cot_a; L[c, c] += 0.5*cot_a
        Mloc = A/12.0*np.array([[2.,1,1],[1,2,1],[1,1,2]])
        for a in range(3):
            for b in range(3):
                I.append(f[a]); J.append(f[b]); Kv.append(L[a,b]); Mv.append(Mloc[a,b])
    K = sp.csr_matrix((Kv,(I,J)), shape=(n,n))
    M = sp.csr_matrix((Mv,(I,J)), shape=(n,n))
    return K, M

def weld(X, F, tol=9):
    key = np.round(X, tol); _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    Xw = X[idx]; Fw = inv[F]
    Fw = np.array([f for f in Fw if len(set(f)) == 3])
    return Xw, Fw

def tube_mesh(ns, nt):
    from tube_torus import emb
    S = np.linspace(0, 2*np.pi, ns, endpoint=False)
    T = np.linspace(0, 2*np.pi, nt, endpoint=False)
    X = np.array([emb(s, t) for s in S for t in T])
    F = []
    for i in range(ns):
        for j in range(nt):
            a = i*nt + j; b = ((i+1) % ns)*nt + j
            c = i*nt + (j+1) % nt; d = ((i+1) % ns)*nt + (j+1) % nt
            F += [(a, b, d), (a, d, c)]
    return weld(X, np.array(F))

def pretzel_mesh(n):
    from pretzel import surf
    V, F = surf(n=n)
    return weld(V, F)

def eigs(V, F, k=8):
    K, M = p1_matrices(V, F)
    area = M.sum()
    vals = np.sort(spla.eigsh(K.tocsc(), k=k, M=M.tocsc(), sigma=-1e-6, which='LM',
                              return_eigenvectors=False).real)
    return area, vals

if __name__ == '__main__':
    out = {}
    print("tube torus (P1 surface FEM)")
    rows = []
    for ns, nt in [(48, 24), (96, 48), (160, 80)]:
        V, F = tube_mesh(ns, nt); area, v = eigs(V, F)
        h = np.sqrt(area/len(F))
        rows.append(dict(ns=ns, nt=nt, nv=len(V), nf=len(F), h=float(h), area=float(area),
                         eigs=[float(x) for x in v]))
        print(f"  {ns}x{nt}  V={len(V):6d}  area={area:.6f}  mu1..4 = "
              + "  ".join(f"{x:.6f}" for x in v[1:5]), flush=True)
    out['tube'] = rows
    print("pretzel (P1 surface FEM)")
    rows = []
    for n in [140, 200, 260]:
        V, F = pretzel_mesh(n); area, v = eigs(V, F)
        h = np.sqrt(area/len(F))
        rows.append(dict(n=n, nv=len(V), nf=len(F), h=float(h), area=float(area),
                         eigs=[float(x) for x in v]))
        print(f"  n={n}  V={len(V):6d}  area={area:.6f}  mu1..4 = "
              + "  ".join(f"{x:.6f}" for x in v[1:5]), flush=True)
    out['pretzel'] = rows
    json.dump(out, open('embedded_eigs.json', 'w'), indent=1)

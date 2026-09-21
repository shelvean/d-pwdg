"""High eigenfunctions of the distorted torus drawn on the embedded surface."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flatquot import Quotient, crisscross_mesh
from dtorus import U, weight, rho, zz, t_of

def emb(u, phi):
    t = t_of(u); return np.array([rho(t)*np.cos(phi), rho(t)*np.sin(phi), zz(t)])

WHICH = [1, 12, 30]
Q = Quotient('torus', 12, 12, 6, 1, a=U, b=2*np.pi)
vals, vecs, Z = Q.eigen(k=max(WHICH)+6, w=weight, q=10)
nu, nv = 241, 121
us = np.linspace(0, U, nu); ps = np.linspace(0, 2*np.pi, nv)
Ug, Pg = np.meshgrid(us, ps, indexing='ij')
P = np.array([Ug.ravel(), Pg.ravel()]).T
X = np.array([emb(u, p) for u, p in P])
tris = []
for i in range(nu-1):
    for j in range(nv-1):
        a, b, c, d = i*nv+j, (i+1)*nv+j, i*nv+j+1, (i+1)*nv+j+1
        tris += [(a, b, d), (a, d, c)]
tris = np.array(tris)
from flatquot import bernstein, multi_indices
def fast_eval(c, pts):
    n = 12; hu, hv = U/n, 2*np.pi/n
    out = np.empty(len(pts)); Nloc = Q.Nloc
    cells = {}
    for t, T in enumerate(Q.tris):
        ctr = Q.V[list(T)].mean(0)
        cells.setdefault((min(int(ctr[0]//hu), n-1), min(int(ctr[1]//hv), n-1)), []).append(t)
    for q, p in enumerate(pts):
        i = min(int((p[0] % U)//hu), n-1); j = min(int((p[1] % (2*np.pi))//hv), n-1)
        for t in cells[(i, j)]:
            V3 = Q.V[list(Q.tris[t])]
            A = np.vstack([V3.T, np.ones(3)])
            b = np.linalg.solve(A, np.array([p[0], p[1], 1.0]))
            if b.min() > -1e-9:
                out[q] = bernstein(Q.d, b[:, None])[:, 0] @ c[t*Nloc:(t+1)*Nloc]
                break
        else:
            out[q] = 0.0
    return out

for k in WHICH:
    u = fast_eval(vecs[:, k], P); u = u/np.abs(u).max()
    fig = mlab.figure(bgcolor=(1, 1, 1), size=(1400, 1000))
    s = mlab.triangular_mesh(X[:, 0], X[:, 1], X[:, 2], tris, scalars=u,
                             colormap='RdBu', vmin=-1, vmax=1)
    s.module_manager.scalar_lut_manager.reverse_lut = True
    s.actor.property.specular = 0.08
    mlab.view(azimuth=-58, elevation=62, distance='auto')
    fig.scene.parallel_projection = True
    mlab.savefig(f'/home/claude/work/hyp/paper/p_dtorus_m{k}.png', size=(1400, 1000))
    mlab.close(fig); print(k, vals[k], flush=True)

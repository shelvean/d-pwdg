"""Curved (surface-conforming) triangulation on the pretzel: the mesh vertices
and edge samples are projected onto the implicit surface by Newton along the
gradient, so every edge is a curve lying on the surface rather than a chord.
Shaded with the Mayavi pipeline used for the other panels."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from pretzel import genus2, surf, decimate

def grad(p, h=1e-5):
    g = np.empty_like(p)
    for k in range(3):
        e = np.zeros(3); e[k] = h
        g[..., k] = (genus2(*(p+e).T) - genus2(*(p-e).T))/(2*h)
    return g

def project(P, iters=12):
    P = np.array(P, dtype=float)
    for _ in range(iters):
        f = genus2(*P.T); g = grad(P)
        n2 = (g*g).sum(1) + 1e-30
        P = P - (f/n2)[:, None]*g
    return P

def render(out, target=20, sub=5, tube=0.009, size=(1500, 1050)):
    V, F = surf()
    Vc, Fc = decimate(V, F, target)
    Vc = project(Vc)
    # curved surface patches: subdivide each triangle and project
    X = []; T = []
    for f in Fc:
        P = Vc[list(f)]; idx = {}
        for i in range(sub+1):
            for j in range(sub+1-i):
                b = np.array([i, j, sub-i-j])/sub
                idx[(i, j)] = len(X); X.append(b @ P)
        for i in range(sub):
            for j in range(sub-i):
                T.append([idx[(i, j)], idx[(i+1, j)], idx[(i, j+1)]])
                if j < sub-i-1: T.append([idx[(i+1, j)], idx[(i+1, j+1)], idx[(i, j+1)]])
    X = project(np.array(X)); T = np.array(T)
    key = np.round(X, 6); _, inv = np.unique(key, axis=0, return_inverse=True)
    Xw = np.zeros((inv.max()+1, 3)); Xw[inv] = X; Tw = inv[T]
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    s = mlab.triangular_mesh(Xw[:, 0], Xw[:, 1], Xw[:, 2], Tw, color=(0.86, 0.88, 0.93))
    s.actor.property.specular = 0.12
    px, py, pz, conn = [], [], [], []; k = 0
    E = set()
    for f in Fc:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])): E.add((min(a, b), max(a, b)))
    for a, b in sorted(E):
        ts = np.linspace(0, 1, 12)
        pts = project(np.array([Vc[a]*(1-t) + Vc[b]*t for t in ts]))
        px += list(pts[:, 0]); py += list(pts[:, 1]); pz += list(pts[:, 2])
        conn += [(k+i, k+i+1) for i in range(len(ts)-1)]; k += len(ts)
    src = mlab.pipeline.scalar_scatter(np.array(px), np.array(py), np.array(pz))
    src.mlab_source.dataset.lines = np.array(conn)
    lines = mlab.pipeline.stripper(src)
    tb = mlab.pipeline.tube(lines, tube_radius=tube); tb.filter.number_of_sides = 8
    mlab.pipeline.surface(tb, color=(0.22, 0.22, 0.28))
    mlab.view(azimuth=-60, elevation=58, distance='auto'); fig.scene.parallel_projection = True
    mlab.savefig(out, size=size); mlab.close(fig)
    print(out, len(Vc), len(Fc), flush=True)

if __name__ == '__main__':
    render('/home/claude/work/hyp/paper/p_pretzel_mesh.png')

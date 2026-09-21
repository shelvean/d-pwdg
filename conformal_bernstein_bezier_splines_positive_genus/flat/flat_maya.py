"""Mayavi renderings of the flat quotients (torus, Mobius strip, Klein bottle)
with the criss-cross mesh mapped through the embedding; light surfaces, mesh
edges as tubes, no colorbars. Run: xvfb-run -a -s "-screen 0 1600x1300x24" python3 flat_maya.py"""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flatquot import crisscross_mesh
from flat_figs import torus, mobius, klein8

def render(emb, n, m, view, out, sub=8, tube=0.012, size=(1600, 1300), inset=None):
    V, T = crisscross_mesh(n, m)
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    # surface: fine sampling of each element
    X = []; F = []; base = 0
    for tri in T:
        P = V[list(tri)]
        idx = {}
        for i in range(sub + 1):
            for j in range(sub + 1 - i):
                b = np.array([i, j, sub - i - j]) / sub; p = b @ P
                idx[(i, j)] = base + len(X) - base; X.append(emb(p[0], p[1]))
        for i in range(sub):
            for j in range(sub - i):
                F.append([idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]])
                if j < sub - i - 1: F.append([idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]])
        base = len(X)
    X = np.array(X); F = np.array(F)
    # weld coincident points so shading has no seams
    key = np.round(X, 6); _, inv = np.unique(key, axis=0, return_inverse=True)
    Xw = np.zeros((inv.max() + 1, 3)); Xw[inv] = X; Fw = inv[F]
    if inset is not None: Xw = inset(Xw)
    s = mlab.triangular_mesh(Xw[:, 0], Xw[:, 1], Xw[:, 2], Fw, color=(0.86, 0.88, 0.93), opacity=1.0)
    s.actor.property.specular = 0.15; s.actor.property.specular_power = 20
    # mesh edges
    for tri in T:
        P = V[list(tri)]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ts = np.linspace(0, 1, 16); pts = np.array([emb(*(P[a] * (1 - t) + P[b] * t)) for t in ts])
            mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(0.25, 0.25, 0.3), tube_radius=tube)
    mlab.view(**view); fig.scene.parallel_projection = True
    mlab.savefig(out, size=size); mlab.close(fig); print(out)

def torus_inset(X, R=2.0, eps=0.03):
    c = X.copy(); rho = np.hypot(X[:, 0], X[:, 1]); cx = np.c_[R * X[:, 0] / rho, R * X[:, 1] / rho, np.zeros(len(X))]
    nrm = X - cx; nrm /= np.linalg.norm(nrm, axis=1)[:, None]; return X - eps * nrm
if __name__ == '__main__':
    render(torus, 8, 8, dict(azimuth=-60, elevation=55, distance='auto'), '/home/claude/work/hyp/paper/fig_torus_mesh.png', tube=0.025, inset=torus_inset)
    render(mobius, 12, 3, dict(azimuth=-50, elevation=50, distance='auto'), '/home/claude/work/hyp/paper/fig_mobius_mesh.png', tube=0.018)
    render(klein8, 8, 8, dict(azimuth=-40, elevation=65, distance='auto'), '/home/claude/work/hyp/paper/fig_klein_mesh.png', tube=0.02)

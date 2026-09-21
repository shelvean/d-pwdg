"""A distorted solid-torus surface of the tube type: a closed core curve with a
rotation-minimizing frame and a cross-section radius that varies along the
core, so the distortion is nonaxisymmetric and visible from outside. Rendered
with the criss-cross triangulation of the parameter rectangle carried through
the tube map, edges as curves on the surface."""
import numpy as np
try:
    from mayavi import mlab
    mlab.options.offscreen = True
except Exception:
    mlab = None
from flatquot import crisscross_mesh

R, B, C, A0, EPS = 2.0, 0.35, 0.55, 0.62, 0.30
def core(s):
    r = R + B*np.cos(3*s)
    return np.array([r*np.cos(s), r*np.sin(s), C*np.sin(2*s)])
def dcore(s, h=1e-6): return (core(s+h)-core(s-h))/(2*h)
def radius(s): return A0*(1 + EPS*np.cos(2*s))

_S = np.linspace(0, 2*np.pi, 2001)
def frames():
    T = np.array([dcore(s) for s in _S]); T /= np.linalg.norm(T, axis=1)[:, None]
    N = np.zeros_like(T); v = np.array([0., 0., 1.])
    n = v - (v@T[0])*T[0]; n /= np.linalg.norm(n); N[0] = n
    for i in range(1, len(_S)):                      # rotation-minimizing transport
        n = N[i-1] - (N[i-1]@T[i])*T[i]; N[i] = n/np.linalg.norm(n)
    # close the frame smoothly: remove the holonomy linearly in s
    ang = np.arctan2(np.cross(N[0], N[-1])@T[-1], N[0]@N[-1])
    for i, s in enumerate(_S):
        a = -ang*s/(2*np.pi); b2 = np.cross(T[i], N[i])
        N[i] = np.cos(a)*N[i] + np.sin(a)*b2
    return T, N
_T, _N = frames()
def frame(s):
    s = np.mod(s, 2*np.pi)
    t = np.array([np.interp(s, _S, _T[:, k]) for k in range(3)])
    n = np.array([np.interp(s, _S, _N[:, k]) for k in range(3)])
    t /= np.linalg.norm(t); n = n - (n@t)*t; n /= np.linalg.norm(n)
    return t, n, np.cross(t, n)
def emb(s, th):
    t, n, b = frame(s); return core(s) + radius(s)*(np.cos(th)*n + np.sin(th)*b)

def render(out, n=8, m=8, sub=8, tube=0.012, size=(1500, 1100), view=None):
    V, T = crisscross_mesh(n, m, 2*np.pi, 2*np.pi)
    X = []; F = []
    for tri in T:
        P = V[list(tri)]; idx = {}
        for i in range(sub+1):
            for j in range(sub+1-i):
                b = np.array([i, j, sub-i-j])/sub; p = b @ P
                idx[(i, j)] = len(X); X.append(emb(p[0], p[1]))
        for i in range(sub):
            for j in range(sub-i):
                F.append([idx[(i, j)], idx[(i+1, j)], idx[(i, j+1)]])
                if j < sub-i-1: F.append([idx[(i+1, j)], idx[(i+1, j+1)], idx[(i, j+1)]])
    X = np.array(X); F = np.array(F)
    key = np.round(X, 6); _, inv = np.unique(key, axis=0, return_inverse=True)
    Xw = np.zeros((inv.max()+1, 3)); Xw[inv] = X; Fw = inv[F]
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    s = mlab.triangular_mesh(Xw[:, 0], Xw[:, 1], Xw[:, 2], Fw, color=(0.86, 0.88, 0.93))
    s.actor.property.specular = 0.12
    px, py, pz, conn = [], [], [], []; k = 0
    for tri in T:
        P = V[list(tri)]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            ts = np.linspace(0, 1, 14)
            pts = np.array([emb(*(P[a]*(1-t)+P[b]*t)) for t in ts])
            px += list(pts[:, 0]); py += list(pts[:, 1]); pz += list(pts[:, 2])
            conn += [(k+i, k+i+1) for i in range(len(ts)-1)]; k += len(ts)
    src = mlab.pipeline.scalar_scatter(np.array(px), np.array(py), np.array(pz))
    src.mlab_source.dataset.lines = np.array(conn)
    lines = mlab.pipeline.stripper(src)
    tb = mlab.pipeline.tube(lines, tube_radius=tube); tb.filter.number_of_sides = 8
    mlab.pipeline.surface(tb, color=(0.22, 0.22, 0.28))
    mlab.view(**(view or dict(azimuth=-55, elevation=58, distance='auto')))
    fig.scene.parallel_projection = True
    mlab.savefig(out, size=size); mlab.close(fig); print(out, flush=True)

if __name__ == '__main__':
    render('/home/claude/work/hyp/paper/p_tube_mesh.png')

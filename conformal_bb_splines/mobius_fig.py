"""Raw Mayavi panels of Mobius-band eigenfunctions at low, medium and high
frequency (no text; annotate in LaTeX)."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from flatquot import Quotient
from flat_figs import mobius
from mobius_eig import w

WHICH = [1, 9, 25]          # low, medium, high
Q = Quotient('mobius', 12, 12, 6, 1)
vals, vecs, Z = Q.eigen(k=max(WHICH) + 4, w=w, q=10, sigma=-1.0)
nx, ny = 241, 41
xs = np.linspace(0, 1, nx); ys = np.linspace(0, 1, ny)
X, Y = np.meshgrid(xs, ys, indexing='ij'); P = np.array([X.ravel(), Y.ravel()]).T
E = np.array([mobius(x, y) for x, y in P])
tris = []
for i in range(nx - 1):
    for j in range(ny - 1):
        a, b, c, d = i*ny+j, (i+1)*ny+j, i*ny+j+1, (i+1)*ny+j+1
        tris += [(a, b, d), (a, d, c)]
tris = np.array(tris)
size = (1500, 1100)
for k in WHICH:
    u = Q.evaluate(vecs[:, k], P); u = u / np.abs(u).max()
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    s = mlab.triangular_mesh(E[:, 0], E[:, 1], E[:, 2], tris, scalars=u, colormap='RdBu',
                             vmin=-1, vmax=1)
    s.module_manager.scalar_lut_manager.reverse_lut = True
    s.actor.property.specular = 0.08
    for yv in [0.0, 1.0]:
        pts = np.array([mobius(x, yv) for x in np.linspace(0, 1, 400)])
        mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(0.2, 0.2, 0.25), tube_radius=0.012)
    mlab.view(azimuth=-50, elevation=50, distance='auto'); fig.scene.parallel_projection = True
    mlab.savefig(f'/home/claude/work/hyp/paper/p_mobius_mode{k}.png', size=size); mlab.close(fig)
    print(k, vals[k], flush=True)

"""
viz_figures.py: Mayavi renderings for the paper (offscreen, white background,
light diverging colormap, no colorbar since eigenfunctions carry arbitrary
sign and scale). Run under xvfb-run.
"""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True

from barynets import indices
from sphsplines import bern_eval
from spheroid import SpheroidWeight

DATA = np.load('results_numerics.npz', allow_pickle=True)


def vis_mesh(v, t, d, c, nsub=12, embed=None):
    """Evaluate the spline on a barycentric lattice of each spherical
    triangle; return (X, Y, Z, tris, vals) with per-triangle duplication."""
    # lattice barycentrics and local subtriangle connectivity
    lam = []
    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            lam.append([i, j, nsub - i - j])
    lam = np.array(lam, dtype=float).T / nsub          # 3 x np
    npt = lam.shape[1]

    def rowstart(i):
        return i * (nsub + 1) - i * (i - 1) // 2

    tris_loc = []
    for i in range(nsub):
        for j in range(nsub - i):
            a = rowstart(i) + j
            b = rowstart(i + 1) + j
            tris_loc.append([a, a + 1, b])
            if j < nsub - i - 1:
                tris_loc.append([a + 1, b + 1, b])
    tris_loc = np.array(tris_loc, dtype=np.int64)

    m = (d + 1) * (d + 2) // 2
    P, V, T = [], [], []
    for kk in range(t.shape[0]):
        v1, v2, v3 = v[t[kk, 0]], v[t[kk, 1]], v[t[kk, 2]]
        Vm = np.column_stack([v1, v2, v3])
        u = Vm @ lam
        r = np.linalg.norm(u, axis=0)
        pts = u / r
        sb = lam / r
        B = bern_eval(d, sb)
        vals = c[kk * m:(kk + 1) * m] @ B
        P.append(pts)
        V.append(vals)
        T.append(tris_loc + kk * npt)
    pts = np.concatenate(P, axis=1)
    vals = np.concatenate(V)
    tris = np.vstack(T)
    if embed is not None:
        pts = embed(pts)
    return pts, tris, vals


def render(pts, tris, vals, fname, view=(45, 60), size=(1200, 1000)):
    vals = vals / np.abs(vals).max()
    if vals[np.argmax(np.abs(vals))] < 0:
        vals = -vals
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=size)
    mlab.triangular_mesh(pts[0], pts[1], pts[2], tris, scalars=vals,
                         colormap='coolwarm', vmin=-1.0, vmax=1.0)
    mlab.view(azimuth=view[0], elevation=view[1],
              distance='auto', focalpoint='auto')
    mlab.savefig(fname, magnification=1)
    mlab.close(fig)
    print("wrote", fname)


# Figure 1: round sphere, eigenfunction inside the exact mu = 42 cluster
v = DATA['sphere_vec_v']; t = DATA['sphere_vec_t']
d = int(DATA['sphere_vec_d']); c = DATA['sphere_vec_c']
pts, tris, vals = vis_mesh(v, t, d, c, nsub=12)
render(pts, tris, vals, 'fig_sphere_eig.png', view=(45, 60))

# Figure 2: spheroid, seventh eigenfunction, on the embedded surface
sw = SpheroidWeight(1.0, 1.5)
v = DATA['spheroid_vec_v']; t = DATA['spheroid_vec_t']
d = int(DATA['spheroid_vec_d']); c = DATA['spheroid_vec_c']
pts, tris, vals = vis_mesh(v, t, d, c, nsub=12, embed=sw.embed)
render(pts, tris, vals, 'fig_spheroid_eig.png', view=(40, 70))

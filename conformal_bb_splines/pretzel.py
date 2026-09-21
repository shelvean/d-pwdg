"""Raw Mayavi panels for the genus-two comparison figure:
(a) a physical pretzel in R^3 (implicit genus-2 surface, marching cubes) with
    a triangle mesh on it, the object a surface FEM discretizes;
(b) the same surface with a coarse flat-triangle mesh (what P1 surface FEM
    actually assembles on);
(c) the hyperbolic octagon mesh on the hyperboloid, the object this paper
    discretizes.
No text; annotate in LaTeX."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from skimage import measure

def genus2(x, y, z, R=1.0, r=0.36, c=1.12, k=0.28):
    """smooth union of two tori: a clean genus-two pretzel as a signed distance
    function, so marching cubes returns a smooth surface"""
    def torus(cx):
        q = np.sqrt((x - cx)**2 + z**2) - R
        return np.sqrt(q**2 + y**2) - r
    d1, d2 = torus(-c), torus(c)
    h = np.clip(0.5 + 0.5*(d2 - d1)/k, 0.0, 1.0)      # polynomial smooth min
    return d2*(1-h) + d1*h - k*h*(1-h)

def surf(n=220, a=2.7, b=0.75, c=1.7):
    xs = np.linspace(-a, a, n); ys = np.linspace(-b, b, max(70, n//3)); zs = np.linspace(-c, c, max(140, 2*n//3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    V = genus2(X, Y, Z)
    verts, faces, _, _ = measure.marching_cubes(V, 0.0,
        spacing=(xs[1]-xs[0], ys[1]-ys[0], zs[1]-zs[0]))
    verts += np.array([xs[0], ys[0], zs[0]])
    return verts, faces

def decimate(verts, faces, target):
    """crude vertex clustering on a grid, keeping topology for coarse display"""
    import collections
    h = (verts.max(0)-verts.min(0)).max()/target
    key = np.floor((verts-verts.min(0))/h).astype(int)
    d = {}
    idx = np.empty(len(verts), dtype=int)
    reps = []
    for i, k in enumerate(map(tuple, key)):
        if k not in d:
            d[k] = len(reps); reps.append(verts[i])
        idx[i] = d[k]
    reps = np.array(reps)
    F = idx[faces]
    F = np.array([f for f in F if len(set(f)) == 3])
    return reps, F

def render(verts, faces, out, wire=True, size=(1500, 1000), view=None, tube=None):
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    s = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces,
                             color=(0.86, 0.88, 0.93))
    s.actor.property.specular = 0.12
    if wire:
        # edges as tubes: one line source with connectivity, so the mesh reads
        # clearly at print size
        import itertools
        E = set()
        for f in faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                E.add((min(a, b), max(a, b)))
        E = np.array(sorted(E))
        src = mlab.pipeline.scalar_scatter(verts[:, 0], verts[:, 1], verts[:, 2])
        src.mlab_source.dataset.lines = E
        lines = mlab.pipeline.stripper(src)
        tb = mlab.pipeline.tube(lines, tube_radius=tube or 0.012)
        tb.filter.number_of_sides = 8
        mlab.pipeline.surface(tb, color=(0.20, 0.20, 0.26))
    mlab.view(**(view or dict(azimuth=-60, elevation=58, distance='auto')))
    fig.scene.parallel_projection = True
    mlab.savefig(out, size=size); mlab.close(fig); print(out, len(verts), len(faces), flush=True)

if __name__ == '__main__':
    V, F = surf()
    render(V, F, '/home/claude/work/hyp/paper/p_pretzel_smooth.png', wire=False)
    Vc, Fc = decimate(V, F, 15)
    render(Vc, Fc, '/home/claude/work/hyp/paper/p_pretzel_mesh.png', wire=True, tube=0.022)

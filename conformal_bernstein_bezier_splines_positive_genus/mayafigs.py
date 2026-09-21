"""Mayavi renderings for the hyperbolic spline manuscript.

Style: light surfaces (user preference), white background, no colorbars or
legends; labels are composed afterwards in matplotlib so the typography matches
the paper. Each mayavi scene is rendered large and downsampled for crispness.
"""
import os
import numpy as np
from mayavi import mlab
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from hypgeom import bolza_octagon, geodesic_point, regular_polygon
from hypbb import bern_indices, bern_eval, bary, local_matrices, constant_coeffs
from hypmesh import octagon_mesh, build_dofs

mlab.options.offscreen = True
W, H = 1600, 1300
BG = (1, 1, 1)
FG = (0.15, 0.15, 0.15)
SURF = (0.93, 0.93, 0.94)
EDGE = (0.18, 0.18, 0.20)
LIGHTGREY = (0.55, 0.55, 0.58)


def new_fig():
    f = mlab.figure(size=(W, H), bgcolor=BG, fgcolor=FG)
    f.scene.parallel_projection = True
    return f


def tube(P, color=EDGE, radius=0.014):
    mlab.plot3d(P[:, 0], P[:, 1], P[:, 2], color=color,
                tube_radius=radius, tube_sides=12)


def geod(a, b, n=60):
    return np.array([geodesic_point(a, b, s) for s in np.linspace(0, 1, n)])


def save(name):
    mlab.savefig(name, magnification=1)
    mlab.close(all=True)


# ---------------------------------------------------------------- scene 1: models
def scene_sphere():
    new_fig()
    u, v = np.mgrid[0:np.pi:80j, 0:2*np.pi:120j]
    mlab.mesh(np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u),
              color=SURF, opacity=1.0)
    tri = np.array([[0, 0, 1.0],
                    [np.sin(1.0), 0, np.cos(1.0)],
                    [np.sin(1.0)*np.cos(1.9), np.sin(1.0)*np.sin(1.9), np.cos(1.0)]])
    for i in range(3):
        a, b = tri[i], tri[(i+1) % 3]
        t = np.linspace(0, 1, 60)[:, None]
        c = a*(1-t)+b*t
        c = 1.002*c/np.linalg.norm(c, axis=1)[:, None]
        tube(c, radius=0.012)
    for p in tri:
        mlab.points3d(*(1.002*p), color=EDGE, scale_factor=0.05)
    mlab.view(azimuth=40, elevation=62, distance=4.6)
    save("m_sphere.png")


def scene_plane():
    new_fig()
    g = np.linspace(-1.25, 1.25, 2)
    G1, G2 = np.meshgrid(g, g)
    mlab.mesh(G1, G2, np.ones_like(G1)*0, color=SURF)
    tri = np.array([[-0.8, -0.65, 0.004], [0.95, -0.5, 0.004],
                    [0.05, 0.9, 0.004], [-0.8, -0.65, 0.004]])
    tube(tri, radius=0.012)
    for p in tri[:3]:
        mlab.points3d(*p, color=EDGE, scale_factor=0.05)
    mlab.view(azimuth=40, elevation=55, distance=5.2)
    save("m_plane.png")


def hyperboloid_patch(rmax=2.0, nr=70, nph=130):
    r, ph = np.mgrid[0:rmax:1j*nr, 0:2*np.pi:1j*nph]
    return r*np.cos(ph), r*np.sin(ph), np.sqrt(1+r*r)


def scene_hyperboloid_model():
    new_fig()
    X, Y, Z = hyperboloid_patch(2.0)
    mlab.mesh(X, Y, Z, color=SURF)
    V0, gens, R = bolza_octagon()
    a = 1.15
    tri = [np.array([1.0, 0.0, 0.0]),
           np.array([np.sqrt(1+a*a), a, 0.0]),
           np.array([np.sqrt(1+a*a), a*np.cos(1.9), a*np.sin(1.9)])]
    for i in range(3):
        A = geod(tri[i], tri[(i+1) % 3], 70)
        tube(A[:, [1, 2, 0]]*1.004, radius=0.014)
    for p in tri:
        mlab.points3d(p[1], p[2], p[0]*1.004, color=EDGE, scale_factor=0.06)
    # cone rays to the triangle vertices
    for p in tri:
        L = np.linspace(0, 1.55, 30)[:, None]*p[None, :]
        mlab.plot3d(L[:, 1], L[:, 2], L[:, 0], color=LIGHTGREY,
                    tube_radius=0.006)
    # light cone, faint
    r, ph = np.mgrid[0:2.0:40j, 0:2*np.pi:90j]
    mlab.mesh(r*np.cos(ph), r*np.sin(ph), r, color=(0.75, 0.75, 0.78),
              opacity=0.16)
    mlab.view(azimuth=35, elevation=34, distance=9.5, focalpoint=(0, 0, 1.5))
    save("m_hyperboloid.png")


# --------------------------------------------------- scene: octagon on hyperboloid
def scene_octagon_3d():
    new_fig()
    X, Y, Z = hyperboloid_patch(6.6)
    mlab.mesh(X, Y, Z, color=SURF, opacity=0.55)
    V0, gens, R = bolza_octagon()
    for j in range(8):
        A = geod(V0[j], V0[(j+1) % 8], 90)
        tube(A[:, [1, 2, 0]]*1.004, radius=0.05)
    O = np.array([1.0, 0, 0])
    for j in range(8):
        A = geod(O, V0[j], 60)
        mlab.plot3d(A[:, 1], A[:, 2], A[:, 0]*1.004, color=LIGHTGREY,
                    tube_radius=0.016)
        mlab.points3d(V0[j][1], V0[j][2], V0[j][0]*1.004, color=EDGE,
                      scale_factor=0.22)
    mlab.view(azimuth=25, elevation=60, distance=27.0, focalpoint=(0, 0, 3.6))
    save("m_octagon3d.png")


# --------------------------------------------------------------- mesh scenes
def octagon_region(level, nsub=6):
    """Triangulated surface of the meshed octagon region on the hyperboloid."""
    tris, V0, gens = octagon_mesh(level)
    lam = np.array([(a/nsub, b/nsub, 1-a/nsub-b/nsub)
                    for a in range(nsub+1) for b in range(nsub+1-a)])
    idx = {(round(l[0]*nsub), round(l[1]*nsub)): k for k, l in enumerate(lam)}
    pts, faces, npt = [], [], 0
    for T in tris:
        X = (T @ lam.T).T
        X = X/np.sqrt(-(-X[:, 0]**2+X[:, 1]**2+X[:, 2]**2))[:, None]
        pts.append(X)
        for a in range(nsub):
            for b in range(nsub-a):
                faces.append([npt+idx[(a, b)], npt+idx[(a+1, b)], npt+idx[(a, b+1)]])
                if a+b < nsub-1:
                    faces.append([npt+idx[(a+1, b)], npt+idx[(a+1, b+1)],
                                  npt+idx[(a, b+1)]])
        npt += X.shape[0]
    return np.vstack(pts), np.array(faces), tris, V0


def scene_mesh(level, name):
    new_fig()
    P, F, tris, V0 = octagon_region(level)
    mlab.triangular_mesh(P[:, 1], P[:, 2], P[:, 0], F, color=SURF)
    rad = {0: 0.032, 1: 0.022, 2: 0.013}[level]
    for T in tris:
        for i in range(3):
            A = geod(T[:, i], T[:, (i+1) % 3], 26)
            mlab.plot3d(A[:, 1], A[:, 2], A[:, 0]*1.004, color=EDGE,
                        tube_radius=rad)
    mlab.view(azimuth=0, elevation=0.1, distance=42.0, focalpoint=(0, 0, 3.4))
    save(name)


# --------------------------------------------------------------- eigenfunctions
def eigfield(level, d, which, nsub=8):
    tris, V0, gens = octagon_mesh(level)
    dof, _ = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d))
    ndof = int(dof.max())+1
    Rr, Cc, mv, kv = [], [], [], []
    for t, T in enumerate(tris):
        M, K = local_matrices(T, d, n=14)
        gl = dof[t*nloc:(t+1)*nloc]
        Rr.append(np.repeat(gl, nloc)); Cc.append(np.tile(gl, nloc))
        mv.append(M.ravel()); kv.append(K.ravel())
    Rr, Cc = np.concatenate(Rr), np.concatenate(Cc)
    Mg = coo_matrix((np.concatenate(mv), (Rr, Cc)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (Rr, Cc)), shape=(ndof, ndof)).tocsr()
    w, U = eigsh(Kg, k=which+2, M=Mg, sigma=-1., which="LM")
    o = np.argsort(w)
    u = U[:, o[which]]
    mu = float(w[o[which]])

    lam = np.array([(a/nsub, b/nsub, 1-a/nsub-b/nsub)
                    for a in range(nsub+1) for b in range(nsub+1-a)])
    idx = {(round(l[0]*nsub), round(l[1]*nsub)): k for k, l in enumerate(lam)}
    pts, vals, faces = [], [], []
    npt = 0
    for t, T in enumerate(tris):
        X = (T @ lam.T).T
        X = X/np.sqrt(-(-X[:, 0]**2+X[:, 1]**2+X[:, 2]**2))[:, None]
        phi = bern_eval(d, bary(T, X))
        c = u[dof[t*nloc:(t+1)*nloc]]
        pts.append(X); vals.append(phi@c)
        for a in range(nsub):
            for b in range(nsub-a):
                faces.append([npt+idx[(a, b)], npt+idx[(a+1, b)], npt+idx[(a, b+1)]])
                if a+b < nsub-1:
                    faces.append([npt+idx[(a+1, b)], npt+idx[(a+1, b+1)],
                                  npt+idx[(a, b+1)]])
        npt += X.shape[0]
    return np.vstack(pts), np.concatenate(vals), np.array(faces), mu


def scene_mode(which, name, view="top"):
    P, v, F, mu = eigfield(2, 4, which)
    v = v/np.abs(v).max()
    new_fig()
    src = mlab.triangular_mesh(P[:, 1], P[:, 2], P[:, 0], F, scalars=v,
                               colormap="RdBu", vmin=-1, vmax=1)
    src.module_manager.scalar_lut_manager.reverse_lut = True
    # octagon boundary
    V0, gens, R = bolza_octagon()
    for j in range(8):
        A = geod(V0[j], V0[(j+1) % 8], 90)
        mlab.plot3d(A[:, 1], A[:, 2], A[:, 0]*1.004, color=EDGE,
                    tube_radius=0.018)
    if view == "top":
        mlab.view(azimuth=0, elevation=0.1, distance=13.0, focalpoint=(0, 0, 2.4))
    else:
        mlab.view(azimuth=30, elevation=55, distance=14.0, focalpoint=(0, 0, 2.4))
    save(name)
    return mu


# --------------------------------------------------- polygons for three surfaces
def scene_polygon(V0, n, name, pair_of=None):
    """Render a fundamental n-gon on the hyperboloid, sides shaded by pair."""
    new_fig()
    rmax = float(np.sqrt(max(V0[:, 0])**2-1))
    # interior region: fan triangles from the apex, subdivided
    O = np.array([1.0, 0.0, 0.0])
    nsub = 8
    lam = np.array([(a/nsub, b/nsub, 1-a/nsub-b/nsub)
                    for a in range(nsub+1) for b in range(nsub+1-a)])
    idx = {(round(l[0]*nsub), round(l[1]*nsub)): k for k, l in enumerate(lam)}
    pts, faces, npt = [], [], 0
    for j in range(n):
        T = np.column_stack([O, V0[j], V0[(j+1) % n]])
        X = (T @ lam.T).T
        X = X/np.sqrt(-(-X[:, 0]**2+X[:, 1]**2+X[:, 2]**2))[:, None]
        pts.append(X)
        for a in range(nsub):
            for b in range(nsub-a):
                faces.append([npt+idx[(a, b)], npt+idx[(a+1, b)], npt+idx[(a, b+1)]])
                if a+b < nsub-1:
                    faces.append([npt+idx[(a+1, b)], npt+idx[(a+1, b+1)],
                                  npt+idx[(a, b+1)]])
        npt += X.shape[0]
    P = np.vstack(pts)
    mlab.triangular_mesh(P[:, 1], P[:, 2], P[:, 0], np.array(faces), color=SURF)
    greys = [(0.10, 0.10, 0.12), (0.35, 0.35, 0.38), (0.55, 0.55, 0.58),
             (0.72, 0.72, 0.75)]
    for j in range(n):
        A = geod(V0[j], V0[(j+1) % n], 80)
        k = pair_of[j] if pair_of else j % 4
        mlab.plot3d(A[:, 1], A[:, 2], A[:, 0]*1.003,
                    color=greys[k % 4], tube_radius=0.022*rmax/3.0)
    for j in range(n):
        mlab.points3d(V0[j][1], V0[j][2], V0[j][0]*1.003, color=EDGE,
                      scale_factor=0.06*rmax/3.0)
    mlab.view(azimuth=0, elevation=0.1, distance=7.0*rmax,
              focalpoint=(0, 0, 0.62*np.sqrt(1+rmax**2)))
    save(name)


if __name__ == "__main__":
    scene_sphere(); print("sphere")
    scene_plane(); print("plane")
    scene_hyperboloid_model(); print("hyperboloid")
    scene_octagon_3d(); print("octagon3d")
    for L in (0, 1, 2):
        scene_mesh(L, f"m_mesh{L}.png"); print("mesh", L)
    mu1 = scene_mode(1, "m_mode1.png"); print("mode1", mu1)
    mu5 = scene_mode(5, "m_mode5.png"); print("mode5", mu5)
    np.save("mode_mus.npy", np.array([mu1, mu5]))

    from klein import surface, OPPOSITE, KLEIN14
    import nonor3
    V, g, p, info = surface(8, OPPOSITE(8))
    scene_polygon(V, 8, "m_poly_bolza.png",
                  pair_of=[j % 4 for j in range(8)]); print("poly bolza")
    V, g, p, info = surface(14, KLEIN14)
    pk = {}
    for k, (a, b) in enumerate(KLEIN14):
        pk[a] = k; pk[b] = k
    scene_polygon(V, 14, "m_poly_klein.png",
                  pair_of=[pk[j] for j in range(14)]); print("poly klein")
    r = nonor3.analyse(6, [(0, 1), (2, 3), (4, 5)], list("RRR"))
    scene_polygon(r["V"], 6, "m_poly_n3.png",
                  pair_of=[0, 0, 1, 1, 2, 2]); print("poly n3")

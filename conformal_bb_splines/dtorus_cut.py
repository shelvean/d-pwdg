"""Cutaway of the distorted torus: the surface with an azimuthal wedge removed,
so the meridian profile is seen face on, with the two cut curves drawn as
tubes and the centroid circle of the profile as the core."""
import numpy as np
from mayavi import mlab
mlab.options.offscreen = True
from dtorus import rho, zz, t_of, U

PHI0, PHI1 = 0.55, 1.75          # the wedge that is removed

def emb(u, phi):
    t = t_of(u); return np.array([rho(t)*np.cos(phi), rho(t)*np.sin(phi), zz(t)])

def render(out, nu=241, nphi=221, size=(1500, 1000), tube=0.030):
    us = np.linspace(0, U, nu)
    ps = np.linspace(PHI1, PHI0 + 2*np.pi, nphi)
    Ug, Pg = np.meshgrid(us, ps, indexing='ij')
    X = np.array([emb(u, p) for u, p in zip(Ug.ravel(), Pg.ravel())])
    tris = []
    for i in range(nu-1):
        for j in range(nphi-1):
            a, b, c, d = i*nphi+j, (i+1)*nphi+j, i*nphi+j+1, (i+1)*nphi+j+1
            tris += [(a, b, d), (a, d, c)]
    tris = np.array(tris)
    fig = mlab.figure(bgcolor=(1, 1, 1), size=size)
    s = mlab.triangular_mesh(X[:, 0], X[:, 1], X[:, 2], tris, color=(0.87, 0.89, 0.93))
    s.actor.property.specular = 0.12
    # cap the two cut faces with a disc coloured by the conformal factor, so the
    # shell reads as a solid section rather than an open tube
    for p in (PHI0, PHI1):
        nu2, nr = 240, 26
        uu = np.linspace(0, U, nu2)
        P = np.array([emb(u, p) for u in uu])
        ctr = P.mean(0)
        V = []; S = []
        for k in range(nr+1):
            lam = k/nr
            for q in range(nu2):
                V.append(ctr + lam*(P[q]-ctr)); S.append(rho(t_of(uu[q]))**2)
        V = np.array(V); S = np.array(S)
        F = []
        for k in range(nr):
            for q in range(nu2):
                a = k*nu2+q; b = k*nu2+(q+1) % nu2
                c = (k+1)*nu2+q; d = (k+1)*nu2+(q+1) % nu2
                F += [(a, b, d), (a, d, c)]
        cap = mlab.triangular_mesh(V[:, 0], V[:, 1], V[:, 2], np.array(F),
                                   color=(0.72, 0.74, 0.79))
        cap.actor.property.specular = 0.04

    # the two cut profiles, coloured by the conformal factor w = rho^2
    for p in (PHI0, PHI1):
        pts = np.array([emb(u, p) for u in np.linspace(0, U, 400)])
        w = np.array([rho(t_of(u))**2 for u in np.linspace(0, U, 400)])
        l = mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], w, colormap='RdBu',
                        tube_radius=tube, vmin=2.0, vmax=11.6)
        l.module_manager.scalar_lut_manager.reverse_lut = True
        prof = l
    # a few meridian lines for orientation
    for p in np.linspace(PHI1, PHI0 + 2*np.pi, 13)[1:-1]:
        pts = np.array([emb(u, p) for u in np.linspace(0, U, 200)])
        mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(0.25, 0.28, 0.42), tube_radius=0.010)
    cb = mlab.colorbar(prof, orientation='vertical', nb_labels=5, label_fmt='%.1f')
    cb.scalar_bar_representation.position = np.array([0.90, 0.28])
    cb.scalar_bar_representation.position2 = np.array([0.035, 0.44])
    cb.label_text_property.color = (0, 0, 0); cb.label_text_property.italic = False
    cb.label_text_property.font_family = 'times'; cb.title_text_property.color = (0, 0, 0)
    mlab.view(azimuth=66, elevation=60, distance='auto')
    fig.scene.camera.zoom(1.6)
    fig.scene.parallel_projection = True
    mlab.savefig(out, size=size); mlab.close(fig); print(out, flush=True)

if __name__ == '__main__':
    render('/home/claude/work/hyp/paper/p_dtorus_cut.png')

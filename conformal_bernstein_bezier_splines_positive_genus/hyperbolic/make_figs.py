"""Figures for the hyperbolic spline manuscript."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import FancyArrowPatch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from hypgeom import (bolza_octagon, geodesic_point, regular_polygon, dist,
                     triangle_area_defect, inv_lorentz)
from hypbb import bern_indices, bern_eval, bary, local_matrices, constant_coeffs
from hypmesh import octagon_mesh, build_dofs, domain_points
from klein import surface, OPPOSITE, KLEIN14

plt.rcParams.update({
    "font.size": 9, "axes.linewidth": 0.6, "figure.dpi": 200,
    "savefig.bbox": "tight", "font.family": "serif",
})
GREY = "0.25"


def poincare(x):
    x = np.atleast_2d(np.asarray(x, float))
    return x[:, 1:] / (1.0 + x[:, :1])


def geod_arc(a, b, n=80):
    t = np.linspace(0, 1, n)
    P = np.array([geodesic_point(a, b, s) for s in t])
    return poincare(P)


# ---------------------------------------------------------------- fig 1: models
def fig_models():
    fig = plt.figure(figsize=(7.6, 2.7))
    th = np.linspace(0, 2*np.pi, 120)

    ax = fig.add_subplot(131, projection="3d")
    u = np.linspace(0, np.pi, 60); v = np.linspace(0, 2*np.pi, 60)
    U, V = np.meshgrid(u, v)
    ax.plot_surface(np.sin(U)*np.cos(V), np.sin(U)*np.sin(V), np.cos(U),
                    color="0.85", alpha=0.55, linewidth=0, shade=True)
    tri = np.array([[0.0, 0.0, 1.0],
                    [np.sin(0.9), 0, np.cos(0.9)],
                    [np.sin(0.9)*np.cos(2.0), np.sin(0.9)*np.sin(2.0), np.cos(0.9)]])
    for i in range(3):
        a, b = tri[i], tri[(i+1) % 3]
        t = np.linspace(0, 1, 60)[:, None]
        c = a*(1-t) + b*t
        c = c/np.linalg.norm(c, axis=1)[:, None]
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color=GREY, lw=1.1)
    ax.set_title(r"$\kappa=+1$:  $\langle x,x\rangle_{+}=1$", pad=0)
    ax.set_axis_off(); ax.set_box_aspect([1, 1, 1]); ax.view_init(22, 35)

    ax = fig.add_subplot(132, projection="3d")
    g = np.linspace(-1.4, 1.4, 2)
    G1, G2 = np.meshgrid(g, g)
    ax.plot_surface(np.ones_like(G1), G1, G2, color="0.85", alpha=0.55, linewidth=0)
    P = np.array([[1, -0.9, -0.7], [1, 1.0, -0.6], [1, 0.1, 1.0], [1, -0.9, -0.7]])
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=GREY, lw=1.1)
    ax.set_title(r"$\kappa=0$:  $x_0=1$", pad=0)
    ax.set_axis_off(); ax.set_box_aspect([1, 1, 1]); ax.view_init(22, 35)

    ax = fig.add_subplot(133, projection="3d")
    r = np.linspace(0, 1.5, 40); ph = np.linspace(0, 2*np.pi, 60)
    R, PH = np.meshgrid(r, ph)
    ax.plot_surface(np.sqrt(1+R**2), R*np.cos(PH), R*np.sin(PH),
                    color="0.85", alpha=0.55, linewidth=0)
    ax.plot_wireframe(R[::12, ::10], (R*np.cos(PH))[::12, ::10],
                      (R*np.sin(PH))[::12, ::10], color="0.7", lw=0.3)
    V0, gens, Rc = bolza_octagon()
    T = [np.array([1., 0, 0]), V0[0], V0[1]]
    for i in range(3):
        A = np.array([geodesic_point(T[i], T[(i+1) % 3], s) for s in np.linspace(0, 1, 60)])
        ax.plot(A[:, 0], A[:, 1], A[:, 2], color=GREY, lw=1.1)
    ax.set_title(r"$\kappa=-1$:  $\langle x,x\rangle_{-}=-1$", pad=0)
    ax.set_axis_off(); ax.set_box_aspect([1, 1, 1]); ax.view_init(18, 35)
    fig.savefig("fig_models.png")
    plt.close(fig)


# ------------------------------------------------- fig 2: hyperboloid and Klein
def fig_hyperboloid():
    fig = plt.figure(figsize=(7.2, 3.2))
    V0, gens, Rc = bolza_octagon()
    O = np.array([1., 0, 0])

    ax = fig.add_subplot(121, projection="3d")
    r = np.linspace(0, 2.0, 60); ph = np.linspace(0, 2*np.pi, 90)
    R, PH = np.meshgrid(r, ph)
    ax.plot_surface(R*np.cos(PH), R*np.sin(PH), np.sqrt(1+R**2),
                    color="0.87", alpha=0.55, linewidth=0)
    ax.plot_wireframe((R*np.cos(PH))[::14, ::12], (R*np.sin(PH))[::14, ::12],
                      np.sqrt(1+R**2)[::14, ::12], color="0.65", lw=0.3)
    rc = np.linspace(0, 2.0, 2)
    Rc2, PH2 = np.meshgrid(rc, ph)
    ax.plot_surface(Rc2*np.cos(PH2), Rc2*np.sin(PH2), Rc2,
                    color="0.5", alpha=0.10, linewidth=0)
    a = 1.05
    tri = [np.array([np.sqrt(1+a**2)*0 + 1.0, 0.0, 0.0]),
           np.array([np.sqrt(1+a**2), a*np.cos(0.0), a*np.sin(0.0)]),
           np.array([np.sqrt(1+a**2), a*np.cos(1.9), a*np.sin(1.9)])]
    for i in range(3):
        A = np.array([geodesic_point(tri[i], tri[(i+1) % 3], s)
                      for s in np.linspace(0, 1, 80)])
        ax.plot(A[:, 1]*1.02, A[:, 2]*1.02, A[:, 0]*1.02, color="0.05", lw=1.8, zorder=10)
    for v in tri:
        ax.plot([0, 1.9*v[1]], [0, 1.9*v[2]], [0, 1.9*v[0]], color="0.45",
                lw=0.6, ls=":")
        ax.plot([v[1]], [v[2]], [v[0]], "o", ms=3.0, color="0.1")
    ax.text(1.5, 1.5, 0.5, "light cone", fontsize=7.5, color="0.4")
    ax.text(0.0, 0.0, 2.55, r"$\langle x,x\rangle=-1$", fontsize=8, color="0.25")
    ax.set_axis_off(); ax.view_init(32, 35); ax.set_box_aspect([1, 1, 0.8])
    ax.set_title("hyperboloid model: a geodesic triangle\nand its cone over the origin",
                 fontsize=8.5, pad=-2)

    ax = fig.add_subplot(122)
    K = np.array([[v[1]/v[0], v[2]/v[0]] for v in V0])
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.75", lw=0.8))
    ax.plot(np.append(K[:, 0], K[0, 0]), np.append(K[:, 1], K[0, 1]),
            color="0.15", lw=1.4)
    for j in range(8):
        A = geod_arc(V0[j], V0[(j+1) % 8])
        ax.plot(A[:, 0], A[:, 1], color="0.5", lw=1.1, ls="--")
    ax.text(0.0, 0.30, "Klein: straight chords", ha="center", fontsize=8, color="0.15")
    ax.text(0.0, 0.05, "Poincare: circular arcs", ha="center", fontsize=8, color="0.45")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("the same octagon in two disk models", fontsize=8.5)
    fig.savefig("fig_hyperboloid.png")
    plt.close(fig)


# ---------------------------------------------------------- fig 3: the octagon
def fig_octagon():
    V0, gens, Rc = bolza_octagon()
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.75", lw=0.8))
    cols = ["0.15", "0.35", "0.55", "0.7"]
    for j in range(8):
        A = geod_arc(V0[j], V0[(j+1) % 8])
        ax.plot(A[:, 0], A[:, 1], color=cols[j % 4], lw=2.0)
        m = A[len(A)//2]
        lab = f"$s_{j}$"
        ax.text(m[0]*1.13, m[1]*1.13, lab, ha="center", va="center", fontsize=8)
    P = np.array([[v[1]/(1+v[0]), v[2]/(1+v[0])] for v in V0])
    ax.plot(P[:, 0], P[:, 1], "o", ms=3.4, color="0.1")
    ax.text(0, 0, "all 8 vertices\nare one point", ha="center", va="center",
            fontsize=7.5, color="0.3")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(r"Bolza octagon: $s_j \sim s_{j+4}$,  angle $\pi/4$,  area $4\pi$",
                 fontsize=8.5)
    fig.savefig("fig_octagon.png")
    plt.close(fig)


# ------------------------------------------------------------ fig 4: the meshes
def fig_mesh():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for ax, L in zip(axes, (0, 1, 2)):
        tris, V0, gens = octagon_mesh(L)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.8", lw=0.6))
        for T in tris:
            for i in range(3):
                A = geod_arc(T[:, i], T[:, (i+1) % 3], 24)
                ax.plot(A[:, 0], A[:, 1], color="0.35", lw=0.45)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"level {L}:  {len(tris)} triangles", fontsize=8.5)
    fig.savefig("fig_mesh.png")
    plt.close(fig)


# ------------------------------------------------- eigenfunction evaluation
def spline_field(level, d, which=1, nsub=6, npairs=OPPOSITE(8), n=8):
    tris, V0, gens = octagon_mesh(level)
    dof, _ = build_dofs(tris, V0, gens, d)
    nloc = len(bern_indices(d))
    ndof = int(dof.max())+1
    R, C, mv, kv = [], [], [], []
    for t, T in enumerate(tris):
        M, K = local_matrices(T, d, n=14)
        gl = dof[t*nloc:(t+1)*nloc]
        R.append(np.repeat(gl, nloc)); C.append(np.tile(gl, nloc))
        mv.append(M.ravel()); kv.append(K.ravel())
    R, C = np.concatenate(R), np.concatenate(C)
    Mg = coo_matrix((np.concatenate(mv), (R, C)), shape=(ndof, ndof)).tocsr()
    Kg = coo_matrix((np.concatenate(kv), (R, C)), shape=(ndof, ndof)).tocsr()
    w, U = eigsh(Kg, k=which+2, M=Mg, sigma=-1., which="LM")
    o = np.argsort(w)
    u = U[:, o[which]]
    pts, vals, faces = [], [], []
    npt = 0
    lam = []
    for a in range(nsub+1):
        for b in range(nsub+1-a):
            lam.append((a/nsub, b/nsub, 1-a/nsub-b/nsub))
    lam = np.array(lam)
    idx = {(round(l[0]*nsub), round(l[1]*nsub)): k for k, l in enumerate(lam)}
    for t, T in enumerate(tris):
        X = (T @ lam.T).T
        X = X/np.sqrt(-(-X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2))[:, None]
        phi = bern_eval(d, bary(T, X))
        c = u[dof[t*nloc:(t+1)*nloc]]
        base = npt
        npt += X.shape[0]
        pts.append(poincare(X)); vals.append(phi @ c)
        for a in range(nsub):
            for b in range(nsub-a):
                faces.append([base+idx[(a, b)], base+idx[(a+1, b)], base+idx[(a, b+1)]])
                if a+b < nsub-1:
                    faces.append([base+idx[(a+1, b)], base+idx[(a+1, b+1)],
                                  base+idx[(a, b+1)]])
    return (np.vstack(pts), np.concatenate(vals), np.array(faces),
            float(w[o[which]]), tris, V0)


def fig_modes():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for ax, which in zip(axes, (1, 5)):
        P, v, F, mu, tris, V0 = spline_field(2, 4, which=which)
        v = v/np.abs(v).max()
        tr = Triangulation(P[:, 0], P[:, 1], F)
        ax.tripcolor(tr, v, cmap="RdBu_r", shading="gouraud",
                     vmin=-1, vmax=1, rasterized=True)
        for j in range(8):
            A = geod_arc(V0[j], V0[(j+1) % 8])
            ax.plot(A[:, 0], A[:, 1], color="0.2", lw=1.0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(rf"$\mu = {mu:.6f}$", fontsize=9)
    fig.savefig("fig_modes.png")
    plt.close(fig)


def fig_convergence():
    lap = {2: [(1022, 6.19e-4)], 4: [(1022, 1.29e-5)], 6: [(1022, 1.66e-8)]}
    lapfull = {2: [(62, 1.22e-1), (254, 8.93e-3), (1022, 6.19e-4)],
               4: [(254, 3.45e-3), (1022, 1.29e-5), (4094, 5.70e-8)],
               6: [(574, 6.68e-5), (2302, 1.66e-8), (9214, 4.54e-12)]}
    bih = {4: [(254, 2.69e-1), (1022, 1.10e-2), (4094, 5.31e-4)],
           6: [(574, 8.56e-3), (2302, 1.76e-5), (9214, 3.22e-8)]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, data, ttl in ((axes[0], lapfull, r"$-\Delta u=\mu u$   ($C^0$, rate $2d$)"),
                          (axes[1], bih, r"$\Delta^2 u=\mu u$   ($C^1$, rate $2(d-1)$)")):
        for d, rows in data.items():
            n = [r[0] for r in rows]; e = [r[1] for r in rows]
            ax.loglog(n, e, "o-", color=GREY, lw=1.0, ms=3.2)
            ax.annotate(rf"$d={d}$", (n[-1], e[-1]), textcoords="offset points",
                        xytext=(6, -1), fontsize=8)
        ax.set_xlabel("degrees of freedom"); ax.grid(True, which="both", lw=0.25, alpha=0.5)
        ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel(r"relative error in $\mu_1$")
    fig.savefig("fig_convergence.png")
    plt.close(fig)


def fig_selberg():
    t = [1.0, 0.7, 0.5, 0.3, 0.2]
    r = [3.50e-4, 9.14e-6, 1.31e-8, 2.19e-7, 5.31e-7]
    share = [0.333, 0.129, 0.031, 1.1e-3, 1.8e-5]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.semilogy(t, r, "o-", color=GREY, lw=1.0, ms=3.4)
    ax.set_xlabel(r"$t$ in $h(r)=e^{-tr^2}$")
    ax.set_ylabel("relative difference of the two sides")
    ax.grid(True, which="both", lw=0.25, alpha=0.5)
    ax.annotate("geodesic truncation", (0.95, 3.5e-4), fontsize=7.5,
                ha="right", va="bottom", color="0.3")
    ax.annotate("spectral discretisation", (0.22, 5.5e-7), fontsize=7.5,
                ha="left", va="bottom", color="0.3")
    fig.savefig("fig_selberg.png")
    plt.close(fig)


def fig_polygons():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    specs = [(8, OPPOSITE(8), "Bolza, genus 2"),
             (14, KLEIN14, "Klein quartic, genus 3"),
             (6, [(0, 1), (2, 3), (4, 5)], r"$N_3$, nonorientable")]
    for ax, (n, pairs, ttl) in zip(axes, specs):
        if n == 6:
            import nonor3
            r = nonor3.analyse(6, [(0, 1), (2, 3), (4, 5)], list("RRR"))
            V0 = r["V"]
        else:
            V0, g, p, info = surface(n, pairs)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.8", lw=0.7))
        style = {}
        for k, (a, b) in enumerate(pairs):
            for s in (a, b):
                A = geod_arc(V0[s], V0[(s+1) % n])
                ax.plot(A[:, 0], A[:, 1], lw=1.8,
                        color=plt.cm.Greys(0.35 + 0.55*(k % 4)/4))
            for s, lbl in ((a, chr(97+k)), (b, chr(97+k))):
                A = geod_arc(V0[s], V0[(s+1) % n])
                m = A[len(A)//2]
                ax.text(m[0]*1.15, m[1]*1.15, lbl, ha="center", va="center", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(ttl, fontsize=8.5)
    fig.savefig("fig_polygons.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_models(); print("models")
    fig_hyperboloid(); print("hyperboloid")
    fig_octagon(); print("octagon")
    fig_mesh(); print("mesh")
    fig_convergence(); print("convergence")
    fig_selberg(); print("selberg")
    fig_polygons(); print("polygons")
    fig_modes(); print("modes")

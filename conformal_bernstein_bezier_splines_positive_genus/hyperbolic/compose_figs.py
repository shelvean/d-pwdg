"""Compose mayavi renders into the paper's figure panels with serif labels."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "savefig.bbox": "tight", "figure.dpi": 200})


def crop(img, pad=0.02):
    """Trim white margins."""
    a = img[..., :3].min(axis=2)
    rows = np.where(a.min(axis=1) < 0.985)[0]
    cols = np.where(a.min(axis=0) < 0.985)[0]
    if len(rows) == 0:
        return img
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    pr = int(pad*(r1-r0)); pc = int(pad*(c1-c0))
    return img[max(0, r0-pr):r1+pr, max(0, c0-pc):c1+pc]


def panel(ax, fname, title):
    ax.imshow(crop(mpimg.imread(fname)))
    ax.axis("off")
    ax.set_title(title, fontsize=9.5)


# fig_models: the three quadrics
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
panel(axes[0], "m_sphere.png", r"$\kappa=+1$:  $\langle x,x\rangle_{+}=1$")
panel(axes[1], "m_plane.png", r"$\kappa=0$:  $x_0=1$")
panel(axes[2], "m_hyperboloid.png", r"$\kappa=-1$:  $\langle x,x\rangle_{-}=-1$")
fig.savefig("fig_models.png"); plt.close(fig)

# fig_hyperboloid: model close-up + octagon on the sheet
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
panel(axes[0], "m_hyperboloid.png",
      "a geodesic triangle on the hyperboloid,\nwith its cone over the origin")
panel(axes[1], "m_octagon3d.png",
      "the Bolza fundamental octagon\non the sheet $\\langle x,x\\rangle=-1$")
fig.savefig("fig_hyperboloid.png"); plt.close(fig)

# fig_octagon: the octagon seen along the x0 axis
fig, ax = plt.subplots(figsize=(3.9, 3.9))
panel(ax, "m_poly_bolza.png",
      "Bolza octagon: $s_j\\sim s_{j+4}$,  angle $\\pi/4$,  area $4\\pi$")
fig.savefig("fig_octagon.png"); plt.close(fig)

# fig_mesh: three refinement levels
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
for ax, L, nt in zip(axes, (0, 1, 2), (8, 32, 128)):
    panel(ax, f"m_mesh{L}.png", f"level {L}:  {nt} triangles")
fig.savefig("fig_mesh.png"); plt.close(fig)

# fig_modes: eigenfunctions
mus = np.load("mode_mus.npy")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
panel(axes[0], "m_mode1.png", rf"$\mu={mus[0]:.6f}$   (multiplicity 3)")
panel(axes[1], "m_mode5.png", rf"$\mu={mus[1]:.6f}$   (multiplicity 4)")
fig.savefig("fig_modes.png"); plt.close(fig)

# fig_polygons: three surfaces
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
panel(axes[0], "m_poly_bolza.png", "Bolza, genus 2")
panel(axes[1], "m_poly_klein.png", "Klein quartic, genus 3")
panel(axes[2], "m_poly_n3.png", r"$N_3$, nonorientable")
fig.savefig("fig_polygons.png"); plt.close(fig)

print("composed")

"""Figures for the JCP manuscript. All plots are computed from the
closed-form expressions in the paper; no data file is required.
Fonts are rendered by LaTeX through the pgf backend."""
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from scipy.special import jv, jvp, hankel1, h1vp
from pathlib import Path

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{amsmath,amssymb}",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 9,
})
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "figures"; out.mkdir(exist_ok=True)
# shared muted diverging colormap for all field visualizations.
# We soften the standard RdBu_r map by trimming the harsh extremes and blending
# every color 30% toward white.  This gives Figs. 3 and 4 the same visual
# standard without changing the underlying data normalization.
from matplotlib.colors import LinearSegmentedColormap
_base = plt.get_cmap("RdBu_r")(np.linspace(0.04, 0.96, 256))
_base[:, :3] = 0.70 * _base[:, :3] + 0.30
CM = LinearSegmentedColormap.from_list("mutedRdBu_shared", _base)

k, kh = 16.0, 3.34
h = kh / k

# ---- trace weights tau_m^2 (eq. T1) ---------------------------------------
m = np.arange(0, 31)
tau2 = 2 * np.pi * h * k * (jv(m, kh) ** 2 + jvp(m, kh) ** 2)

# ---- equispaced PW Gram condition number (eq. T2-eigs) --------------------
M = 200
mm = np.arange(-M, M + 1)
t2 = 2 * np.pi * h * k * (jv(mm, kh) ** 2 + jvp(mm, kh) ** 2)
qs = np.arange(3, 40)
cond = []
for q in qs:
    lam = np.array([q * t2[(-mm) % q == s].sum() for s in range(q)])
    cond.append(lam.max() / lam.min())
cond = np.array(cond)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.35))
a1.semilogy(m, tau2 / tau2.max(), "k.-", lw=0.9, ms=3.5)
a1.set_xlabel(r"angular order $m$")
a1.set_ylabel(r"$\tau_m^2/\max_j\tau_j^2$")
a1.set_title(r"(a) Fourier--Bessel trace weights")
a1.axhline(1e-16, color="0.5", lw=0.7, ls="--")
a1.text(1, 3e-16, r"$\varepsilon_{\mathrm{mach}}$", color="0.3", fontsize=8)
a2.semilogy(qs, cond, "k.-", lw=0.9, ms=3.5)
a2.set_xlabel(r"number of equispaced directions $q$")
a2.set_ylabel(r"$\mathrm{cond}\,G_q^{\mathrm{PW}}$")
a2.set_title(r"(b) plane-wave trace Gram")
a2.axhline(1e16, color="0.5", lw=0.7, ls="--")
a2.text(4, 3e16, r"$\varepsilon_{\mathrm{mach}}^{-1}$", color="0.3", fontsize=8)
for a in (a1, a2):
    a.grid(True, which="major", lw=0.3, color="0.85")
fig.tight_layout(w_pad=1.5)
fig.savefig(out / "modal_weights_and_pw_gram_k16.pdf")
plt.close(fig)

# ---- local coordinates (reconstruction; archived exact panel is retained) ----
# The manuscript uses figures/basis_visuals_k16.pdf recovered from the earlier
# manuscript version.  This block writes a separately named reconstruction so
# rerunning the figure script cannot overwrite the restored archived figure.
alpha = np.deg2rad(35.0)
xx = np.linspace(-1, 1, 361)
X, Y = np.meshgrid(xx, xx)
r = np.hypot(X, Y); th = np.arctan2(Y, X)
disk = r <= 1.0
psi = np.exp(1j * k * (np.cos(alpha) * X + np.sin(alpha) * Y))
phi4 = jv(4, k * r) * np.exp(4j * (th - alpha))
# Restore the earlier pictures exactly in content: the hybrid panel is the
# physical 0.6 PW + 0.4 FB superposition, rather than a mixture of separately
# trace-normalized coordinates.  Each panel is still scaled independently for
# visualization, as stated in the earlier manuscript caption.
panels = [
    (np.real(psi), r"(a) normalized plane wave"),
    (0.6 * np.real(psi) + 0.4 * np.real(phi4), r"(b) hybrid superposition"),
    (np.real(phi4), r"(c) normalized Fourier--Bessel mode"),
]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))
tcirc = np.linspace(0, 2 * np.pi, 400)
for a, (Z, ttl) in zip(axs, panels):
    Zm = np.where(disk, Z, np.nan)
    vmax = np.nanmax(np.abs(Zm))
    a.pcolormesh(X, Y, Zm, cmap=CM, vmin=-vmax, vmax=vmax,
                 shading="auto", rasterized=True)
    a.plot(np.cos(tcirc), np.sin(tcirc), "k-", lw=0.7)
    a.set_aspect("equal")
    a.set_title(ttl, fontsize=9)
    a.set_xticks([-1, -0.5, 0, 0.5, 1])
    a.set_yticks([-1, -0.5, 0, 0.5, 1])
    a.set_xlabel("$x$")
    a.set_ylabel("$y$")
    a.set_xlim(-1.02, 1.02); a.set_ylim(-1.02, 1.02)
# direction arrow and angle label on the hybrid panel
ax = axs[1]
L = 0.68
ax.annotate("", xy=(L * np.cos(alpha), L * np.sin(alpha)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0, mutation_scale=9))
ax.plot([0, 0.30], [0, 0], "k-", lw=0.6)
arc = np.linspace(0, alpha, 30)
ax.plot(0.22 * np.cos(arc), 0.22 * np.sin(arc), "k-", lw=0.6)
ax.text(0.27, 0.07, r"$\alpha$", fontsize=8)
ax.text(-0.72, -0.56, r"hybrid PW + FB", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70))
fig.suptitle(r"Local Trefftz coordinates at $k=16$, $\alpha=35^\circ$", fontsize=10, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=0.8)
fig.savefig(out / "basis_visuals_k16_regenerated.pdf", dpi=220)
plt.close(fig)

# ---- outgoing Herglotz field (eq. 72, density 74) with selected counts ------
a_in, R = 0.5, 1.0
Mtr = 70
nphi = 4096
phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
g = np.exp(1.55 * np.cos(phi - 0.48) + 0.22 * np.cos(2 * phi + 0.35)) * \
    np.exp(1j * (0.28 * np.sin(3 * phi + 0.20) + 0.10 * np.cos(5 * phi - 0.40)))
ms = np.arange(-Mtr, Mtr + 1)
ghat = np.array([(g * np.exp(-1j * mi * phi)).mean() for mi in ms])
cm = 2 * np.pi * (1j ** ms) * jv(ms, k * a_in) * ghat / hankel1(ms, k * a_in)
xx = np.linspace(-1, 1, 361)
X, Y = np.meshgrid(xx, xx)
r = np.hypot(X, Y); th = np.arctan2(Y, X)
mask = (r >= a_in) & (r <= R)
U = np.zeros_like(X, dtype=complex)
rr = r[mask]; tt = th[mask]
for c, mi in zip(cm, ms):
    U[mask] += c * hankel1(mi, k * rr) * np.exp(1j * mi * tt)
Ur = np.where(mask, np.real(U), np.nan)
sel = [(3, 18), (4, 17), (3, 18), (1, 20), (4, 17), (3, 18), (3, 18), (1, 20)]
fig, ax = plt.subplots(figsize=(3.6, 3.3))
vm = np.nanmax(np.abs(Ur))
pc = ax.pcolormesh(X, Y, Ur, cmap=CM, vmin=-vm, vmax=vm, shading="auto", rasterized=True)
tline = np.linspace(0, 2 * np.pi, 400)
for rad in (a_in, R):
    ax.plot(rad * np.cos(tline), rad * np.sin(tline), "k-", lw=0.7)
for j in range(8):
    ang = np.deg2rad(45 * j)
    ax.plot([a_in * np.cos(ang), R * np.cos(ang)], [a_in * np.sin(ang), R * np.sin(ang)], "k-", lw=0.5)
    cang = np.deg2rad(22.5 + 45 * j); rc = 0.76
    ax.text(rc * np.cos(cang), rc * np.sin(cang), r"$(%d,%d)$" % sel[j], ha="center", va="center",
            fontsize=7.5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.4", lw=0.4))
ax.set_aspect("equal"); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04); cb.set_label(r"$\Re u$")
fig.tight_layout()
fig.savefig(out / "herglotz_dtn_selection_k16.pdf", dpi=200)
plt.close(fig)
print("done")

# ---- two-medium transmission field (eq. transmission) ----------------------
n1, n2, om = 2.0, 1.0, 12.0
k1, k2 = om * n1, om * n2
xx = np.linspace(-1, 1, 401); X, Y = np.meshgrid(xx, xx)
def tfield(thi_deg):
    thi = np.deg2rad(thi_deg)
    xi, eta1 = k1 * np.cos(thi), k1 * np.sin(thi)
    eta2 = np.sqrt(k2**2 - xi**2 + 0j)
    if eta2.imag < 0: eta2 = -eta2
    Rc = (eta1 - eta2) / (eta1 + eta2); Tc = 2 * eta1 / (eta1 + eta2)
    U = np.where(Y < 0, np.exp(1j * (xi * X + eta1 * Y)) + Rc * np.exp(1j * (xi * X - eta1 * Y)),
                 Tc * np.exp(1j * (xi * X + eta2 * Y)))
    return U, xi, eta2
U69, xi69, _ = tfield(69.0)
U29, xi29, eta29 = tfield(29.0)
fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
vm = max(np.abs(U69).max(), np.abs(U29).max())
panels = [(np.real(U69), r"(a) $\Re u$, $\theta_i=69^\circ$ (propagating)"),
          (np.real(U29), r"(b) $\Re u$, $\theta_i=29^\circ$ (total reflection)")]
for a, (Z, ttl) in zip(axs, panels):
    a.pcolormesh(X, Y, Z, cmap=CM, vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    for v in (-1, 0, 1):
        a.plot([v, v], [-1, 1], "k-", lw=0.4); a.plot([-1, 1], [v, v], "k-", lw=0.4)
    for cx in (-1, 0):
        for cy in (-1, 0):
            a.plot([cx, cx + 1], [cy, cy + 1], "k-", lw=0.4)
    a.plot([-1, 1], [0, 0], "k-", lw=1.0)
    a.set_aspect("equal"); a.set_title(ttl, fontsize=8)
    a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1]); a.set_xlabel("$x$")
axs[0].set_ylabel("$y$")
thi = np.deg2rad(69.0); thr = np.deg2rad(291.0); tht = np.arccos(xi69 / k2)
for ang, x0, y0 in [(thi, -0.75, -0.75), (thr, -0.75, -0.2), (tht, -0.75, 0.3)]:
    axs[0].annotate("", xy=(x0 + 0.4 * np.cos(ang), y0 + 0.4 * np.sin(ang)), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2, mutation_scale=9))
fig.tight_layout(w_pad=0.8)
fig.savefig(out / "transmission_field.pdf", dpi=200)
plt.close(fig)
print("transmission done; TIR decay rate Im eta2 =", eta29.imag)

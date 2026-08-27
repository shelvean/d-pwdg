"""Regenerate the three panels of the centered-Hankel high-p figure
(global_error_pw_hybrid.pdf, retained_rank_vs_p_full.pdf,
global_kappaGR_vs_p_full.pdf) from the shipped CSV data, with LaTeX
(Computer Modern) fonts and direct labels instead of legends."""
import csv, pathlib
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
})

root = pathlib.Path(__file__).resolve().parents[1]
fig_dir = root / "figures"

def col(rows, k, cast=float):
    return [cast(r[k]) for r in rows]

pw = list(csv.DictReader(open(root / "data" / "highp_global_dtn_pwdg.csv")))
hy = list(csv.DictReader(open(root / "data" / "hybrid_selector_global_results.csv")))

p_pw, e_pw = col(pw, "p"), col(pw, "E_L2")
r_pw = col(pw, "local_rank")
kgr_pw = col(pw, "kappa_GR")
p_hy, e_hy = col(hy, "p"), col(hy, "E_L2")

# Panel 1: PW-only vs selected PW-FB global error
fig, ax = plt.subplots(figsize=(3.1, 2.6))
ax.semilogy(p_pw, e_pw, "o-", color="0.15", ms=3.5, lw=1.0)
ax.semilogy(p_hy, e_hy, "s--", color="tab:blue", ms=3.5, lw=1.0)
ax.annotate("PW only", xy=(70, 4e-8), color="0.15")
ax.annotate("PW--FB selector", xy=(31, 2e-11), color="tab:blue")
ax.set_xlabel(r"local budget $p$")
ax.set_ylabel(r"relative $L^2$ error")
ax.grid(True, which="both", lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig(fig_dir / "global_error_pw_hybrid.pdf")
plt.close(fig)

# Panel 2: retained local rank, PW-only sweep
fig, ax = plt.subplots(figsize=(3.1, 2.6))
ax.plot(p_pw, r_pw, "o-", color="0.15", ms=3.5, lw=1.0)
ax.axhline(37, color="0.6", lw=0.7, ls=":")
ax.annotate(r"$r_K=37$", xy=(12, 38.2), color="0.35")
ax.set_xlabel(r"nominal directions $p$")
ax.set_ylabel(r"retained local rank $r_K$")
ax.set_ylim(5, 42)
ax.grid(True, lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig(fig_dir / "retained_rank_vs_p_full.pdf")
plt.close(fig)

# Panel 3: graph-Riesz condition number, PW-only sweep
fig, ax = plt.subplots(figsize=(3.1, 2.6))
ax.plot(p_pw, kgr_pw, "o-", color="0.15", ms=3.5, lw=1.0)
ax.set_xlabel(r"nominal directions $p$")
ax.set_ylabel(r"$\kappa_{\mathrm{GR}}=\kappa_2(B^*K_hB)$")
ax.set_ylim(0, 8)
ax.grid(True, lw=0.3, alpha=0.4)
fig.tight_layout()
fig.savefig(fig_dir / "global_kappaGR_vs_p_full.pdf")
plt.close(fig)
print("wrote three panels")

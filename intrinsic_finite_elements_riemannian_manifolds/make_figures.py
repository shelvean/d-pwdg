import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10,
                     "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in"})

# ---------------------------------------------------------------- degree convergence
p = np.arange(4, 11)
t1 = np.array([2.04076734146, 2.03952275577, 2.039312153907, 2.039275710327,
               2.039269211748, 2.039268078713, 2.039267857103])
t2 = np.array([2.04099789284, 2.03972353058, 2.039333792565, 2.039283753358,
               2.039270320565, 2.039268406858, 2.039267904929])
mean = (t1 + t2) / 2
dif = np.abs(np.diff(mean))
gap = t2 - t1

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.semilogy(p[1:], dif, marker="o", ls="-", color="0.15", ms=5, mfc="white",
            label=r"change of the mean, $|\bar\lambda_p-\bar\lambda_{p-1}|$")
ax.semilogy(p, gap, marker="s", ls="--", color="0.45", ms=5,
            label=r"gap between the two triplets")
ax.set_xlabel(r"polynomial degree $p$")
ax.set_ylabel(r"size")
ax.set_xticks(p)
ax.grid(True, which="major", lw=0.3, color="0.85")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("figures/fig_convergence.pdf")

# ---------------------------------------------------------------- trace-formula check
B = pickle.load(open("run/booker_curves.pkl", "rb"))
r, f = B["r"], B["f"]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(r, f[40], ls="--", color="0.5", lw=1.1, label=r"$N=40$")
ax.plot(r, f[640], ls="-", color="0.1", lw=1.3, label=r"$N=640$")
ax.axhline(6.0, color="0.6", lw=0.7, ls=":")
t1fem = np.sqrt(2.0392678)
ax.axvline(t1fem, color="0.3", lw=0.8, ls="-.")
ax.annotate(r"finite element $t_1$", xy=(t1fem, 6.0), xytext=(1.4257, 6.045),
            fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.5, color="0.3"))
ax.set_xlabel(r"spectral parameter $r$")
ax.set_ylabel(r"$f(r)$")
ax.set_xlim(r[0], r[-1])
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
fig.tight_layout()
fig.savefig("figures/fig_booker.pdf")

# ---------------------------------------------------------------- threshold in N
N = np.array([28, 56, 96, 128, 192, 256, 320, 448, 640])
lam6 = np.array([2.0385746878, 2.0390370688, 2.0391470540, 2.0391726092, 2.0391911454,
                 2.0391977104, 2.0392007682, 2.0392034439, 2.0392048721])
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.semilogx(N, lam6, marker="^", ls="-", color="0.15", ms=5, mfc="white",
            label=r"threshold $\lambda_{(6)}$, cutoff $R=7$")
ax.axhline(2.0392678, color="0.45", ls="--", lw=0.9, label=r"finite element value")
ax.set_xlabel(r"matrix size $N$")
ax.set_ylabel(r"$\lambda$")
ax.ticklabel_format(axis="y", useOffset=False, style="plain")
ax.legend(frameon=False, loc="lower right", fontsize=8.5)
ax.grid(True, which="major", lw=0.3, color="0.85")
fig.tight_layout()
fig.savefig("figures/fig_threshold.pdf")
print("figures written")

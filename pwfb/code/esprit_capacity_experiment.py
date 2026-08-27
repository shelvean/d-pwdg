"""Finite-direction capacity experiment for target-frequency ESPRIT.

The first 20 directions and coefficient phases reproduce the nested finite-ray
family used in Kapita (2026), arXiv:2608.18380.  The sequence is extended to
41 directions by a deterministic maximin rule on a 0.1-degree grid.  ESPRIT is
fed N=81 consecutive phase-corrected modal coefficients gamma_m directly, so a
balanced 41 x 41 Hankel matrix supports at most q=40 rays under the exact
recovery theorem.  We also force q=41 once, outside the theorem, to record
the numerical signature of the sample-count failure.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{amsmath,amssymb}",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 9,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

OLD_DIR = np.array([
    17, 123, 251, 68, 191, 315, 39, 158, 286, 340,
    92, 224, 5, 145, 273, 327, 54, 178, 235, 301
], dtype=float)
OLD_PHASE = np.array([
    0, 0.37, -0.51, 0.81, -1.04, 1.33, -1.52, 0.22, 1.71, -0.93,
    0.58, -1.22, 1.48, -0.31, 0.96, -1.73, 0.11, 1.18, -0.77, 1.91
], dtype=float)


def extend_maximin(nmax=40, grid_step=0.1):
    """Extend OLD_DIR by repeatedly maximizing circular separation."""
    directions = list(OLD_DIR)
    candidates = np.arange(0.0, 360.0, grid_step)
    while len(directions) < nmax:
        d = np.asarray(directions)
        delta = np.abs(candidates[:, None] - d[None, :]) % 360.0
        sep = np.minimum(delta, 360.0 - delta).min(axis=1)
        j = int(np.argmax(sep))
        directions.append(float(candidates[j]))
        candidates = np.delete(candidates, j)
    return np.asarray(directions)


DIRECTIONS = extend_maximin(41)
PHASES = np.empty(41)
PHASES[:20] = OLD_PHASE
j = np.arange(20, 41)
PHASES[20:] = np.mod(0.731 * j + 0.173 * j * j, 2 * np.pi) - np.pi
COEFFICIENTS = np.exp(1j * PHASES)


def modal_sequence(q, N=81):
    z = np.exp(-1j * np.deg2rad(DIRECTIONS[:q]))
    m = np.arange(N)
    return np.sum(COEFFICIENTS[:q, None] * z[:, None] ** m[None, :], axis=0)


def esprit(gamma, q):
    N = len(gamma)
    L = (N + 1) // 2
    K = N - L + 1
    if L < q + 1 or K < q:
        raise ValueError("modal window does not contain enough samples for rank-q ESPRIT")
    H = np.empty((L, K), dtype=complex)
    for r in range(L):
        H[r, :] = gamma[r:r + K]
    U, s, _ = svd(H, full_matrices=False, lapack_driver="gesvd")
    Uq = U[:, :q]
    S = np.linalg.lstsq(Uq[:-1, :], Uq[1:, :], rcond=None)[0]
    return np.linalg.eigvals(S), s[q - 1] / s[0]


def forced_esprit_outside_sample_bound(gamma, q):
    """Force the ESPRIT shift fit when L=q, so U[:-1] has only q-1 rows.

    This is deliberately outside the exact-recovery hypothesis.  The returned
    values quantify the failure rather than defining a valid ESPRIT estimate.
    """
    N = len(gamma)
    L = (N + 1) // 2
    K = N - L + 1
    H = np.empty((L, K), dtype=complex)
    for r in range(L):
        H[r, :] = gamma[r:r + K]
    U, s, _ = svd(H, full_matrices=False, lapack_driver="gesvd")
    Uq = U[:, :q]
    A = Uq[:-1, :]
    B = Uq[1:, :]
    S = np.linalg.lstsq(A, B, rcond=None)[0]
    nodes = np.linalg.eigvals(S)
    return nodes, s[q - 1] / s[0], np.linalg.matrix_rank(A)


def matching_error_deg(nodes, truth_deg):
    estimate = (-np.angle(nodes) * 180.0 / np.pi) % 360.0
    truth = np.asarray(truth_deg) % 360.0
    delta = np.abs(estimate[:, None] - truth[None, :]) % 360.0
    delta = np.minimum(delta, 360.0 - delta)
    rows, cols = linear_sum_assignment(delta)
    errors = delta[rows, cols]
    return float(errors.max()), float(errors.mean())


rows = []
for q in range(1, 41):
    gamma = modal_sequence(q, 81)
    nodes, sigma_ratio = esprit(gamma, q)
    max_error, mean_error = matching_error_deg(nodes, DIRECTIONS[:q])
    rows.append((q, max_error, mean_error, sigma_ratio))

# One deliberately forced calculation at M=41.  The 41x41 signal Hankel
# matrix exists, but the shifted signal-subspace matrix has shape 40x41 and
# rank 40, so the ESPRIT shift is underdetermined.
gamma41 = modal_sequence(41, 81)
nodes41, sigma41_ratio, shift_rank41 = forced_esprit_outside_sample_bound(gamma41, 41)
maxerr41, meanerr41 = matching_error_deg(nodes41, DIRECTIONS[:41])
mod41 = np.abs(nodes41)
forced41 = (41, maxerr41, meanerr41, sigma41_ratio, float(mod41.min()), float(mod41.max()), shift_rank41)

csv_path = DATA / "esprit_capacity_N81_reproduced.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["M", "status", "max_angle_error_deg", "mean_angle_error_deg",
                "sigma_q_over_sigma_1", "node_modulus_min", "node_modulus_max", "shift_rank"])
    for q, maxe, meane, sr in rows:
        w.writerow([q, "valid", maxe, meane, sr, 1.0, 1.0, q])
    w.writerow([forced41[0], "forced_outside_sample_bound", forced41[1], forced41[2],
                forced41[3], forced41[4], forced41[5], forced41[6]])

M = np.array([r[0] for r in rows])
err = np.array([r[1] for r in rows])
sig = np.array([r[3] for r in rows])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.55))
ax1.semilogy(M, err, "k.-", lw=0.9, ms=3.5)
ax1.axvline(19, color="0.55", ls="--", lw=0.8)
ax1.set_xlabel(r"number of rays $q$")
ax1.set_ylabel(r"maximum angle error (degrees)")
ax1.set_title(r"(a) direct ESPRIT recovery")
ax1.plot([41], [maxerr41], "kx", ms=5.5, mew=1.0)
ax1.set_xlim(1, 42)
ax1.set_xticks([1, 10, 20, 30, 40])
ax1.grid(True, which="major", lw=0.3, color="0.85")
ax1.text(18.3, 5e-12, r"residual path: 19", rotation=90,
         ha="right", va="bottom", fontsize=7.5, color="0.35")

ax2.semilogy(M, sig, "k.-", lw=0.9, ms=3.5)
ax2.axvline(19, color="0.55", ls="--", lw=0.8)
ax2.set_xlabel(r"number of rays $q$")
ax2.set_ylabel(r"$\sigma_M(H)/\sigma_1(H)$")
ax2.set_title(r"(b) resolved signal singular value")
ax2.plot([41], [sigma41_ratio], "kx", ms=5.5, mew=1.0)
ax2.set_xlim(1, 42)
ax2.set_xticks([1, 10, 20, 30, 40])
ax2.set_yticks([1e-2, 1e-1, 1e0])
ax2.grid(True, which="major", lw=0.3, color="0.85")

fig.subplots_adjust(left=0.11, right=0.985, bottom=0.20, top=0.88, wspace=0.34)
fig.savefig(OUT / "esprit_capacity_N81_reproduced.pdf")
plt.close(fig)

print("N=81, balanced Hankel 41x41; theorem permits M <= 40")
for q in [19, 20, 25, 30, 35, 39, 40]:
    r = rows[q - 1]
    print(f"M={q:2d}: max angle error={r[1]:.3e} deg, sigma_M/sigma_1={r[3]:.3e}")
print(f"M=41 forced outside theorem: max angle error={maxerr41:.6f} deg, "
      f"mean angle error={meanerr41:.6f} deg, sigma_41/sigma_1={sigma41_ratio:.6e}, "
      f"node modulus range=[{mod41.min():.6f},{mod41.max():.6f}], shift rank={shift_rank41}/41")

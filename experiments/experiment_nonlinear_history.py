"""Nonlinear convergence history for the two-direction crossing-wave solve.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This experiment records the skeleton objective, the error in the two recovered
angles, and the size of the accepted angular update.  It is the data source for
the nonlinear convergence-history figure in the manuscript.

Run
---

    python -m experiments.experiment_nonlinear_history

Outputs
-------
``results/two_plane_nonlinear_history.csv``
``results/two_plane_nonlinear_convergence.png``
``results/two_plane_nonlinear_convergence.pdf``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from core.direction_adaptive import l2_error, physical_skeleton_residual
from experiments.experiment_crossing_waves import residual_vector, solve_with_angles

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

EXACT_ANGLES = np.deg2rad([37.0, 128.0])
INITIAL_ANGLES = np.array([0.3, 2.5])


def periodic_difference(a, b):
    """Return the componentwise angular difference in ``[-pi,pi)``."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    history = []
    previous = [None]

    def record_iteration(x, *args, **kwargs):
        angles = np.mod(np.asarray(x, dtype=float), 2.0 * np.pi)
        state = solve_with_angles(angles, nq=10)
        objective = physical_skeleton_residual(state) ** 2

        direction_error = periodic_difference(angles, EXACT_ANGLES)
        rms_direction_error = np.degrees(
            np.linalg.norm(direction_error) / np.sqrt(2.0)
        )

        if previous[0] is None:
            rms_update = np.nan
        else:
            update = periodic_difference(angles, previous[0])
            rms_update = np.degrees(np.linalg.norm(update) / np.sqrt(2.0))

        history.append(
            {
                "iteration": len(history),
                "J": objective,
                "rms_direction_error_deg": rms_direction_error,
                "rms_angle_update_deg": rms_update,
                "theta1_deg": np.degrees(angles[0]) % 360.0,
                "theta2_deg": np.degrees(angles[1]) % 360.0,
            }
        )
        previous[0] = angles.copy()

    result = least_squares(
        residual_vector,
        INITIAL_ANGLES,
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=120,
        diff_step=2.0e-5,
        callback=record_iteration,
    )

    final_angles = np.mod(result.x, 2.0 * np.pi)
    final_state = solve_with_angles(final_angles, nq=20)
    print("Nonlinear convergence history")
    print("nfev:", result.nfev)
    print("J:", physical_skeleton_residual(final_state) ** 2)
    print("relative L2 error:", l2_error(final_state, nsub=14))
    print("angles (degrees):", np.degrees(final_angles) % 360.0)

    frame = pd.DataFrame(history)
    csv_path = RESULTS / "two_plane_nonlinear_history.csv"
    frame.to_csv(csv_path, index=False)

    # The manuscript figures use explicitly labeled ticks and visible grids.
    plt.figure(figsize=(6.8, 4.8))
    plt.semilogy(
        frame["iteration"],
        frame["J"],
        marker="o",
        label=r"skeleton residual $\mathcal{J}$",
    )
    plt.semilogy(
        frame["iteration"],
        frame["rms_direction_error_deg"],
        marker="s",
        label="RMS direction error (degrees)",
    )
    ax = plt.gca()
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0, 100.0])
    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4)
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Residual / angle-error scale")
    ax.legend()
    plt.tight_layout()

    png_path = RESULTS / "two_plane_nonlinear_convergence.png"
    pdf_path = RESULTS / "two_plane_nonlinear_convergence.pdf"
    plt.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()

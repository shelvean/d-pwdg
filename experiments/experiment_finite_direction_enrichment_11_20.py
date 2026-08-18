"""Continue residual-based ENRICH--MOVE from M=11 to M=20.

Author: Shelvean Kapita
Date: August 2026

The starting M=10 state is read from the polished continuation file.  The birth
step uses a 72-angle residual-only dictionary; this refines the search grid and
is not a cap on the number of active plane waves.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from core.direction_adaptive import (
    graph_data,
    l2_error,
    physical_skeleton_residual,
    physical_skeleton_residual_vector,
)
from experiments.experiment_finite_direction_capacity import (
    FinitePlaneSum,
    TRUE_DEG,
    circular_match_error,
    state_for,
)

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def residual_vector(m: int, angles, nq: int = 16):
    wrapped = np.mod(angles, 2.0 * np.pi)
    state = state_for(m, wrapped, nq=nq)
    return physical_skeleton_residual_vector(state)


def choose_birth(m: int, previous_angles, n_candidates: int = 72):
    """Select the new direction by the current skeleton residual only."""
    candidates = np.mod(
        np.deg2rad(2.5) + 2.0 * np.pi * np.arange(n_candidates) / n_candidates,
        2.0 * np.pi,
    )
    scored = []

    for candidate in candidates:
        separation = np.min(
            np.abs(
                np.angle(np.exp(1j * (candidate - previous_angles)))
            )
        )
        if separation < np.deg2rad(2.0):
            continue

        state = state_for(m, np.r_[previous_angles, candidate], nq=12)
        objective = physical_skeleton_residual(state) ** 2
        scored.append((objective, candidate))

    return min(scored, key=lambda item: item[0])


def main():
    base = pd.read_csv(RESULTS / "finite_direction_enrichment_polished.csv")
    previous_angles = np.deg2rad(
        [float(x) for x in str(base.iloc[-1].found_angles_deg).split(";")]
    )

    # Sorting the previous directions is harmless because the local basis is
    # invariant under permutations of the active rays.
    rows = []

    for m in range(11, 21):
        birth_objective, candidate = choose_birth(m, previous_angles)
        start = np.r_[previous_angles, candidate]
        tic = time.time()

        # First nonlinear solve: moderate quadrature and finite-difference step.
        optimum = least_squares(
            lambda z: residual_vector(m, z, nq=16),
            start,
            method="trf",
            x_scale=np.ones(m),
            ftol=1.0e-13,
            xtol=1.0e-13,
            gtol=1.0e-13,
            max_nfev=180,
            diff_step=1.0e-6,
        )

        # Polish the converged state using higher quadrature and a smaller
        # finite-difference step.  The approximation space is unchanged.
        polished = least_squares(
            lambda z: residual_vector(m, z, nq=22),
            optimum.x,
            method="trf",
            x_scale=np.ones(m),
            ftol=1.0e-15,
            xtol=1.0e-15,
            gtol=1.0e-15,
            max_nfev=100,
            diff_step=2.0e-7,
        )

        angles = np.mod(polished.x, 2.0 * np.pi)
        state = state_for(m, angles, nq=28)
        mean_error, max_error = circular_match_error(
            angles,
            FinitePlaneSum(m).angles,
        )
        conditioning = graph_data(state.A)

        uniform_angles = np.deg2rad(11.0) + 2.0 * np.pi * np.arange(m) / m
        uniform_state = state_for(m, uniform_angles, nq=20)

        row = {
            "M": m,
            "p": m,
            "dofs": state.ndof,
            "birth_deg": float(np.rad2deg(candidate) % 360.0),
            "birth_J": birth_objective,
            "nfev": optimum.nfev,
            "nfev_polish": polished.nfev,
            "adaptive_L2": l2_error(state, nsub=18),
            "adaptive_J": physical_skeleton_residual(state) ** 2,
            "mean_angle_error_deg": mean_error,
            "max_angle_error_deg": max_error,
            "uniform_L2": l2_error(uniform_state, nsub=14),
            "cond_raw": conditioning["cond_raw"],
            "cond_gr": conditioning["cond_gr"],
            "elapsed_sec": time.time() - tic,
            "found_angles_deg": ";".join(
                f"{x:.9f}" for x in np.sort(np.rad2deg(angles) % 360.0)
            ),
            "true_angles_deg": ";".join(
                f"{x:.1f}" for x in np.sort(TRUE_DEG[:m])
            ),
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(
            RESULTS / "finite_direction_enrichment_11_20.csv",
            index=False,
        )
        print(row, flush=True)
        previous_angles = angles


if __name__ == "__main__":
    main()

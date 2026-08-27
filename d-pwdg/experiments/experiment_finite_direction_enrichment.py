"""Residual-based ENRICH--MOVE continuation for M=1,...,10.

Author: Shelvean Kapita
Date: August 2026

The new direction is chosen from a fixed residual-only angular dictionary.  The
exact directions are not used by ENRICH or MOVE; they are used afterward only
for error diagnostics.
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
    K,
    TRUE_DEG,
    circular_match_error,
    state_for,
)

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def residual_vector(m: int, angles):
    try:
        return physical_skeleton_residual_vector(state_for(m, angles, nq=10))
    except Exception:
        return np.full(5000, 1.0e4)


def optimize_angles(m: int, start, max_nfev: int = 160):
    optimum = least_squares(
        lambda z: residual_vector(m, z),
        np.asarray(start),
        method="trf",
        x_scale="jac",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=max_nfev,
        diff_step=2.0e-5,
    )
    angles = np.mod(optimum.x, 2.0 * np.pi)
    state = state_for(m, angles, nq=20)
    return optimum, angles, state


def choose_birth(m: int, previous_angles, n_candidates: int = 36):
    """Choose one new ray by testing a fixed angular dictionary."""
    candidates = np.mod(
        np.deg2rad(7.0) + 2.0 * np.pi * np.arange(n_candidates) / n_candidates,
        2.0 * np.pi,
    )
    scored = []

    for candidate in candidates:
        if len(previous_angles):
            separation = np.min(
                np.abs(
                    np.angle(
                        np.exp(1j * (candidate - np.asarray(previous_angles)))
                    )
                )
            )
            if separation < np.deg2rad(5.0):
                continue

        trial_angles = np.r_[previous_angles, candidate]
        try:
            state = state_for(m, trial_angles, nq=10)
            objective = physical_skeleton_residual(state) ** 2
        except Exception:
            objective = np.inf

        scored.append((objective, candidate))

    scored.sort(key=lambda item: item[0])
    return scored[0]


def main():
    rows = []
    previous_angles = np.array([], dtype=float)

    for m in range(1, 11):
        if m == 1:
            start = np.array([np.deg2rad(11.0)])
            birth_deg = np.nan
            birth_objective = np.nan
        else:
            birth_objective, candidate = choose_birth(m, previous_angles, 36)
            birth_deg = float(np.rad2deg(candidate) % 360.0)
            start = np.r_[previous_angles, candidate]

        tic = time.time()
        optimum, angles, state = optimize_angles(m, start)
        elapsed = time.time() - tic

        adaptive_error = l2_error(state, nsub=14)
        adaptive_objective = physical_skeleton_residual(state) ** 2
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
            "birth_deg": birth_deg,
            "birth_J": birth_objective,
            "nfev": optimum.nfev,
            "adaptive_L2": adaptive_error,
            "adaptive_J": adaptive_objective,
            "mean_angle_error_deg": mean_error,
            "max_angle_error_deg": max_error,
            "uniform_L2": l2_error(uniform_state, nsub=14),
            "cond_raw": conditioning["cond_raw"],
            "cond_gr": conditioning["cond_gr"],
            "elapsed_sec": elapsed,
            "found_angles_deg": ";".join(
                f"{x:.6f}" for x in np.sort(np.rad2deg(angles) % 360.0)
            ),
            "true_angles_deg": ";".join(
                f"{x:.1f}" for x in np.sort(TRUE_DEG[:m])
            ),
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(
            RESULTS / "finite_direction_enrichment.csv",
            index=False,
        )
        print(row, flush=True)
        previous_angles = angles


if __name__ == "__main__":
    main()

"""Finite-direction capacity experiment for direction-adaptive PWDG.

Author
------
Shelvean Kapita
Department of Mathematics, Texas A&M University

Date
----
August 2026

This script studies a nested exact solution containing up to twenty plane-wave
components.  For each M, the adaptive and uniform spaces use the same number of
coefficients.  The exact directions are used only to generate boundary data
and to measure the recovered angles after the nonlinear solve.

The implementation is intentionally direct: every trial direction state is
followed by a PWDG Galerkin solve, then the physical skeleton residual is
returned to SciPy's nonlinear least-squares solver.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment

from core.direction_adaptive import (
    PWDGDictionary,
    graph_data,
    l2_error,
    physical_skeleton_residual,
    physical_skeleton_residual_vector,
    q_real,
)
from core.pwdg import square_mesh

__author__ = "Shelvean Kapita"
__date__ = "August 2026"

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

KAPPA = 12.0
MESH_N = 2  # 2 x 2 squares, each split into two triangles: 8 elements.

TRUE_DIRECTIONS_DEG = np.array(
    [
        17.0,
        123.0,
        251.0,
        68.0,
        191.0,
        315.0,
        39.0,
        158.0,
        286.0,
        340.0,
        92.0,
        224.0,
        5.0,
        145.0,
        273.0,
        327.0,
        54.0,
        178.0,
        235.0,
        301.0,
    ]
)

# Equal-modulus coefficients with deterministic phases.  The phase offsets
# prevent the manufactured sum from having an accidental symmetry.
AMPLITUDE_PHASES = np.array(
    [
        0.0,
        0.37,
        -0.51,
        0.81,
        -1.04,
        1.33,
        -1.52,
        0.22,
        1.71,
        -0.93,
        0.58,
        -1.22,
        1.48,
        -0.31,
        0.96,
        -1.73,
        0.11,
        1.18,
        -0.77,
        1.91,
    ]
)
AMPLITUDES = np.exp(1j * AMPLITUDE_PHASES)


class FinitePlaneWaveSum:
    """Exact sum of the first ``m`` waves in the nested direction family."""

    def __init__(self, m: int, kappa: float = KAPPA) -> None:
        self.k = float(kappa)
        self.m = int(m)
        self.angles = np.deg2rad(TRUE_DIRECTIONS_DEG[:m])
        self.amplitudes = AMPLITUDES[:m]

    def wavenumber(self, y):
        return self.k * np.ones_like(np.asarray(y, dtype=float))

    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        value = np.zeros(np.broadcast(x, y).shape, dtype=complex)

        for amplitude, theta in zip(self.amplitudes, self.angles):
            phase = np.cos(theta) * x + np.sin(theta) * y
            value += amplitude * np.exp(1j * self.k * phase)

        return value


# Backward-compatible name used by the enrichment scripts.
FinitePlaneSum = FinitePlaneWaveSum
TRUE_DEG = TRUE_DIRECTIONS_DEG
K = KAPPA
N = MESH_N


def state_for(m: int, angles, nq: int = 10) -> PWDGDictionary:
    """Assemble and solve the PWDG state for one global set of directions."""
    exact = FinitePlaneWaveSum(m)
    _, triangles = square_mesh(MESH_N)

    wave_vectors = np.array(
        [q_real(KAPPA, theta) for theta in np.mod(angles, 2.0 * np.pi)]
    )
    dictionary = [wave_vectors.copy() for _ in range(len(triangles))]

    state = PWDGDictionary(MESH_N, exact, dictionary, nq=nq)
    state.solve()
    return state


def residual_vector(m: int, angles):
    """Return the real skeleton residual used by nonlinear least squares."""
    try:
        state = state_for(m, angles, nq=10)
        return physical_skeleton_residual_vector(state)
    except Exception:
        # A large fixed residual keeps an invalid trial away from the optimizer.
        # This branch is exceptional and therefore does not affect normal speed.
        return np.full(4000, 1.0e4)


def circular_match_error(found, true):
    """Mean and maximum angular matching errors in degrees."""
    distance = np.abs(
        np.angle(np.exp(1j * (found[:, None] - true[None, :])))
    )
    row, col = linear_sum_assignment(distance)
    matched = np.rad2deg(distance[row, col])
    return float(matched.mean()), float(matched.max())


def run(m: int, max_nfev: int = 220):
    """Run the equal-dimension uniform/adaptive comparison for one ``M``."""
    exact = FinitePlaneWaveSum(m)

    # Same nominal space size for the uniform and adaptive calculations.  The
    # 11-degree offset avoids an accidental exact hit by the uniform grid.
    start = np.mod(
        np.deg2rad(11.0) + 2.0 * np.pi * np.arange(m) / m,
        2.0 * np.pi,
    )

    uniform_state = state_for(m, start, nq=20)
    uniform_error = l2_error(uniform_state, nsub=14)
    uniform_objective = physical_skeleton_residual(uniform_state) ** 2

    tic = time.time()
    optimum = least_squares(
        lambda z: residual_vector(m, z),
        start,
        method="trf",
        x_scale="jac",
        ftol=2.0e-12,
        xtol=2.0e-12,
        gtol=2.0e-12,
        max_nfev=max_nfev,
        diff_step=2.0e-5,
    )
    elapsed = time.time() - tic

    angles = np.mod(optimum.x, 2.0 * np.pi)
    adaptive_state = state_for(m, angles, nq=20)
    adaptive_error = l2_error(adaptive_state, nsub=14)
    adaptive_objective = physical_skeleton_residual(adaptive_state) ** 2

    mean_angle_error, max_angle_error = circular_match_error(
        angles,
        exact.angles,
    )
    conditioning = graph_data(adaptive_state.A)

    return {
        "M": m,
        "p": m,
        "dofs": adaptive_state.ndof,
        "uniform_L2": uniform_error,
        "uniform_J": uniform_objective,
        "adaptive_L2": adaptive_error,
        "adaptive_J": adaptive_objective,
        "nfev": optimum.nfev,
        "success": bool(optimum.success),
        "mean_angle_error_deg": mean_angle_error,
        "max_angle_error_deg": max_angle_error,
        "cond_raw": conditioning["cond_raw"],
        "cond_gr": conditioning["cond_gr"],
        "elapsed_sec": elapsed,
        "found_angles_deg": ";".join(
            f"{x:.6f}" for x in np.sort(np.rad2deg(angles) % 360.0)
        ),
        "true_angles_deg": ";".join(
            f"{x:.1f}" for x in np.sort(TRUE_DIRECTIONS_DEG[:m])
        ),
    }


if __name__ == "__main__":
    rows = []
    for m in range(1, 9):
        result = run(m)
        rows.append(result)
        print(result, flush=True)

    pd.DataFrame(rows).to_csv(
        RESULTS / "finite_direction_capacity_M1_M8.csv",
        index=False,
    )
